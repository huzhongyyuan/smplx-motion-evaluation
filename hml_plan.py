"""Freeze original T2M captions/crops for known-duration external generation.

This is explicitly not CompV6's model-specific learned length estimator. All
compared models must use the same known-duration plan.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import random
import sys
from types import SimpleNamespace
import numpy as np
import torch
from contracts import read_manifest, require
from evaluator_common import ROOT, UP, setup, write_json
from source_lock import sha
from hml_generated import expected_entry_count

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--split', choices=('train','val','test'), required=True)
    p.add_argument('--limit', type=int)
    p.add_argument('--seed', type=int, default=112)
    p.add_argument('--replications', type=int, default=20)
    args = p.parse_args(); setup()
    rows = [r for r in read_manifest(args.manifest) if r['dataset'] == 'HumanML3D' and r['split'] == args.split]
    require(rows and args.replications > 0, 'Empty plan or invalid replications')
    if args.limit is not None:
        require(args.split == 'train' and args.limit > 0, 'Partial diagnostic plans restricted toTRAIN')
        rows = rows[:args.limit]
    from test_protocol import validate_test_protocol
    input_protocol = validate_test_protocol(SimpleNamespace(dataset='HumanML3D',split=args.split,
        replications=args.replications,membership_profile='all_official_sources',reference_features=None),rows)
    output = args.output.resolve(); output.mkdir(parents=True, exist_ok=False)
    split = output/'split.txt'; split.write_text(''.join(r['source_id']+'\n' for r in rows))
    sys.path.insert(0, str(UP/'T2M'))
    from data.dataset import Text2MotionDatasetV2
    from utils.word_vectorizer import WordVectorizer
    mean, std = (np.load(ROOT/'assets/t2m'/f'{n}.npy') for n in ('mean','std'))
    data_root = (ROOT/'data/HumanML3D')
    from hml_sources import stage_hml_sources
    native_dir = stage_hml_sources(rows, output/'native263')
    opt = SimpleNamespace(dataset_name='t2m', max_motion_length=196, unit_length=4, max_text_len=20,
                          motion_dir=str(native_dir), text_dir=str(data_root/'texts'))
    vectorizer = WordVectorizer(str(UP/'T2M/glove'), 'our_vab')
    random.seed(args.seed); np.random.seed(args.seed)
    dataset = Text2MotionDatasetV2(opt, mean, std, str(split), vectorizer)
    require(len(dataset) == expected_entry_count(rows, data_root/'texts'), 'Original condition loader silently dropped expected entries')
    require(len(dataset)//32*32 > 300, 'Diversity300 needs more than300 evaluated entries')
    conditions = []
    for rep in range(args.replications):
        seed = args.seed+rep
        torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
        mm = set(np.random.choice(len(dataset), 100, replace=False).tolist())
        loader = torch.utils.data.DataLoader(dataset, batch_size=1, num_workers=0, shuffle=True)
        # Freeze the RandomSampler order and original Dataset.__getitem__.
        # Do not batch the pre-existing length-sorted name_list directly.
        for order, idx in enumerate(loader.sampler):
            _, _, caption, cap_len, _, length, tokens = dataset[idx]
            name = dataset.name_list[dataset.pointer+idx]; length = int(length)
            require(36 <= length <= 196 and length % 4 == 0, 'Invalid original sampled duration')
            conditions.append(dict(replication=rep, order_index=order, entry_name=name,
                caption=caption, tokens=tokens.split('_'), cap_len=int(cap_len),
                length20=length, length_policy='known_gt_duration_original_T2M_dataset_crop',
                repeats=30 if order in mm else 1, condition_seed=seed,
                generation_seed_base=seed*1000000+order*30))
    target = output/'conditions.jsonl'
    target.write_text(''.join(json.dumps(r)+'\n' for r in conditions))
    write_json(output/'receipt.json', dict(status='CONDITIONS_FROZEN_NO_MODEL_GENERATION',
        created_at=datetime.now(timezone.utc).isoformat(), manifest_path=str(args.manifest.resolve()), manifest_sha256=sha(args.manifest),
        conditions_sha256=sha(target), dataset_entries=len(dataset), samples=len(rows),
        seed=args.seed,
        inputs=[dict(source_id=r['source_id'], native_sha256=sha(r['native_path']),
                     text_sha256=sha(data_root/'texts'/(r['source_id']+'.txt'))) for r in rows],
        replication_count=args.replications, expected_outputs=sum(r['repeats'] for r in conditions),
        input_protocol=input_protocol,
        generation_order='frozen torch RandomSampler; original T2M Dataset caption/crop sampling',
        length_policy='known_gt_duration_original_T2M_dataset_crop',
        comparison_requirement='all methods use this known-duration plan; not CompV6 learned length prediction',
        sources_lock_sha256=sha(ROOT/'sources.lock.json'), code_sha256=sha(__file__)))
    print(str(target), flush=True)

if __name__ == '__main__': main()
