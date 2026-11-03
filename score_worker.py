"""One interpreter per benchmark: preserve the upstream 'features'/'utils' namespaces."""
import argparse
import copy
import io
import json
from pathlib import Path
import random
import sys
import os
import shutil
from datetime import datetime, timezone
from types import SimpleNamespace
import numpy as np
import torch
from evaluator_common import ROOT, UP, setup, load_file, cwd, cpu_checkpoint_loading, write_json
from source_lock import sha
from contracts import require

def dance(args, rows):
    require(args.dataset == 'FineDance', 'Use aist-smplx for AIST++ SMPL-X30 scoring')
    sys.path[:0] = [str(UP/'LODGE/metric'), str(UP/'LODGE')]
    metric = load_file('official_dance_metric', UP/'LODGE/metric/metrics_finedance.py')
    from dld.data.render_joints.smplfk import SMPLX_Skeleton
    metric.device = 'cpu'
    metric.smplx_model = SMPLX_Skeleton(Jpath=str(UP/'LODGE/data/smplx_neu_J_1.npy'))
    pred, ref = args.output/'prediction', args.output/'reference'
    pred.mkdir(); ref.mkdir()
    inputs = []
    from official_preprocess import finedance_preprocess
    native_preprocessed, crop_receipt = finedance_preprocess(
        {r['source_id']: r['native_path'] for r in rows}, args.output/'native_preprocess')
    for row in rows:
        sid = row['source_id']; native = Path(row['native_path'])
        require(native.is_file(), f'Missing native: {native}')
        require(row['fps'] == 30, 'FineDance native must be30fps')
        x = np.load(native_preprocessed/f'{sid}.npy', allow_pickle=False)
        require(x.ndim == 2 and x.shape[1] in (315, 319, 135, 139) and np.isfinite(x).all(), 'LODGE requires native 315/319 or body135/139; 266D RIC is not its input')
        if args.arm == 'native_gt':
            candidate = x.copy()
        else:
            require(row['prediction_format'] == 'lodge139_preprocessed', 'FineDance requires official-preprocessed139/319D; do not double-crop native data')
            require(row['prediction_native_start_frame'] == crop_receipt[sid]['native_source_start_frame'], 'FineDance prediction crop differs from author preprocessing')
            candidate = np.load(row['prediction_path'], allow_pickle=False)
        require(candidate.ndim == 2 and candidate.shape[1] in (135, 139, 315, 319), 'FineDance adapter format invalid')
        # Freeze the identical source timeline before the official1024-frame cap.
        require(len(candidate) >= min(len(x), 1024), f'{sid}: prediction fails to cover official1024-frame/native-shorter support')
        for target, value in ((ref, x), (pred, candidate)):
            np.save(target/f'{sid}.npy', value)
        inputs.append(dict(source_id=sid, native=str(native), native_sha256=sha(native),
                           prediction_sha256=sha(row['prediction_path']) if args.arm != 'native_gt' else sha(native)))
    calc_and_save_feats = metric.calc_and_save_feats
    calc_and_save_feats(str(ref)); calc_and_save_feats(str(pred))
    expected = {r['source_id']+'.npy' for r in rows}
    for d in (ref, pred):
        for kind in ('kinetic_features', 'manual_features_new'):
            require({p.name for p in (d/kind).iterdir()} == expected, 'Missing/extra dance feature')
            require(all(np.isfinite(np.load(p)).all() for p in (d/kind).iterdir()), 'Nonfinite dance feature')
    scoring_ref = ref
    reference_scope = 'exactly manifest native inputs; diagnostic paired reference, not presumed original precomputed paper reference pool'
    if args.reference_features:
        scoring_ref = args.reference_features.resolve()
        receipt = json.loads((scoring_ref/'reference_receipt.json').read_text())
        require(receipt['dataset'] == args.dataset, 'Wrong reference benchmark')
        for name, expected_sha in receipt['files'].items():
            require(sha(scoring_ref/name) == expected_sha, 'Frozen GT reference feature changed')
        expected_files = set(receipt['files'])
        actual_files = {str(p.relative_to(scoring_ref)) for k in ('kinetic_features','manual_features_new') for p in (scoring_ref/k).iterdir()}
        require(actual_files == expected_files, 'Unexpected or missing reference features')
        reference_scope = receipt
    scores = metric.quantized_metrics(str(pred), str(scoring_ref))
    return dict(metrics=scores, inputs=inputs,
                reference_scope=reference_scope,
                feature_protocol=None,
                upstream_feature_cap=1024,
                identity_features_equal=all(np.array_equal(np.load(pred/k/p.name), np.load(p)) for k in ('kinetic_features', 'manual_features_new') for p in (ref/k).iterdir()))

def humanml(args, rows):
    sys.path.insert(0, str(UP/'T2M'))
    from data.dataset import Text2MotionDatasetV2, collate_fn
    from utils.word_vectorizer import WordVectorizer
    from networks.evaluator_wrapper import EvaluatorModelWrapper
    official = load_file('official_t2m_final', UP/'T2M/final_evaluations.py')
    root = (ROOT/'data/HumanML3D')
    from hml_sources import stage_hml_sources
    native_dir = stage_hml_sources(rows, args.output/'native263')
    split = args.output/'split.txt'; split.write_text(''.join(r['source_id']+'\n' for r in rows))
    candidate_dir = args.output/'motion263'; candidate_dir.mkdir()
    inputs = []
    for r in rows:
        native = np.load(r['native_path'], allow_pickle=False)
        if args.arm != 'generated':
            candidate = native if args.arm == 'native_gt' else np.load(r['prediction_path'], allow_pickle=False)
            require(candidate.shape == native.shape and candidate.ndim == 2 and candidate.shape[1] == 263 and np.isfinite(candidate).all(), f"{r['source_id']}: 263D length/finite contract")
            np.save(candidate_dir/f"{r['source_id']}.npy", candidate)
        text = root/'texts'/f"{r['source_id']}.txt"
        require(text.is_file(), f'Missing official caption: {text}')
        inputs.append(dict(source_id=r['source_id'], native_sha256=sha(r['native_path']), text_sha256=sha(text),
                           prediction_sha256=sha(r['prediction_path']) if args.arm not in ('native_gt','generated') else sha(r['native_path']) if args.arm == 'native_gt' else None))
    from hml_generated import expected_entry_count
    expected_entries = expected_entry_count(rows, root/'texts')
    mean, std = (np.load(ROOT/'assets/t2m'/f'{name}.npy') for name in ('mean', 'std'))
    wv = WordVectorizer(str(UP/'T2M/glove'), 'our_vab')
    opt = SimpleNamespace(dataset_name='t2m', max_motion_length=196, unit_length=4, max_text_len=20,
                          motion_dir=str(native_dir), text_dir=str(root/'texts'),
                          checkpoints_dir=str(ROOT/'assets/t2m'), device=torch.device('cpu'),
                          dim_movement_enc_hidden=512, dim_movement_latent=512)
    # Same seed at construction preserves identical official caption-subsequence identities.
    random.seed(args.seed); np.random.seed(args.seed)
    native_ds = Text2MotionDatasetV2(opt, mean, std, str(split), wv)
    require(len(native_ds) == expected_entries, 'Official loader silently discarded expected HML entries')
    if args.arm == 'generated':
        plan_receipt = json.loads((args.condition_plan.parent/'receipt.json').read_text())
        require(plan_receipt['conditions_sha256'] == sha(args.condition_plan), 'Condition plan differs from its frozen receipt')
        require(plan_receipt['seed'] == args.seed and plan_receipt['replication_count'] == args.replications, 'Condition sampler seeds/repetitions differ')
        source_inputs = [{k:r[k] for k in ('source_id','native_sha256','text_sha256')} for r in inputs]
        require(plan_receipt['inputs'] == source_inputs, 'Native source/caption bytes differ from condition-plan inputs')
    if args.arm != 'generated':
        candopt = copy.copy(opt); candopt.motion_dir = str(candidate_dir)
        random.seed(args.seed); np.random.seed(args.seed)
        cand_ds = Text2MotionDatasetV2(candopt, mean, std, str(split), wv)
        require(native_ds.name_list == cand_ds.name_list and np.array_equal(native_ds.length_arr, cand_ds.length_arr), 'Official HML dataset entries differ')
    with cpu_checkpoint_loading():
        official.eval_wrapper = EvaluatorModelWrapper(opt)
    official.diversity_times = 300; official.mm_num_times = 10
    official.replication_times = args.replications
    def loader(ds):
        return torch.utils.data.DataLoader(ds, batch_size=32, num_workers=0, shuffle=True, drop_last=True, collate_fn=collate_fn)
    repetitions = []
    require(len(native_ds)//32*32 > 300, 'Official Diversity300 needs >300 evaluated motions; do not lower its setting for smoke')
    for rep in range(args.replications):
        seed = args.seed+rep; torch.manual_seed(seed); random.seed(seed); np.random.seed(seed)
        # Call the author's actual metric driver, not a rewritten formula.
        with (args.output/f'rep{rep:02d}.log').open('w') as log:
            if args.arm == 'generated':
                from hml_generated import loaders_for_rep
                (candidate_loader, mm_loader), generated_count = loaders_for_rep(args.generated_records, args.condition_plan, rep, native_ds, opt, wv, mean, std)
            else:
                candidate_loader = loader(cand_ds)
            matching, retrieval, activations = official.evaluate_matching_score({'native_gt': loader(native_ds), 'candidate_'+args.arm: candidate_loader}, log)
            fid = official.evaluate_fid(loader(native_ds), activations, log)
            div = official.evaluate_diversity(activations, log)
            mm = official.evaluate_multimodality({'candidate_'+args.arm: mm_loader}, log) if args.arm == 'generated' else None
        repetitions.append(dict(Matching_Score=matching, R_precision=retrieval, FID=fid, Diversity=div, Multimodality=mm))
    summary = {}
    for metric_name in repetitions[0]:
        if repetitions[0][metric_name] is None: continue
        summary[metric_name] = {}
        for arm in repetitions[0][metric_name]:
            mean_value, confidence = official.get_metric_statistics(np.asarray([r[metric_name][arm] for r in repetitions]))
            summary[metric_name][arm] = dict(mean=mean_value, confidence_interval_95=confidence)
    return dict(repetitions=repetitions, metric_summary=summary, inputs=inputs, metrics_scope='official T2M metric driver, encoder, normalization, caption sampler, batch32/drop_last, crop4, Diversity300',
                dataset_entries=len(native_ds), evaluated_per_replication=len(native_ds)//32*32,
                Multimodality='see repetitions' if args.arm == 'generated' else None,
                Multimodality_reason='original100x30 protocol; draw10 pairs' if args.arm == 'generated' else 'GT reconstruction is not multimodal sampling',
                generated_records_sha256=sha(args.generated_records) if args.generated_records else None,
                condition_plan_sha256=sha(args.condition_plan) if args.condition_plan else None,
                weights=json.loads((ROOT/'assets/t2m_author_verification.json').read_text()) if (ROOT/'assets/t2m_author_verification.json').exists()
                        else 'UNVERIFIED_LOCAL_TENSORS: diagnostic only; must not enter a formal result')

def beat(args, rows):
    from beat2024 import run
    return run(args, rows)

def parse_args(argv=None):
    p = argparse.ArgumentParser(); p.add_argument('--dataset', choices=('HumanML3D','AIST++','FineDance','BEAT2'), required=True)
    p.add_argument('--manifest', type=Path, required=True); p.add_argument('--output', type=Path, required=True)
    p.add_argument('--arm', choices=('native_gt','native_format_control','time_resample_control','previous_smplx_gt','soma_gt_roundtrip','tokenizer_roundtrip','generated'), required=True)
    p.add_argument('--split', choices=('train','val','test'), required=True); p.add_argument('--limit', type=int)
    p.add_argument('--seed', type=int, default=112); p.add_argument('--replications', type=int, default=20)
    p.add_argument('--generated-records', type=Path)
    p.add_argument('--condition-plan', type=Path)
    p.add_argument('--reference-features', type=Path)
    p.add_argument('--membership-profile', choices=('all_official_sources','emage_author_speaker2'))
    args = p.parse_args(argv)
    if args.dataset == 'AIST++':
        p.error('Use evaluate.py aist-smplx: the only current AIST++ protocol is SMPL-X30 / TRAIN952')
    args.beat_profile = 'emage2024_bc_l1' if args.dataset == 'BEAT2' else None
    if args.membership_profile is None:
        args.membership_profile = 'emage_author_speaker2' if args.dataset == 'BEAT2' else 'all_official_sources'
    if args.dataset == 'BEAT2' and args.membership_profile != 'emage_author_speaker2':
        p.error('BEAT2 uses the fixed EMAGE2024 speaker2 protocol; previous full-CSV scoring is archived in legacy/')
    if args.dataset != 'BEAT2' and args.membership_profile != 'all_official_sources':
        p.error('speaker2 membership only applies to BEAT2')
    return args


def main():
    args = parse_args(); args.output = args.output.resolve(); args.manifest = args.manifest.resolve(); setup()
    require(args.replications > 0, 'Replications must be positive')
    require(not (args.dataset == 'HumanML3D' and args.arm == 'generated' and not (args.generated_records and args.condition_plan)),
            'Generated HML requires caption-specific frozen generation inputs; source-wise GT reconstruction loader must not be called a formal generation protocol')
    if args.generated_records: args.generated_records = args.generated_records.resolve()
    if args.condition_plan: args.condition_plan = args.condition_plan.resolve()
    if args.dataset == 'HumanML3D' and args.arm == 'generated':
        for path in (args.generated_records, args.condition_plan):
            records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
            require({r['replication'] for r in records} == set(range(args.replications)), 'Missing/extra generation replications')
        outputs = [json.loads(line) for line in args.generated_records.read_text().splitlines() if line.strip()]
        require(len({r['model_checkpoint_sha256'] for r in outputs}) == 1, 'Mixed model checkpoints across replications')
    if args.reference_features: args.reference_features = args.reference_features.resolve()
    from contracts import read_manifest
    rows = [r for r in read_manifest(args.manifest) if r['dataset'] == args.dataset and r['split'] == args.split]
    require(bool(rows), 'No requested sources')
    if args.limit is not None:
        require(args.split == 'train' and args.limit > 0, 'Subset diagnostics only allowed on TRAIN')
        rows = rows[:args.limit]
    from test_protocol import validate_test_protocol
    input_protocol = validate_test_protocol(args, rows)
    if args.dataset == 'BEAT2':
        from beat2024 import verify_profile
        verify_profile()
    args.output.mkdir(parents=True, exist_ok=False)
    for name in ('PROTOCOL.md', 'sources.lock.json', 'assets.lock.json'):
        shutil.copy2(ROOT/name, args.output/name)
    code_paths = [ROOT/name for name in ('score_worker.py', 'contracts.py', 'evaluator_common.py',
                  'official_preprocess.py', 'hml_generated.py', 'hml_sources.py', 'test_protocol.py', 'aist30.py', 'evaluate.py', 'source_lock.py', 'assets_lock.py')]
    if args.dataset == 'BEAT2':
        code_paths.append(ROOT/'beat2024.py')
        from beat2024 import ASSETS
        shutil.copy2(ASSETS/'receipt.json', args.output/'beat_profile_receipt.json')
        shutil.copy2(ASSETS/'source/utils/metric.py', args.output/'beat_author_metric.py')
    (args.output/'code').mkdir()
    for path in code_paths: shutil.copy2(path, args.output/'code'/path.name)
    record = dict(status='RUNNING', arguments=vars(args) | {'manifest': str(args.manifest), 'output': str(args.output), 'generated_records': str(args.generated_records) if args.generated_records else None, 'condition_plan': str(args.condition_plan) if args.condition_plan else None, 'reference_features': str(args.reference_features) if args.reference_features else None},
                  started_at=datetime.now(timezone.utc).isoformat(), root_pid=os.getpid(),
                  cpu_affinity=sorted(os.sched_getaffinity(0)), cuda_visible_devices=os.environ['CUDA_VISIBLE_DEVICES'],
                  manifest_sha256=sha(args.manifest), samples=len(rows), formal_official_result=False,
                  input_protocol=input_protocol,
                  protocol_sha256=sha(ROOT/'PROTOCOL.md'), sources_lock_sha256=sha(ROOT/'sources.lock.json'),
                  assets_lock_sha256=sha(ROOT/'assets.lock.json') if (ROOT/'assets.lock.json').exists() else None,
                  own_code_sha256={p.name: sha(p) for p in code_paths})
    write_json(args.output/'run.json', record)
    try:
        result = humanml(args, rows) if args.dataset == 'HumanML3D' else beat(args, rows) if args.dataset == 'BEAT2' else dance(args, rows)
        result.update(dataset=args.dataset, arm=args.arm, split=args.split, samples=len(rows), status='SCORER_RUN_FINISHED',
                      input_protocol=input_protocol,
                      formal_official_result=False, acceptance='scorer output only; source/paper protocol, full coverage and transform fidelity require separate acceptance')
        if args.dataset == 'BEAT2':
            result['beat_profile'] = args.beat_profile
        write_json(args.output/'result.json', result)
        record.update(status='SCORER_RUN_FINISHED', finished_at=datetime.now(timezone.utc).isoformat())
        write_json(args.output/'run.json', record)
        print(json.dumps({'status': result['status'], 'result': str(args.output/'result.json')}))
    except BaseException as error:
        write_json(args.output/'failure.json', dict(status='PIPELINE_FAILURE', error=repr(error), automatic_retry=False))
        record.update(status='PIPELINE_FAILURE', finished_at=datetime.now(timezone.utc).isoformat(), error=repr(error))
        write_json(args.output/'run.json', record)
        raise

if __name__ == '__main__': main()
