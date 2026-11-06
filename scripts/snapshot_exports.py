#!/usr/bin/env python3
"""Bind a downloaded GT snapshot to the existing frozen evaluation adapters.

Only creates local manifests and explicitly labelled GT hand sidecars. Never
attaches source GT hands to model generation or tokenizer reconstruction.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--dataset',choices=('AIST++','FineDance','BEAT2'),required=True)
    p.add_argument('--evaluation-split',choices=('all','test'),default='all',help='AIST only: all=TEST20+VAL20')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--metadata',type=Path,default=ROOT/'metadata')
    a=p.parse_args()
    ds='AISTpp' if a.dataset=='AIST++' else a.dataset
    rows=list(map(json.loads,(a.root/ds/'manifest.jsonl').read_text().splitlines()))
    roles=('test','val') if a.dataset=='AIST++' and a.evaluation_split=='all' else ('test',)
    official=list(map(json.loads,(a.metadata/'members.jsonl').read_text().splitlines()))
    ids={r['source_id'] for r in official if r['dataset']==a.dataset and r['split'] in roles}
    if a.dataset=='BEAT2':ids=set(json.loads((a.metadata/'beat_speaker2.json').read_text())['test'])
    selected=[r for r in rows if r['source_id'] in ids]
    assert len(selected)==len(ids) and {r['source_id'] for r in selected}==ids
    assert all(r['split'] in roles and r['fps']==30 for r in selected)
    a.output=a.output.resolve();a.output.mkdir(parents=True,exist_ok=False)
    exports=[]
    for r in selected:
        path=(a.root/r['smplx_path']).resolve()
        assert path.is_relative_to(a.root.resolve()) and sha(path)==r['smplx_sha256']
        e=dict(dataset=a.dataset,source_id=r['source_id'],arm='soma_gt_roundtrip',fps=30,
               smplx_path=str(path),smplx_sha256=r['smplx_sha256'],time_origin_s=r['time_origin_s'],
               axis_convention=r['axis_convention'],units=r['units'],converter_version='native-transport-inverse-v1',
               producer=r['producer'],root_scale=r['root_scale'],shape_encoding='smplx_betas_only',
               model_checkpoint_sha256=None,hand_sidecar_path=None,hand_sidecar_sha256=None)
        if a.dataset=='BEAT2':
            hp=(a.root/r['hands_path']).resolve()
            assert hp.is_relative_to(a.root.resolve()) and sha(hp)==r['hands_sha256']
            with np.load(hp,allow_pickle=False) as z:
                fingers=z['finger_rotvec'].copy()
                np.testing.assert_array_equal(z['source_frame30'],np.arange(r['frames']))
                assert fingers.shape==(r['frames'],90) and np.isfinite(fingers).all()
            sidecar=a.output/(r['source_id']+'.hands.npz')
            np.savez_compressed(sidecar,left_hand_pose=fingers[:,:45],right_hand_pose=fingers[:,45:],
                                arm='soma_gt_roundtrip',source_id=r['source_id'],fps=30,time_origin_s=r['time_origin_s'])
            e.update(hand_sidecar_path=str(sidecar),hand_sidecar_sha256=sha(sidecar))
        exports.append(e)
    with (a.output/'exports.jsonl').open('x') as f:
        for e in exports:f.write(json.dumps(e)+'\n')
    print(json.dumps(dict(dataset=a.dataset,sources=len(exports),arm='soma_gt_roundtrip',exports=str(a.output/'exports.jsonl'))))


if __name__=='__main__':main()
