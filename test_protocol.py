"""Frozen source membership; BEAT2 uses only the accepted author speaker2 protocol."""
import json
from pathlib import Path
from contracts import read_manifest,require
from evaluator_common import ROOT
from source_lock import sha

def validate_test_protocol(args,rows):
    require(args.dataset!='AIST++','Use aist-smplx: actual SMPL-X30 geometry and full1363 reference')
    if args.dataset == 'BEAT2':
        require(args.membership_profile == 'emage_author_speaker2', 'BEAT2 uses only the frozen speaker2 protocol')
    if args.split!='test': return dict(scope='TRAIN/VAL diagnostic; no formal TEST coverage claim')
    profile=getattr(args,'membership_profile','all_official_sources')
    require(profile in ('all_official_sources','emage_author_speaker2'),'Unknown profile')
    canonical=ROOT/'membership_v1/manifest.jsonl'
    if profile=='emage_author_speaker2':
        require(args.dataset=='BEAT2','speaker2 only applies to BEAT2')
        canonical=ROOT/'protocol_profiles_v1/emage_author_speaker2.jsonl'
    if args.dataset=='HumanML3D': require(args.replications==20,'Standard HML TEST requires20 repetitions')
    expected=[r for r in read_manifest(canonical) if r['dataset']==args.dataset and r['split']=='test']
    require([r['source_id'] for r in rows]==[r['source_id'] for r in expected],'Incomplete/reordered TEST')
    for r,e in zip(rows,expected): require(all(r.get(k)==v for k,v in e.items()),'Source metadata changed: '+r['source_id'])
    if args.dataset=='FineDance':
        ref=ROOT/'assets/lodge_published_reference_v1'
        require(args.reference_features and Path(args.reference_features).resolve()==ref.resolve(),'Use the frozen published202-source LODGE reference')
        require((ref/'reference_receipt.json').is_file(),'Published reference not frozen')
    return dict(scope='COMPLETE_FROZEN_PROFILE_INPUTS',membership_profile=profile,sources=len(expected),canonical_manifest_sha256=sha(canonical),
                paper_inference_membership_verified=False if args.dataset=='FineDance' else None)
