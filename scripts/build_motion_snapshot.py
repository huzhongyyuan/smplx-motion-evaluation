#!/usr/bin/env python3
"""Build a source-pinned, portable three-dataset motion release on CPU.

Inputs are read-only. SOMA motion arrays are copied unchanged; SMPL-X body uses the
declared producer's matched rotation inverse. Native data supplies only fingers
and shape; native body/root are used for independent diagnostics, never fitting.
"""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='2',
                  OPENBLAS_NUM_THREADS='2', MKL_NUM_THREADS='2')
import argparse
import collections
import csv
import hashlib
import json
import pickle
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def now():
    return datetime.now(timezone.utc).isoformat()


def write(path, value):
    with path.open('x') as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.write('\n')


def global_rot(local, parents):
    result = []
    for j, p in enumerate(parents):
        result.append(local[:, j] if p < 0 else result[p] @ local[:, j])
    return np.stack(result, axis=1)


def row6_matrices(value):
    x, y = value[..., :3], value[..., 3:]
    if np.min(np.linalg.norm(x, axis=-1)) < 1e-6:
        raise ValueError('Degenerate native row6 first axis')
    x = x / np.linalg.norm(x, axis=-1, keepdims=True)
    y = y - (y * x).sum(-1, keepdims=True) * x
    if np.min(np.linalg.norm(y, axis=-1)) < 1e-6:
        raise ValueError('Degenerate native row6 second axis')
    y = y / np.linalg.norm(y, axis=-1, keepdims=True)
    return np.stack((x, y, np.cross(x, y)), axis=-2)


def convert(row, args, native, transport, restored_sources, cached_sources):
    ds, sid = row['dataset'], row['source_id']
    dirname = 'AISTpp' if ds == 'AIST++' else ds
    dest = args.output / dirname
    source = args.unified / row['motion_path']
    source_sha = sha(source)
    assert source_sha == row['motion_sha256'], ('Source SHA changed', ds, sid)
    with np.load(source, allow_pickle=False) as z:
        arrays = {k: z[k].copy() for k in z.files}
    local, root = arrays['local_rot_mats'], arrays['root_positions']
    n = len(root)
    assert n == row['frames30'] and float(arrays['fps']) == 30
    scale = float(arrays.get('root_scale', 1.))
    producer = str(arrays.get('conversion_version', 'soma_v4_fd_delivery'
                            if ds == 'FineDance' else ''))
    producer_evidence = 'embedded conversion_version' if 'conversion_version' in arrays else 'declared FineDance delivery contract'
    if not producer and (ds, sid) in restored_sources:
        restored = restored_sources[ds, sid]
        assert restored['motion_sha256'] == source_sha and row['origin'] == 'est_restore_v1'
        assert 'observed_frame' in arrays and np.asarray(arrays['observed_frame']).all()
        # This producer is declared by the hash-bound restoration report and
        # audited builder, rather than inferred from the rotation array shape.
        producer = 'soma_native55_transport'
        assert scale == 1.
        producer_evidence = 'legacy native55 restoration report, source SHA matched'
    assert producer in transport.PRODUCERS, (ds, sid, producer)
    # Source frame indices refer to the uncropped source's 30FPS clock.
    clock = np.asarray(arrays.get('source_frame30', np.arange(n)), dtype=np.int64)
    assert clock.shape == (n,) and clock[0] >= 0
    np.testing.assert_array_equal(np.diff(clock), np.ones(n - 1, dtype=np.int64))
    for key in ('timestamps', 'frame_times', 'times_seconds'):
        if key in arrays:
            timing = np.asarray(arrays[key], dtype=np.float64)
            assert timing.shape == (n,) and np.isfinite(timing).all()
            assert np.max(np.abs(timing - clock / 30.)) < 1e-7
    if 'native_frame_range30' in row:
        assert row['native_frame_range30'] == [int(clock[0]), int(clock[-1]) + 1]
    estimated = np.asarray(arrays.get('estimated_frame', np.zeros(n, dtype=bool))).copy()
    assert estimated.shape == (n,)
    original_estimated_count = int(estimated.sum())
    if 'observed_frame' in arrays:
        assert np.asarray(arrays['observed_frame']).shape == (n,) and np.asarray(arrays['observed_frame']).all()
    for key, value in arrays.items():
        if np.issubdtype(value.dtype, np.number):
            assert np.isfinite(value).all(), (ds, sid, key)
        elif value.dtype.kind in 'US':
            assert all(not str(v).startswith(('/data', '/home', '/Users'))
                       for v in value.reshape(-1)), ('Private path in source', ds, sid, key)

    native_row = native.get((ds, sid))
    path = Path(native_row['native_path']) if native_row else args.beat_native / (sid + '.npz')
    native_sha = sha(path)
    if (ds, sid) in restored_sources:
        assert restored_sources[ds, sid]['source_sha256'] == native_sha
    fingers = None
    betas = np.zeros(20, dtype=np.float64)
    sidecar_sha = None
    if ds == 'BEAT2':
        with np.load(path, allow_pickle=False) as z:
            assert float(z['mocap_frame_rate']) == 30 and len(z['poses']) == n
            native_body = Rotation.from_rotvec(z['poses'][:, :66].reshape(-1, 3)).as_matrix().reshape(n, 22, 3, 3)
            native_trans = z['trans'].copy()
            fingers = z['poses'][:, 75:165].reshape(n, 30, 3).copy()
            betas = z['betas'].copy()
        sidecar = args.unified / row['hand_path']
        sidecar_sha = sha(sidecar)
        assert sidecar_sha == row['hand_sha256']
        with np.load(sidecar, allow_pickle=False) as z:
            assert str(z['source_sha256']) == native_sha and float(z['fps']) == 30
            np.testing.assert_array_equal(z['finger_rotvec'], fingers.reshape(n, 90))
            np.testing.assert_array_equal(z['betas'], betas)
        hand_policy, shape_policy = 'native SMPL-X fingers, exact passthrough', 'native SMPL-X betas300'
    elif ds == 'FineDance':
        source315 = np.load(path, allow_pickle=False)
        assert source315.ndim == 2 and source315.shape[1] == 315
        assert clock[-1] < len(source315)
        aligned = source315[clock]
        matrix = row6_matrices(aligned[:, 3:].astype(np.float64).reshape(n, 52, 6))
        native_body = matrix[:, :22]
        fingers = Rotation.from_matrix(matrix[:, 22:].reshape(-1, 3, 3)).as_rotvec().reshape(n, 30, 3)
        native_trans = aligned[:, :3]
        hand_policy, shape_policy = 'native52 row6 fingers on actual source_frame30', 'neutral zero20; shape not provided'
    else:
        # Only the trusted, hash-pinned author AIST pickle is read with pickle.
        with path.open('rb') as f:
            raw = pickle.load(f)
        body = np.asarray(raw['smpl_poses']).reshape(-1, 24, 3)[::2, :22]
        assert len(body) == n
        native_body = Rotation.from_rotvec(body.reshape(-1, 3)).as_matrix().reshape(n, 22, 3, 3)
        native_trans = np.asarray(raw['smpl_trans'] / raw['smpl_scaling'])[::2]
        hand_policy, shape_policy = 'unavailable; neutral fingers, not supervision', 'neutral zero20'
    assert np.isfinite(betas).all() and betas.ndim == 1
    poses, trans = transport.soma_to_smplx(local, root, producer=producer,
                                         root_scale=scale, hands=fingers)
    restored = Rotation.from_rotvec(poses[:, :66].reshape(-1, 3)).as_matrix().reshape(n, 22, 3, 3)
    mapped = global_rot(local, transport.SOMA_PARENTS)[:, transport.BODY_FROM_SOMA]
    global_error = float(np.max(np.abs(global_rot(restored, transport.SMPLX_PARENTS) - mapped)))
    assert global_error < 2e-6, (ds, sid, global_error)
    angles = Rotation.from_matrix((native_body.swapaxes(-1, -2) @ restored).reshape(-1, 3, 3)).magnitude()
    angle_max = float(np.degrees(angles.max()))
    # Independent native diagnostics gate the declared 0.1-degree envelope.
    assert angle_max < .1, (ds, sid, 'native rotation mismatch', angle_max)
    offset = trans - native_trans
    root_variation = float(np.max(np.abs(offset - offset[0])))
    assert root_variation < 5e-5, (ds, sid, 'native root trajectory mismatch', root_variation)
    metadata_correction = None
    if original_estimated_count:
        # A historical provenance flag is not sufficient to declare the actual
        # frame estimated. Only native-equivalent values may be relabelled; no
        # native body/root values are copied into the motion.
        per_frame_angles = np.degrees(angles.reshape(n, 22).max(axis=1))
        assert per_frame_angles[estimated].max() < 1e-5
        assert np.max(np.abs((offset-offset[0])[estimated])) < 1e-6
        arrays['original_estimated_frame'] = estimated.copy()
        estimated[:] = False
        arrays['estimated_frame'] = estimated.copy()
        metadata_correction = dict(field='estimated_frame',original_flagged_frames=original_estimated_count,
                                   actual_body_rotation_error_max_degrees=float(per_frame_angles[arrays['original_estimated_frame']].max()),
                                   reason='Native-equivalent frame; historical flag retained in original_estimated_frame; no motion values changed')
    if fingers is not None:
        np.testing.assert_array_equal(poses[:, 75:], fingers.reshape(n, 90))
    soma_path, smplx_path = dest / 'SOMA30' / (sid + '.npz'), dest / 'SMPLX30' / (sid + '.npz')
    cached = cached_sources.get((ds,sid)) if not metadata_correction else None
    if cached:
        assert cached['source_native_sha256'] == native_sha and cached['soma_sha256'] == source_sha
        assert cached['producer'] == producer and cached['root_scale'] == scale
        for kind in ('soma','smplx','hands'):
            if cached.get(kind+'_path'):
                assert sha(args.reuse_from/cached[kind+'_path']) == cached[kind+'_sha256']
        old_smplx=args.reuse_from/cached['smplx_path']
        with np.load(old_smplx,allow_pickle=False) as z:
            for key,expected in [('poses',poses),('trans',trans),('betas',betas),('source_frame30',clock),('estimated_frame',estimated)]:
                np.testing.assert_array_equal(z[key],expected)
        os.link(args.reuse_from/cached['soma_path'],soma_path)
        os.link(old_smplx,smplx_path)
    else:
        if metadata_correction:
            with soma_path.open('xb') as f:np.savez_compressed(f,**arrays)
            with np.load(soma_path,allow_pickle=False) as z:
                with np.load(source,allow_pickle=False) as original:
                    for key in original.files:
                        if key!='estimated_frame':np.testing.assert_array_equal(z[key],original[key])
                    np.testing.assert_array_equal(z['original_estimated_frame'],original['estimated_frame'])
        else:
            shutil.copyfile(source,soma_path)
            assert sha(soma_path)==source_sha
        with smplx_path.open('xb') as f:
            np.savez_compressed(f, poses=poses, trans=trans, betas=betas, expression=np.zeros(100 if ds == 'BEAT2' else 10),
                                fps=np.asarray(30), source_frame30=clock, estimated_frame=estimated)
    with np.load(smplx_path, allow_pickle=False) as z:
        for key, expected in [('poses', poses), ('trans', root.astype(np.float64) / scale), ('source_frame30', clock), ('betas', betas)]:
            np.testing.assert_array_equal(z[key], expected)
    # The same native finger parameters accompany SOMA as SMPL-X, never HandRep.
    hand_rel, hand_sha = None, None
    if fingers is not None:
        hp = dest / 'SMPLX_HANDS' / (sid + '.npz')
        if cached:
            old_hand=args.reuse_from/cached['hands_path']
            with np.load(old_hand,allow_pickle=False) as z:
                np.testing.assert_array_equal(z['finger_rotvec'],fingers.reshape(n,90))
                np.testing.assert_array_equal(z['betas'],betas)
                np.testing.assert_array_equal(z['source_frame30'],clock)
            os.link(old_hand,hp)
        else:
            with hp.open('xb') as f:
                np.savez_compressed(f, finger_rotvec=fingers.reshape(n, 90), betas=betas,
                                    fps=np.asarray(30), source_frame30=clock)
        hand_rel, hand_sha = str(hp.relative_to(args.output)), sha(hp)
    return dict(dataset=ds, source_id=sid, split=row['split'], train_allowed=row['split'] == 'train',
                frames=n, fps=30, source_frame30_start=int(clock[0]), source_frame30_end_exclusive=int(clock[-1])+1,
                time_origin_s=float(clock[0])/30, soma_path=str(soma_path.relative_to(args.output)), soma_sha256=sha(soma_path),
                source_soma_sha256=source_sha,metadata_correction=metadata_correction,
                smplx_path=str(smplx_path.relative_to(args.output)), smplx_sha256=sha(smplx_path),
                hands_path=hand_rel, hands_sha256=hand_sha, hand_policy=hand_policy, shape_policy=shape_policy,
                source_native_sha256=native_sha, original_hand_sidecar_sha256=sidecar_sha,
                producer=producer, producer_evidence=producer_evidence, root_scale=scale,
                face_policy='neutral jaw, eyes and expression; facial annotation not recovered',
                world_root_policy='producer root / scale, fixed offset retained',
                native_root_fixed_offset_m=offset[0].tolist(), native_root_offset_variation_max_m=root_variation,
                native_body_rotation_max_degrees=angle_max, inverse_global_rotation_max_abs=global_error,
                estimated_frames=int(estimated.sum()),original_estimated_frame_flags=original_estimated_count,
                reused_verified_output=bool(cached),smoothing=False, fitting=False,
                hand_retargeting=False, axis_convention='Y_UP', units='m')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--unified', type=Path, required=True)
    p.add_argument('--native-manifest', type=Path, required=True)
    p.add_argument('--beat-native', type=Path, required=True)
    p.add_argument('--beat-splits', type=Path, required=True)
    p.add_argument('--converter', type=Path, required=True)
    p.add_argument('--restore-report', type=Path, required=True)
    p.add_argument('--restore-builder', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--reuse-from',type=Path,help='Recheck native inputs and exact output arrays, then reuse verified immutable files')
    p.add_argument('--diagnostic', action='store_true', help='One sample per dataset; never a full release')
    a = p.parse_args()
    if hasattr(os, 'sched_getaffinity'):
        assert len(os.sched_getaffinity(0)) <= 2, 'Start with taskset, at most two CPUs'
    sys.path.insert(0, str(a.converter))
    from soma_smplx import native_transport as t
    a.output.mkdir(parents=True, exist_ok=False)
    (a.output/'code').mkdir()
    shutil.copyfile(__file__, a.output/'code/build_motion_snapshot.py')
    shutil.copyfile(t.__file__, a.output/'code/native_transport.py')
    run = dict(id='HS-THREE-MOTION-20261106', started_at=now(), pid=os.getpid(), cpu_limit=2, gpu=False,
               unified_manifest_sha256=sha(a.unified/'manifest.jsonl'), native_manifest_sha256=sha(a.native_manifest),
               beat_split_csv_sha256=sha(a.beat_splits), converter_version=t.VERSION,
               converter_sha256=sha(Path(t.__file__)), builder_sha256=sha(Path(__file__)),
               restore_report_sha256=sha(a.restore_report), restore_builder_sha256=sha(a.restore_builder),
               publication_authorization='Project owner explicitly confirmed written author permissions for all three datasets in this chat, 2026-11-06',
               diagnostic=a.diagnostic, scope='motion only; no audio, text, model assets, meshes or learned features')
    write(a.output/'run.json', run)
    cached_sources={}
    if a.reuse_from:
        old_run=json.loads((a.reuse_from/'run.json').read_text())
        assert old_run['unified_manifest_sha256']==run['unified_manifest_sha256']
        assert old_run['converter_sha256']==run['converter_sha256']
        cached_rows=list(map(json.loads,(a.reuse_from/'manifest.jsonl').read_text().splitlines()))
        cached_sources={(r['dataset'],r['source_id']):r for r in cached_rows}
        assert len(cached_sources)==len(cached_rows)
        write(a.output/'reuse.json',dict(prior_run_sha256=sha(a.reuse_from/'run.json'),
              prior_builder_sha256=old_run['builder_sha256'],prior_manifest_sha256=sha(a.reuse_from/'manifest.jsonl'),
              reusable_sources=len(cached_sources),policy='Native body/root/hands rechecked, output arrays compared exactly before immutable hardlink reuse'))
    allrows = [json.loads(s) for s in (a.unified/'manifest.jsonl').read_text().splitlines()]
    rows = [r for r in allrows if r['dataset'] in ('AIST++', 'FineDance', 'BEAT2')]
    native = {(r['dataset'], r['source_id']):r for r in map(json.loads, a.native_manifest.read_text().splitlines())}
    assert sha(a.restore_builder) == '1156eba39acf4cab8987283726a679366061ed963ac2199bac123d229f071aa9'
    restored_rows = json.loads(a.restore_report.read_text())['results']
    assert len(restored_rows) == 36
    restored_sources = {(r['dataset'], r['source_id']):r for r in restored_rows}
    assert len(restored_sources) == 36
    with a.beat_splits.open(newline='') as f:
        beat = {r['id']:r['type'] for r in csv.DictReader(f)}
    keys = {(r['dataset'], r['source_id']) for r in rows}
    assert len(keys) == len(rows)
    for ds in ('AIST++', 'FineDance', 'BEAT2'):
        actual = {r['source_id']:r['split'] for r in rows if r['dataset'] == ds}
        expected = beat if ds == 'BEAT2' else {sid:r['split'] for (d, sid),r in native.items() if d == ds}
        assert actual == expected, (ds, 'Official membership/split mismatch')
    rows.sort(key=lambda r:(r['dataset'],r['source_id']))
    counts = collections.Counter((r['dataset'],r['split']) for r in rows)
    expected_counts = {'AIST++': {'train':952,'val':20,'test':20}, 'FineDance':{'train':183,'test':19},
                       'BEAT2':{'train':1093,'val':106,'test':265,'additional':156}}
    assert {ds:{s:counts[ds,s] for s in ss} for ds,ss in expected_counts.items()} == expected_counts
    write(a.output/'membership.json', dict(counts=expected_counts, source_count=len(rows), official_member_sets_verified=True,
                                         additional_training=False, aist_fid_reference_sources=1363,
                                         aist_training_sources=952, hml_included=False))
    if a.diagnostic:
        rows = [next(r for r in rows if r['dataset']==ds) for ds in ('AIST++','FineDance','BEAT2')]
    for dirname in ('AISTpp','FineDance','BEAT2'):
        for part in ('SOMA30','SMPLX30','SMPLX_HANDS'):
            (a.output/dirname/part).mkdir(parents=True)
    completed = []
    with (a.output/'manifest.jsonl').open('x', buffering=1) as f:
        for row in rows:
            rec = convert(row,a,native,t,restored_sources,cached_sources)
            f.write(json.dumps(rec,allow_nan=False)+'\n')
            completed.append(rec)
            if len(completed) <= 3 or len(completed) % 25 == 0:
                print(json.dumps(dict(completed=len(completed),expected=len(rows),last=rec['dataset']+'/'+rec['source_id'])),flush=True)
    for dirname, ds in [('AISTpp','AIST++'),('FineDance','FineDance'),('BEAT2','BEAT2')]:
        with (a.output/dirname/'manifest.jsonl').open('x') as f:
            for rec in completed:
                if rec['dataset'] == ds: f.write(json.dumps(rec,allow_nan=False)+'\n')
    stats = {ds:dict(sources=sum(r['dataset']==ds for r in completed),
                    frames=sum(r['frames'] for r in completed if r['dataset']==ds),
                    native_body_rotation_max_degrees=max(r['native_body_rotation_max_degrees'] for r in completed if r['dataset']==ds),
                    native_root_offset_variation_max_m=max(r['native_root_offset_variation_max_m'] for r in completed if r['dataset']==ds))
             for ds in ('AIST++','FineDance','BEAT2')}
    assert sha(Path(__file__)) == run['builder_sha256']
    assert sha(Path(t.__file__)) == run['converter_sha256']
    assert sha(a.unified/'manifest.jsonl') == run['unified_manifest_sha256']
    write(a.output/'acceptance.json',dict(run,status='DIAGNOSTIC_ONLY' if a.diagnostic else 'FULL_SOURCE_TRANSPORT_INTEGRITY_VERIFIED',
          completed_at=now(),completed=len(completed),datasets=stats,manifest_sha256=sha(a.output/'manifest.jsonl'),
          metadata_corrected_sources=[dict(dataset=r['dataset'],source_id=r['source_id'],source_soma_sha256=r['source_soma_sha256'],soma_sha256=r['soma_sha256'],correction=r['metadata_correction']) for r in completed if r['metadata_correction']],
          reused_sources=sum(r['reused_verified_output'] for r in completed),
          quality_scope='Rotation/root/clock/hands/inverse integrity; aggregate scoring covers AIST40, FineDance19, BEAT speaker2 TEST15 only; condition synchronization and mesh quality not inferred'))
    print(json.dumps(dict(status='DONE',completed=len(completed),datasets=stats)),flush=True)


if __name__ == '__main__':
    main()
