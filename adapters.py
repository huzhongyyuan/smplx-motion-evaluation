"""SMPL-X exports to benchmark-native input. No SOMA conversion or metric code.

The separately owned converter must supply versioned SMPL-X exports and clocks.
Missing real endpoint samples are errors, never filled with held/extrapolated GT.
"""
import argparse
import json
from pathlib import Path
import pickle
import shutil
import sys
import numpy as np
from scipy.spatial.transform import Rotation
import torch
from contracts import require, load_smplx, smplx_shape, merge_body_hands, read_manifest, sample_at
from evaluator_common import ROOT, UP, setup, smpl_male, write_json, load_file
from source_lock import sha

def smplx_neutral():
    from smplx import SMPLX
    return SMPLX(str(ROOT/'assets/emage_evaltools/smplx_models/smplx/SMPLX_NEUTRAL_2020.npz'),
                 num_betas=20, num_expression_coeffs=10, use_pca=False, flat_hand_mean=True).eval()

def smplx_joints(model, poses, trans, betas=None, expression=None):
    betas = np.zeros(20) if betas is None else np.asarray(betas)
    expression = np.zeros(10) if expression is None else np.asarray(expression)
    require(betas.ndim == 1 and len(betas) in (10, 20), 'Body FK requires declared10/20 shape coefficients')
    require(expression.shape == (10,), 'Body FK requires10 expression coefficients')
    betas = np.pad(betas, (0, 20-len(betas)))
    with torch.no_grad():
        result = []
        for k in range(0, len(poses), 64):
            p = torch.as_tensor(poses[k:k+64], dtype=torch.float32); n = len(p)
            result.append(model(global_orient=p[:, :3], body_pose=p[:, 3:66],
                                left_hand_pose=p[:, 75:120], right_hand_pose=p[:, 120:165],
                                jaw_pose=torch.zeros(n, 3), leye_pose=torch.zeros(n, 3), reye_pose=torch.zeros(n, 3),
                                betas=torch.as_tensor(betas, dtype=torch.float32)[None].expand(n, -1),
                                transl=torch.as_tensor(trans[k:k+64], dtype=torch.float32),
                                expression=torch.as_tensor(expression, dtype=torch.float32)[None].expand(n, -1)).joints[:, :22].numpy())
        return np.concatenate(result)

def hml_process():
    sys.path.insert(0, str(UP/'T2M'))
    from scripts import motion_process as process
    from common.skeleton import Skeleton
    from utils.paramUtil import t2m_raw_offsets, t2m_kinematic_chain
    process.l_idx1, process.l_idx2 = 5, 8
    process.fid_r, process.fid_l = [8, 11], [7, 10]
    process.face_joint_indx = [2, 1, 17, 16]
    process.n_raw_offsets = torch.from_numpy(t2m_raw_offsets)
    process.kinematic_chain = t2m_kinematic_chain
    example_path = ROOT/'assets/hml_author_endpoint_controls_joints_v1/000021.npy'
    require(sha(example_path) == '2fb3ec52c0413981a7ecd5a75a12c3b7a69c09ec72b12e70127c151cf4f3d4e0',
            'HML author reference skeleton changed')
    example = np.load(example_path, allow_pickle=False)
    process.tgt_offsets = Skeleton(process.n_raw_offsets, t2m_kinematic_chain, 'cpu').get_offsets_joints(torch.from_numpy(example[0]))
    return process

def adapt(args):
    require(args.dataset != 'AIST++', 'Use evaluate.py aist-smplx for actual SMPL-X-domain scoring')
    setup()
    official = read_manifest(args.manifest)
    rows = [r for r in official if r['dataset'] == args.dataset and r['split'] == args.split]
    require(rows, 'No official members selected')
    exports = read_manifest(args.exports)
    exports = {(r['dataset'], r['source_id']): r for r in exports}
    if args.limit is not None:
        require(args.split == 'train' and args.limit > 0, 'Partial interface diagnostics restricted toTRAIN')
        rows = rows[:args.limit]
    args.output.mkdir(parents=True, exist_ok=False)
    code_paths=[ROOT/name for name in ('adapters.py','contracts.py','evaluator_common.py',
                'official_preprocess.py','aist30.py','source_lock.py','assets_lock.py')]
    (args.output/'code').mkdir()
    for path in code_paths:
        shutil.copy2(path,args.output/'code'/path.name)
    geometry = smplx_neutral() if args.dataset != 'BEAT2' else None
    interpolation = getattr(args, 'rotation_interpolation', 'local')
    parents = geometry.parents.numpy() if geometry is not None and interpolation == 'global_fk' else None
    male = smpl_male() if args.dataset == 'AIST++' else None
    if male is not None:
        with torch.no_grad(): male_rest = male().joints[0, 0].numpy()
    process = hml_process() if args.dataset == 'HumanML3D' else None
    fd_receipts = None
    if args.dataset == 'FineDance':
        from official_preprocess import finedance_preprocess
        _, fd_receipts = finedance_preprocess({r['source_id']: r['native_path'] for r in rows}, args.output/'native_preprocess')
        sys.path[:0] = [str(UP/'LODGE'), str(UP/'LODGE/metric')]
        from dld.data.render_joints.smplfk import SMPLX_Skeleton
        ground = load_file('official_lodge_grounding', UP/'LODGE/data/code/preprocess.py')
        fd_skeleton = SMPLX_Skeleton(Jpath=str(UP/'LODGE/data/smplx_neu_J_1.npy'))
        lodge_rest = np.load(UP/'LODGE/data/smplx_neu_J_1.npy')[0]
    adapted, receipts = [], []
    for row in rows:
        sid = row['source_id']; key = (args.dataset, sid)
        require(key in exports, f'Missing export: {key}')
        e = exports[key]
        require(e['arm'] == args.arm and e['fps'] == 30, 'Export arm/fps mismatch')
        require(e['axis_convention'] == 'Y_UP' and e['units'] == 'm', 'Adapter requires explicit Y_UP meters; no guessed axis transform')
        require(e['converter_version'], 'Missing frozen converter version')
        require(sha(e['smplx_path']) == e['smplx_sha256'], 'Export bytes changed')
        if args.arm in ('tokenizer_roundtrip', 'generated'):
            require(len(e['model_checkpoint_sha256']) == 64, 'Missing checkpoint identity')
        poses, trans, betas = load_smplx(e['smplx_path'], 30, body_only=args.dataset != 'BEAT2')
        shape_encoding = e.get('shape_encoding', 'smplx_betas_only')
        betas, expression = smplx_shape(e['smplx_path'], shape_encoding)
        origin = float(e['time_origin_s']); require(np.isfinite(origin), 'Invalid source clock')
        out = args.output/sid
        if args.dataset == 'BEAT2':
            require(e.get('hand_sidecar_path'), 'BEAT2 needs native SMPL-X hand sidecar of the same arm')
            require(sha(e['hand_sidecar_path']) == e['hand_sidecar_sha256'], 'Hand bytes changed')
            require(origin == 0, 'Official full-source BEAT2 starts at0s')
            out = out.with_suffix('.npz')
            merge_body_hands(e['smplx_path'], e['hand_sidecar_path'], out, args.dataset, args.arm, sid, origin,
                             shape_encoding=shape_encoding)
            with np.load(row['native_path'], allow_pickle=False) as native:
                require(len(poses) == len(native['poses']), 'BEAT2 length mismatch')
            fmt = 'smplx165'
        else:
            # Preserve the converter's declared shape. Forcing zero here would
            # reinterpret its translation and recreate a pelvis/root offset.
            rest = smplx_joints(geometry, np.zeros((2, 165)), np.zeros((2, 3)), betas, expression)[0, 0]
            matrices = Rotation.from_rotvec(poses.reshape(-1, 3)).as_matrix().reshape(len(poses), 55, 3, 3)
            if args.dataset == 'HumanML3D':
                n = len(np.load(row['native_path'], mmap_mode='r', allow_pickle=False))
                # process_file consumes N+1 positions to produce N263D features.
                rr, tt = sample_at(matrices, trans, 30, origin, np.arange(n+1)/20, parents=parents)
                aa = Rotation.from_matrix(rr.reshape(-1, 3, 3)).as_rotvec().reshape(n+1, 165)
                positions = smplx_joints(geometry, aa, tt, betas, expression)
                value = process.process_file(positions.copy(), .002)[0]
                require(value.shape == (n, 263) and np.isfinite(value).all(), 'HML official preprocessing failed')
                fmt = 'humanml263'
            elif args.dataset == 'AIST++':
                from aist30 import CAP, FORMAT, select_export_frames
                with Path(row['native_path']).open('rb') as f: n = min((len(pickle.load(f)['smpl_poses'])+1)//2, CAP)
                rr, tt = select_export_frames(matrices, trans, origin, n)
                aa = Rotation.from_matrix(rr[:, :22].reshape(-1, 3, 3)).as_rotvec().reshape(n, 66)
                # SMPL24 and SMPL-X body22 share local rotation names. The two
                # extra SMPL hand joints use defaults; SMPL-X jaw/eye slots are
                # never mistaken for hands. Preserve world pelvis, not raw
                # model-dependent translation coefficients.
                st = tt+rest-male_rest; value = []
                with torch.no_grad():
                    for k in range(0, n, 64):
                        p = torch.as_tensor(aa[k:k+64], dtype=torch.float32)
                        body = torch.cat([p[:, 3:], torch.zeros(len(p), 6)], dim=-1)
                        value.append(male(global_orient=p[:, :3], body_pose=body,
                                          transl=torch.as_tensor(st[k:k+64], dtype=torch.float32)).joints[:, :24].numpy())
                value = np.concatenate(value); fmt = FORMAT
            else:
                crop = fd_receipts[sid]['native_source_start_frame']; n = min(fd_receipts[sid]['frames'], 1024)
                rr, tt = sample_at(matrices, trans, 30, origin, (crop+np.arange(n))/30, parents=parents)
                body = Rotation.from_matrix(rr[:, :22].reshape(-1, 3, 3)).as_rotvec().reshape(n, 66)
                local = torch.cat([torch.as_tensor(body, dtype=torch.float32), torch.zeros(n, 90)], dim=-1)
                root = torch.as_tensor(tt+rest-lodge_rest, dtype=torch.float32)
                # Original constant grounding; contacts are not scored by LODGE.
                root, local = ground.set_on_ground(root, local, fd_skeleton)
                row6 = ground.ax_to_6v(local.reshape(n, 52, 3))[:, :22].reshape(n, 132).numpy()
                value = np.concatenate([np.zeros((n, 4)), root.numpy(), row6], axis=-1)
                fmt = 'lodge139_preprocessed'
                row = dict(row, prediction_native_start_frame=crop)
            out = out.with_suffix('.npy'); np.save(out, value)
        adapted.append(dict(row, prediction_path=str(out), prediction_format=fmt))
        receipts.append(dict(source_id=sid, export=e, prediction=str(out), prediction_sha256=sha(out)))
    (args.output/'manifest.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in adapted))
    write_json(args.output/'receipt.json', dict(status='ADAPTED_NOT_QUALITY_ACCEPTED', arm=args.arm, dataset=args.dataset,
                                              own_code_sha256={p.name:sha(p) for p in code_paths},
                                              sources_lock_sha256=sha(ROOT/'sources.lock.json'),
                                              assets_lock_sha256=sha(ROOT/'assets.lock.json'),
                                              rotation_interpolation='none_exact30fps' if args.dataset=='AIST++' else interpolation,
                                              original_manifest_sha256=sha(args.manifest), exports_manifest_sha256=sha(args.exports),
                                              preprocessing_scope='official target-format preparation; SMPL-X-to-SMPL parameter transfer must be quantified by GT controls', sources=receipts))

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', choices=('HumanML3D','AIST++','FineDance','BEAT2'), required=True)
    p.add_argument('--manifest', type=Path, required=True); p.add_argument('--exports', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True); p.add_argument('--arm', required=True)
    p.add_argument('--split', choices=('train','val','test'), required=True); p.add_argument('--limit', type=int)
    p.add_argument('--rotation-interpolation', choices=('local','global_fk'), default='local')
    args = p.parse_args(); args.output = args.output.resolve()
    try:
        adapt(args)
    except BaseException as error:
        if args.output.is_dir():
            write_json(args.output/'failure.json',dict(status='ADAPTER_PIPELINE_FAILURE',error=repr(error),automatic_retry=False))
        raise

if __name__ == '__main__': main()
