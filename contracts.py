"""Time/representation contracts at the evaluator boundary. No score formulas."""
from pathlib import Path
import json
import numpy as np
from scipy.spatial.transform import Rotation, Slerp

DATASETS = ('HumanML3D', 'AIST++', 'FineDance', 'BEAT2')

def require(value, message):
    if not value:
        raise ValueError(message)

def finite(x, shape_tail, name):
    x = np.asarray(x)
    require(x.ndim == len(shape_tail)+1 and x.shape[1:] == shape_tail, f'{name}: expected [T,{shape_tail}], got {x.shape}')
    require(len(x) >= 2 and np.isfinite(x).all(), f'{name}: short/nonfinite motion')
    return x

def rotations(x, count):
    x = finite(x, (count, 3, 3), 'rotations')
    require(np.max(np.abs(x.swapaxes(-1, -2) @ x - np.eye(3))) < 1e-4, 'Nonorthogonal rotations')
    require(np.max(np.abs(np.linalg.det(x)-1)) < 1e-4, 'Reflected/invalid rotations')
    return x

def sample_times(length, fps, target_fps):
    require(length >= 2 and fps > 0 and target_fps > 0, 'Invalid time domain')
    # Preserve actual sample timestamps: never stretch endpoints with linspace.
    end = (length-1)/fps
    return np.arange(int(np.floor(end*target_fps+1e-8))+1, dtype=np.float64)/target_fps

def resample_motion(rot, trans, fps, target_fps):
    rot = rotations(rot, rot.shape[1]); trans = finite(trans, (3,), 'translation')
    require(len(rot) == len(trans), 'Rotation/translation lengths differ')
    new_t = sample_times(len(rot), fps, target_fps)
    old_t = np.arange(len(rot), dtype=np.float64)/fps
    if fps == target_fps:
        return rot.copy(), trans.copy(), new_t
    rr = np.stack([Slerp(old_t, Rotation.from_matrix(rot[:, j]))(new_t).as_matrix() for j in range(rot.shape[1])], axis=1)
    tt = np.stack([np.interp(new_t, old_t, trans[:, j]) for j in range(3)], axis=-1)
    return rr, tt, new_t

def sample_at(rot, trans, fps, time_origin_s, query_times, parents=None):
    """Evaluate fixed absolute timestamps without extending the source domain."""
    rot = rotations(rot, rot.shape[1]); trans = finite(trans, (3,), 'translation')
    require(len(rot) == len(trans), 'Rotation/translation lengths differ')
    old = time_origin_s+np.arange(len(rot), dtype=np.float64)/fps
    times = np.asarray(query_times, dtype=np.float64)
    require(times.ndim == 1 and len(times) >= 2 and np.isfinite(times).all() and (np.diff(times) > 0).all(), 'Invalid target timestamps')
    require(times[0] >= old[0]-1e-8 and times[-1] <= old[-1]+1e-8,
            f'Missing real endpoint: available[{old[0]},{old[-1]}]s; requested[{times[0]},{times[-1]}]s. No tail padding/extrapolation allowed')
    times = np.clip(times, old[0], old[-1])  # floating-point endpoint tolerance only
    grid = (times-time_origin_s)*fps
    if np.max(np.abs(grid-np.rint(grid))) < 1e-8:
        index = np.rint(grid).astype(int)
        return rot[index].copy(), trans[index].copy()
    if parents is not None:
        parents = np.asarray(parents)
        require(parents.shape == (rot.shape[1],) and parents[0] == -1 and
                all(0 <= parents[j] < j for j in range(1, len(parents))), 'Invalid FK interpolation hierarchy')
        glob = rot.copy()
        for j in range(1, len(parents)): glob[:, j] = glob[:, parents[j]]@rot[:, j]
        world = np.stack([Slerp(old, Rotation.from_matrix(glob[:, j]))(times).as_matrix() for j in range(rot.shape[1])], axis=1)
        rr = world.copy()
        for j in range(1, len(parents)): rr[:, j] = world[:, parents[j]].swapaxes(-1,-2)@world[:, j]
    else:
        rr = np.stack([Slerp(old, Rotation.from_matrix(rot[:, j]))(times).as_matrix() for j in range(rot.shape[1])], axis=1)
    tt = np.stack([np.interp(times, old, trans[:, j]) for j in range(3)], axis=-1)
    return rr, tt

def load_smplx(path, fps, body_only=False):
    with np.load(path, allow_pickle=False) as z:
        require('poses' in z and 'trans' in z, f'{path}: require poses/trans; no guessed keys')
        poses = finite(z['poses'], (165,), 'SMPL-X poses')
        trans = finite(z['trans'], (3,), 'SMPL-X trans')
        require(len(poses) == len(trans), 'SMPL-X length mismatch')
        if 'mocap_frame_rate' in z:
            require(float(z['mocap_frame_rate']) == fps, 'Metadata FPS mismatch')
        if 'fps' in z:
            require(float(z['fps']) == fps, 'Metadata FPS mismatch')
        betas = z['betas'].copy() if 'betas' in z else None
    if body_only:
        # SMPL-X 0..21 includes wrists. Only finger/jaw/eye local angles reset.
        poses[:, 66:] = 0
    return poses, trans, betas

def smplx_shape(path, encoding='smplx_betas_only'):
    """Read declared shape semantics; never mistake expressions for betas."""
    with np.load(path, allow_pickle=False) as z:
        beta = np.asarray(z['betas'] if 'betas' in z else np.zeros(10))
        require(beta.ndim == 1 and np.isfinite(beta).all(), 'Require one finite frozen body shape')
        if encoding == 'betas20_expression10_concatenated':
            require(beta.shape == (30,) and 'expression' not in z, 'Declared packed20+10 shape mismatch')
            return beta[:20].copy(), beta[20:].copy()
        require(encoding == 'smplx_betas_only', 'Unknown shape encoding')
        require(len(beta) in (10, 20, 300), 'Ambiguous betas count; declare packed20+10 encoding explicitly')
        expression = np.asarray(z['expression'] if 'expression' in z else np.zeros(10))
        require(expression.ndim == 1 and len(expression) in (10, 100) and np.isfinite(expression).all(), 'Invalid frozen expression coefficients')
        return beta, expression

def merge_body_hands(converted_path, hand_path, output, dataset, arm, source_id, time_origin_s=0,
                     shape_encoding='smplx_betas_only'):
    require(dataset in DATASETS, 'Unknown dataset')
    poses, trans, betas = load_smplx(converted_path, 30, body_only=True)
    betas, expression = smplx_shape(converted_path, shape_encoding)
    if dataset == 'BEAT2':
        require(hand_path is not None, 'BEAT2 requires model/GT native SMPL-X hands; cannot infer full hands from SOMA30')
        with np.load(hand_path, allow_pickle=False) as z:
            require(str(z['arm'].item()) == arm, 'Hand sidecar arm mismatch; do not attach GT hands to reconstruction/generation')
            require(str(z['source_id'].item()) == source_id, 'Hand sidecar source mismatch')
            require(int(z['fps']) == 30 and float(z['time_origin_s']) == time_origin_s, 'Hand/body clock mismatch')
            left = finite(z['left_hand_pose'], (45,), 'left hand')
            right = finite(z['right_hand_pose'], (45,), 'right hand')
        require(len(left) == len(right) == len(poses), 'Hand/body timeline mismatch')
        poses[:, 75:120] = left; poses[:, 120:165] = right
    else:
        require(hand_path is None, 'Body-only dataset must not have hand predictions')
    output = Path(output); require(not output.exists(), 'Refuse overwriting export')
    np.savez_compressed(output, poses=poses, trans=trans, betas=betas, expression=expression,
                        fps=30, mocap_frame_rate=30, arm=arm, dataset=dataset, source_id=source_id, time_origin_s=time_origin_s)

def read_manifest(path):
    rows = [json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()]
    require(rows, 'Empty manifest')
    keys = [(x['dataset'], x['source_id']) for x in rows]
    require(len(set(keys)) == len(keys), 'Duplicate source IDs')
    require(all(x['dataset'] in DATASETS for x in rows), 'Unknown dataset')
    return rows
