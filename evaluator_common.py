import contextlib
import importlib.util
import json
import os
from pathlib import Path
import pickle
import sys
from unittest.mock import patch
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
UP = ROOT/'upstream'

def setup():
    from source_lock import verify
    verify()
    if (ROOT/'assets.lock.json').exists():
        from assets_lock import verify as verify_assets
        verify_assets()
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise ValueError('CPU evaluation entry requires CUDA_VISIBLE_DEVICES empty')
    torch.set_num_threads(2)
    sys.path.insert(0, str(UP/'pytorch3d'))
    # Python3.10 implements pickle protocol5 natively; LODGE imports the old backport.
    sys.modules['pickle5'] = pickle

def load_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module

@contextlib.contextmanager
def cwd(path):
    old = Path.cwd(); os.chdir(path)
    try: yield
    finally: os.chdir(old)

@contextlib.contextmanager
def cpu_checkpoint_loading():
    # Official published checkpoints can contain CUDA storages or numpy metadata.
    # Restrict this IO compatibility change to the trusted frozen model constructor.
    original = torch.load
    def load(*args, **kwargs):
        kwargs['map_location'] = 'cpu'; kwargs['weights_only'] = False
        return original(*args, **kwargs)
    with patch.object(torch, 'load', load): yield

def json_value(x):
    if isinstance(x, dict): return {k: json_value(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)): return [json_value(v) for v in x]
    if isinstance(x, np.ndarray): return json_value(x.tolist())
    if isinstance(x, np.generic): x = x.item()
    if isinstance(x, complex):
        if not (np.isfinite(x.real) and np.isfinite(x.imag)):
            raise ValueError('Nonfinite official complex metric')
        # Some unmodified author sqrtm implementations return complex scalars
        # even for roundoff-size imaginary residuals. Preserve both components;
        # a JSON transport layer must neither set a scientific tolerance nor
        # silently discard the imaginary component.
        return dict(type='complex',real=x.real,imag=x.imag)
    if isinstance(x, float) and not np.isfinite(x): raise ValueError('Nonfinite official metric')
    return x

def write_json(path, value):
    Path(path).write_text(json.dumps(json_value(value), indent=2, allow_nan=False)+'\n')

def smpl_male():
    from smplx import SMPL
    return SMPL(str(ROOT/'assets/smpl'), gender='male', batch_size=1).eval()

def native_aist_joints(path, model, frame_limit=None, frame_step=1):
    with Path(path).open('rb') as f: item = pickle.load(f)
    pose = np.asarray(item['smpl_poses'], dtype=np.float32).reshape(-1, 72)
    trans = np.asarray(item['smpl_trans']/item['smpl_scaling'], dtype=np.float32)
    if frame_limit is not None: pose, trans = pose[:frame_limit], trans[:frame_limit]
    if frame_step not in (1, 2): raise ValueError('AIST native sampling must be60fps or direct30fps')
    pose, trans = pose[::frame_step], trans[::frame_step]
    with torch.no_grad():
        return np.concatenate([model(global_orient=torch.from_numpy(pose[k:k+64, :3]),
                                     body_pose=torch.from_numpy(pose[k:k+64, 3:]),
                                     transl=torch.from_numpy(trans[k:k+64])).joints[:, :24].numpy()
                               for k in range(0, len(pose), 64)])
