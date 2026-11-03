"""Invoke pinned preprocessing; keep the clock separate from conditioning audio."""
from pathlib import Path
from functools import partial
import numpy as np
import sys
from evaluator_common import UP, cwd, load_file, write_json
from source_lock import sha
from contracts import require

def finedance_preprocess(paths, destination):
    """Run LODGE's unmodified motion_feats_extract, including its source-ID crops.

    That function also crops music features but never uses them to compute motion.
    A one-column *frame-index clock*, not audio/features, is supplied solely to
    retrieve and validate the exact official temporal crop. It is never scored,
    exported as a condition, or fed to the model. The original motion FK, constant
    grounding, contact extraction, and ROW6D conversions execute unchanged.
    """
    root = Path(destination).resolve(); root.mkdir(parents=True, exist_ok=False)
    raw, clock, motion, cropped_clock = [root/x for x in ('raw', 'source_frame_clock', 'motion319', 'cropped_frame_clock')]
    raw.mkdir(); clock.mkdir()
    for sid, path in paths.items():
        x = np.load(path, allow_pickle=False)
        require(x.ndim == 2 and x.shape[1] in (135, 139, 315, 319) and np.isfinite(x).all(), 'Invalid FineDance motion')
        if x.shape[1] in (135, 139):
            # The metric uses body22 and default fingers. Complete the upstream
            # preprocessor's required52-joint layout with identity finger angles.
            fingers = np.tile(np.array([1, 0, 0, 0, 1, 0], dtype=x.dtype), (len(x), 30))
            x = np.concatenate([x, fingers], axis=1)
        np.save(raw/f'{sid}.npy', x)
        np.save(clock/f'{sid}.npy', np.arange(len(x), dtype=np.int64)[:, None])
    sys.path[:0] = [str(UP/'LODGE'), str(UP/'LODGE/metric')]
    with cwd(UP/'LODGE'):
        official = load_file('official_lodge_preprocess', UP/'LODGE/data/code/preprocess.py')
        # Resolve the author's machine-specific asset location; the class, asset
        # bytes and all computations remain the published implementations.
        official.SMPLX_Skeleton = partial(official.SMPLX_Skeleton, Jpath=str(UP/'LODGE/data/smplx_neu_J_1.npy'))
        official.motion_feats_extract(str(raw), str(motion), str(clock), str(cropped_clock))
    receipt = {}
    for sid, path in paths.items():
        index = np.load(cropped_clock/f'{sid}.npy', allow_pickle=False)[:, 0]
        x = np.load(motion/f'{sid}.npy', allow_pickle=False)
        require(len(index) == len(x) and len(index) > 1 and np.array_equal(np.diff(index), np.ones(len(index)-1)), 'Invalid official crop clock')
        require(x.shape[1] == 319 and np.isfinite(x).all(), 'Invalid official preprocessor result')
        receipt[sid] = dict(input=str(path), input_sha256=sha(path), output_sha256=sha(motion/f'{sid}.npy'),
                            native_source_start_frame=int(index[0]), frames=len(x), fps=30)
    write_json(root/'receipt.json', dict(upstream=str(UP/'LODGE/data/code/preprocess.py'),
                                       upstream_sha256=sha(UP/'LODGE/data/code/preprocess.py'),
                                       clock_is_conditioning_audio=False, sources=receipt))
    return motion, receipt
