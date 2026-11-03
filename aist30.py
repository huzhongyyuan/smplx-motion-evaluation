"""Explicit Bailando-derived 30fps protocol; pinned upstream files stay intact.

Only sampling, time parameters and the 20-second frame cap change. The author's
feature equations (including their acceleration expression), joint order,
thresholds, frame-count smoothing window and FID/diversity remain unchanged.
This protocol is not numerically interchangeable with published 60fps scores.
"""
import functools
import importlib
from pathlib import Path
import sys
import types
import numpy as np
from contracts import require
from source_lock import sha

ROOT=Path(__file__).resolve().parent
FPS=30
CAP=600
REFERENCE='references/aist_bailando30_allmapped_v1'
FORMAT='aist_joints24_30fps'


def protocol():
    return dict(id='bailando_30fps_v1',fps=FPS,native_sampling='60fps[::2], no interpolation',
                kinetic_frame_time_s=1/FPS,manual_time_per_frame_s=1/FPS,
                kinetic_sliding_window_frames=2,prediction_cap_frames=CAP,
                reference_cap_frames=None,source_lock_sha256=sha(ROOT/'sources.lock.json'),
                time_adapter_sha256=sha(__file__),published_60fps_score_comparable=False)


def validate_reference(receipt):
    require(receipt.get('feature_protocol')==protocol(),
            'AIST requires the frozen Bailando30 reference; old60fps/unversioned features cannot be reused')


def select_export_frames(rotations, translation, origin, count):
    """Select exact30fps source samples; never invent an intermediate pose."""
    start=-float(origin)*FPS
    require(np.isfinite(start) and abs(start-round(start))<1e-7,
            'AIST export clock must lie on the30fps grid')
    start=int(round(start))
    require(count>1 and start>=0 and start+count<=len(rotations) and len(rotations)==len(translation),
            'AIST export does not cover required30fps support')
    return rotations[start:start+count],translation[start:start+count]


@functools.lru_cache(maxsize=1)
def extractors():
    # Private package avoids sharing LODGE's identically named features/utils.
    name='_soma_eval_bailando_features'
    package=types.ModuleType(name)
    package.__path__=[str(ROOT/'upstream/Bailando/utils/features')]
    sys.modules[name]=package
    kinetic=importlib.import_module(name+'.kinetic')
    manual=importlib.import_module(name+'.manual_new')
    utilities=importlib.import_module(name+'.utils')
    def bind(function, **changes):
        context=dict(function.__globals__); context.update(changes)
        return types.FunctionType(function.__code__,context,function.__name__,
                                  function.__defaults__,function.__closure__)
    # Bind explicit time without mutating any upstream module/class/defaults.
    timed_utils=types.SimpleNamespace(**vars(utilities))
    for key in ('velocity_direction_above_threshold',
                'velocity_direction_above_threshold_normal','velocity_above_threshold'):
        setattr(timed_utils,key,functools.partial(getattr(utilities,key),time_per_frame=1/FPS))
    timed_manual=type('ManualFeatures30',(manual.ManualFeatures,),{
        key:bind(getattr(manual.ManualFeatures,key),feat_utils=timed_utils)
        for key in ('f_move','f_nmove','f_fast')})
    return (bind(kinetic.extract_kinetic_features,
                 KineticFeatures=functools.partial(kinetic.KineticFeatures,frame_time=1/FPS)),
            bind(manual.extract_manual_features,ManualFeatures=timed_manual))


def features(joints):
    x=np.asarray(joints)
    require(x.ndim==3 and x.shape[1:]==(24,3) and len(x)>2 and np.isfinite(x).all(),
            'AIST30 requires finite T>2,24,3 input')
    x=x-x[:1,:1]
    kinetic,manual=extractors()
    result=kinetic(x),manual(x)
    require(all(np.isfinite(v).all() for v in result),'Nonfinite AIST30 features')
    return result


def calc_and_save_feats(root):
    root=Path(root)
    folders=[root/'kinetic_features',root/'manual_features_new']
    for folder in folders: folder.mkdir(exist_ok=False)
    for path in sorted(root.glob('*.npy')):
        value=np.load(path,allow_pickle=True).item()['pred_position']
        value=np.asarray(value).reshape(-1,24,3)[:CAP]
        for folder,array in zip(folders,features(value)):
            np.save(folder/path.name,array)
