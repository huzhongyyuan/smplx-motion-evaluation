"""Explicit 263D temporal transport diagnostic, not a new30fps T2M evaluator.

263D increments must change units and 6D rotations must stay on SO(3).
This feature-space control is not proof of skeletal/mesh retargeting fidelity.
"""
import numpy as np
from scipy.spatial.transform import Rotation,Slerp
from contracts import require,sample_times

def transport(value,fps,target_fps):
    x=np.asarray(value,dtype=np.float64)
    require(x.ndim==2 and x.shape[1]==263 and len(x)>=2 and np.isfinite(x).all(),'Require finite263D')
    if fps==target_fps: return x.copy(),np.arange(len(x))/fps
    old=np.arange(len(x))/fps; new=sample_times(len(x),fps,target_fps)
    y=np.empty((len(new),263),dtype=np.float64)
    # Positional channels: root height plus root-relative22-joint locations.
    for j in range(3,67): y[:,j]=np.interp(new,old,x[:,j])
    # Official HML uses the first two matrix columns for continuous6D.
    six=x[:,67:193].reshape(len(x),21,6)
    b1=six[...,:3]; b1=b1/np.linalg.norm(b1,axis=-1,keepdims=True)
    b2=six[...,3:]; b2=b2-(b1*b2).sum(-1,keepdims=True)*b1
    b2=b2/np.linalg.norm(b2,axis=-1,keepdims=True)
    matrices=np.stack([b1,b2,np.cross(b1,b2)],axis=-1)
    require(np.isfinite(matrices).all(),'Degenerate6D rotation')
    out=np.stack([Slerp(old,Rotation.from_matrix(matrices[:,j]))(new).as_matrix() for j in range(21)],axis=1)
    y[:,67:193]=np.concatenate([out[...,0],out[...,1]],axis=-1).reshape(len(new),126)
    # Per-frame increments are rates times the destination interval. This does
    # not interpolate raw axis angles or treat a30fps increment as a20fps one.
    ratio=fps/target_fps
    for j in list(range(3))+list(range(193,259)): y[:,j]=np.interp(new,old,x[:,j])*ratio
    # Contacts are categorical. Choose the closest original timestamp.
    index=np.clip(np.floor(new*fps+.5).astype(int),0,len(x)-1)
    y[:,259:263]=x[index,259:263]
    require(np.isfinite(y).all(),'Nonfinite temporal transport')
    return y.astype(value.dtype),new

def roundtrip20_30_20(value):
    up,_=transport(value,20,30)
    down,_=transport(up,30,20)
    # Exact common support only; no extra terminal sample or held tail.
    return up,down,np.asarray(value)[:len(down)].copy()
