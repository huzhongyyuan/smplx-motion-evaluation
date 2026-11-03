"""AIST++30 actual SMPL-X/body24, strict TRAIN952 reference. Not SMPL60 paper scores."""
import argparse
from collections import Counter
from datetime import datetime,timezone
import json
from pathlib import Path
import pickle
import sys
import numpy as np
import torch
from adapters import smplx_neutral
from aist30 import features,protocol
from contracts import read_manifest,require,load_smplx
from evaluator_common import ROOT,UP,setup,smpl_male,load_file,write_json,native_aist_joints
from source_lock import sha

KINDS=('kinetic_features','manual_features_new')
MAP24=list(range(22))+[28,43]

def full_smplx_positions(model,poses,trans):
    values=[]
    with torch.no_grad():
        for k in range(0,len(poses),64):
            p=torch.as_tensor(poses[k:k+64],dtype=torch.float32); n=len(p)
            values.append(model(global_orient=p[:,:3],body_pose=p[:,3:66],left_hand_pose=p[:,75:120],
                right_hand_pose=p[:,120:165],jaw_pose=p[:,66:69],leye_pose=p[:,69:72],reye_pose=p[:,72:75],
                betas=torch.zeros(n,20),expression=torch.zeros(n,10),
                transl=torch.as_tensor(trans[k:k+64],dtype=torch.float32)).joints[:,MAP24].numpy())
    result=np.concatenate(values)
    require(result.shape==(len(poses),24,3) and np.isfinite(result).all(),'Invalid SMPL-X geometry')
    return result

def smplx_positions(model,poses,trans):
    from smplx.lbs import vertices2joints,batch_rodrigues,batch_rigid_transform
    require(torch.count_nonzero(model.pose_mean).item()==0,'Require flat-hand zero-pose model')
    with torch.no_grad():
        rest=vertices2joints(model.J_regressor,model.v_template[None])[0]
        rot=batch_rodrigues(torch.as_tensor(poses,dtype=torch.float32).reshape(-1,3)).reshape(len(poses),55,3,3)
        joints,_=batch_rigid_transform(rot,rest[None].expand(len(poses),-1,-1),model.parents)
        value=joints[:,MAP24].numpy()+trans[:,None,:]
    index=np.array([0,len(poses)//2,len(poses)-1])
    exact=full_smplx_positions(model,poses[index],trans[index])
    np.testing.assert_allclose(value[index],exact,atol=1e-6,rtol=0)
    return value,float(np.abs(value[index]-exact).max())

def save_features(joints,target,sid):
    for kind,value in zip(KINDS,features(joints)):
        (target/kind).mkdir(parents=True,exist_ok=True); np.save(target/kind/(sid+'.npy'),value)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',type=Path,required=True); p.add_argument('--output',type=Path,required=True)
    p.add_argument('--exports',type=Path,help='Optional candidate body SMPL-X30 exports; same zero shape and clocks as GT')
    a=p.parse_args(); setup()
    rows=[r for r in read_manifest(a.manifest) if r['dataset']=='AIST++']
    require(Counter(r['split'] for r in rows)==dict(train=952,val=20,test=20),'Require full official992 members')
    public=[json.loads(s) for s in (ROOT/'metadata/members.jsonl').read_text().splitlines()]
    require({(r['source_id'],r['split']) for r in rows}=={(r['source_id'],r['split']) for r in public if r['dataset']=='AIST++'},'Official IDs changed')
    out=a.output.resolve(); out.mkdir(parents=True,exist_ok=False)
    male,model=smpl_male(),smplx_neutral()
    with torch.no_grad():
        male_root=male().joints[0,0].numpy()
        delta=male_root-model().joints[0,0].numpy()
    exports={r['source_id']:r for r in read_manifest(a.exports)} if a.exports else {}
    records=[]
    for row in sorted(rows,key=lambda r:(r['split']!='train',r['split'],r['source_id'])):
        sid,split=row['source_id'],row['split']; digest=sha(row['native_path'])
        require(digest==row['native_sha256'],'Native source changed: '+sid)
        with Path(row['native_path']).open('rb') as f: raw=pickle.load(f)
        pp=np.asarray(raw['smpl_poses'],dtype=np.float32).reshape(-1,72)[::2]
        tt=np.asarray(raw['smpl_trans']/raw['smpl_scaling'],dtype=np.float32)[::2]
        if split!='train': pp,tt=pp[:600],tt[:600]
        poses=np.zeros((len(pp),165),dtype=np.float32); poses[:,:66]=pp[:,:66]; trans=tt+delta
        sx,error=smplx_positions(model,poses,trans)
        np.testing.assert_allclose(sx[:,0],tt+male_root,atol=1e-6,rtol=0)
        smpl=native_aist_joints(row['native_path'],male,frame_limit=None if split=='train' else 1200,frame_step=2)
        roles=['train952'] if split=='train' else ['eval40']+(['test20'] if split=='test' else [])
        for domain,joints in (('smpl',smpl),('smplx',sx)):
            for role in roles: save_features(joints,out/domain/role,sid)
        if exports and split!='train':
            e=exports[sid]; require(e['fps']==30 and e['time_origin_s']==0,'Candidate clock mismatch')
            require(sha(e['smplx_path'])==e['smplx_sha256'],'Candidate bytes changed')
            cp,ct,betas=load_smplx(e['smplx_path'],30,body_only=True)
            require(betas is not None and np.all(np.asarray(betas)==0),'This AIST protocol requires zero shape')
            require(len(cp)>=len(poses),'Incomplete prediction support')
            cj,ce=smplx_positions(model,cp[:len(poses)],ct[:len(poses)])
            for role in roles: save_features(cj,out/'candidate'/role,sid)
        records.append(dict(source_id=sid,official_split=split,frames=len(poses),native_sha256=digest,full_model_check_max_m=error))
        print(json.dumps(dict(completed=len(records),total=992,source_id=sid)),flush=True)
    sys.path.insert(0,str(UP/'Bailando/utils'))
    metric=load_file('aist_smplx_author_metric',UP/'Bailando/utils/metrics_new.py'); scores={}
    for domain in ('smpl','smplx')+(('candidate',) if exports else ()):
        for role in ('test20','eval40'):
            scores[domain+'_'+role+'_vs_train952']=metric.quantized_metrics(str(out/domain/role),str(out/('smplx' if domain=='candidate' else domain)/'train952'))
    write_json(out/'result.json',dict(status='SCORER_RUN_FINISHED',verified_at=datetime.now(timezone.utc).isoformat(),
        protocol=dict(time=protocol(),joint_map=MAP24,reference='TRAIN952 only',shape='zero20',hands='default',
                      published_smpl60_score_comparable=False),metrics=scores,sources=records,formal_official_result=False))

if __name__=='__main__': main()
