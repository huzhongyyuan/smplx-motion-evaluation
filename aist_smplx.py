"""AIST++ actual SMPL-X30/body24, frozen full1363 reference. Native60 is separate."""
import argparse
from collections import Counter
from datetime import datetime,timezone
import json
from pathlib import Path
import pickle
import sys
import numpy as np
import torch
# Host-local, already validated chumpy compatibility for legacy SMPL pickles.
# On installations with chumpy installed this directory need not exist.
sys.path.insert(0,str(Path(__file__).resolve().parent/'compat'))
from adapters import smplx_neutral
from aist30 import features,protocol
from contracts import read_manifest,require,load_smplx
from evaluator_common import ROOT,UP,setup,smpl_male,load_file,write_json
from source_lock import sha
from metric_issues import known_issues
from aist_reference import DEFAULT_REFERENCE, PROTOCOL_ID, verify_reference

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
    p.add_argument('--reference',type=Path,default=DEFAULT_REFERENCE,
                   help='Frozen actual SMPL-X30 full1363 cache; other pools are rejected')
    p.add_argument('--evaluation-split',choices=('all','test'),default='all',
                   help='all=TEST20+VAL20, reported separately; test=official TEST20 only')
    a=p.parse_args(); setup()
    rows=[r for r in read_manifest(a.manifest) if r['dataset']=='AIST++']
    require(Counter(r['split'] for r in rows)==dict(train=952,val=20,test=20),'Require full official992 members')
    public=[json.loads(s) for s in (ROOT/'metadata/members.jsonl').read_text().splitlines()]
    require({(r['source_id'],r['split']) for r in rows}=={(r['source_id'],r['split']) for r in public if r['dataset']=='AIST++'},'Official IDs changed')
    reference,receipt=verify_reference(a.reference)
    pins={r['source_id']:r['native_sha256'] for r in receipt['inputs']}
    rows=[r for r in rows if r['split']!='train' and (a.evaluation_split=='all' or r['split']=='test')]
    score_roles=('test20','eval40') if a.evaluation_split=='all' else ('test20',)
    exports={r['source_id']:r for r in read_manifest(a.exports)} if a.exports else {}
    if exports:
        require(set(exports)=={r['source_id'] for r in rows}, 'Require exact selected official evaluation exports')
        require(all(e['dataset']=='AIST++' and e['fps']==30 and e['time_origin_s']==0
                    and e['axis_convention']=='Y_UP' and e['units']=='m' for e in exports.values()),
                'Candidate dataset/clock/coordinate mismatch')
    out=a.output.resolve(); out.mkdir(parents=True,exist_ok=False)
    write_json(out/'run.json',dict(status='RUNNING',protocol_id=PROTOCOL_ID,
        evaluation_split=a.evaluation_split,
        started_at=datetime.now(timezone.utc).isoformat(),code_sha256=sha(__file__),
        reference=str(reference),reference_receipt_sha256=sha(reference/'reference_receipt.json'),
        manifest_sha256=sha(a.manifest),exports_sha256=sha(a.exports) if a.exports else None))
    male,model=smpl_male(),smplx_neutral()
    with torch.no_grad():
        male_root=male().joints[0,0].numpy()
        delta=male_root-model().joints[0,0].numpy()
    records=[]
    for row in sorted(rows,key=lambda r:(r['split'],r['source_id'])):
        sid,split=row['source_id'],row['split']; digest=sha(row['native_path'])
        require(digest==pins[sid] and digest==row.get('native_sha256',digest),'Native source changed: '+sid)
        with Path(row['native_path']).open('rb') as f: raw=pickle.load(f)
        pp=np.asarray(raw['smpl_poses'],dtype=np.float32).reshape(-1,72)[::2]
        tt=np.asarray(raw['smpl_trans']/raw['smpl_scaling'],dtype=np.float32)[::2]
        if split!='train': pp,tt=pp[:600],tt[:600]
        poses=np.zeros((len(pp),165),dtype=np.float32); poses[:,:66]=pp[:,:66]; trans=tt+delta
        sx,error=smplx_positions(model,poses,trans)
        np.testing.assert_allclose(sx[:,0],tt+male_root,atol=1e-6,rtol=0)
        roles=(['eval40'] if a.evaluation_split=='all' else [])+(['test20'] if split=='test' else [])
        for role in roles: save_features(sx,out/'smplx'/role,sid)
        if exports and split!='train':
            e=exports[sid]; require(e['fps']==30 and e['time_origin_s']==0,'Candidate clock mismatch')
            require(sha(e['smplx_path'])==e['smplx_sha256'],'Candidate bytes changed')
            cp,ct,betas=load_smplx(e['smplx_path'],30,body_only=True)
            require(betas is not None and np.all(np.asarray(betas)==0),'This AIST protocol requires zero shape')
            require(len(cp)>=len(poses),'Incomplete prediction support')
            cj,ce=smplx_positions(model,cp[:len(poses)],ct[:len(poses)])
            for role in roles: save_features(cj,out/'candidate'/role,sid)
        records.append(dict(source_id=sid,official_split=split,frames=len(poses),native_sha256=digest,
            full_model_check_max_m=error,candidate_sha256=exports[sid]['smplx_sha256'] if exports else None,
            candidate_full_model_check_max_m=ce if exports else None,
            candidate_arm=exports[sid].get('arm') if exports else None))
        print(json.dumps(dict(completed=len(records),total=len(rows),source_id=sid)),flush=True)
    sys.path.insert(0,str(UP/'Bailando/utils'))
    metric=load_file('aist_smplx_author_metric',UP/'Bailando/utils/metrics_new.py'); scores={}
    for domain in ('smplx',)+(('candidate',) if exports else ()):
        for role in score_roles:
            scores[domain+'_'+role+'_vs_full1363']=metric.quantized_metrics(str(out/domain/role),str(reference))
    write_json(out/'result.json',dict(status='SCORER_RUN_FINISHED',verified_at=datetime.now(timezone.utc).isoformat(),
        known_metric_issues=known_issues('AIST++'),
        protocol=dict(id=PROTOCOL_ID,time=protocol(),joint_map=MAP24,reference='full1363',
                      reference_sources=1363,reference_receipt_sha256=sha(reference/'reference_receipt.json'),
                      reference_cap_frames=None,evaluation_cap_frames=600,shape='zero20',hands='default',
                      evaluation_split=a.evaluation_split,
                      training_membership='TRAIN952 unchanged; reference is not training data',
                      published_smpl60_score_comparable=False),metrics=scores,sources=records,formal_official_result=False))
    run=json.loads((out/'run.json').read_text())
    run.update(status='SCORER_RUN_FINISHED',finished_at=datetime.now(timezone.utc).isoformat(),
               result_sha256=sha(out/'result.json'))
    write_json(out/'run.json',run)

if __name__=='__main__': main()
