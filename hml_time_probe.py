"""512 official TRAIN sources: author20 vs feature-time20->30->20 control.

Uses unchanged author T2M scorer, batch32/drop_last, Diversity300, seed112.
One repetition is a diagnostic, never an official TEST score or confidence interval.
"""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from contracts import read_manifest,require
from evaluator_common import setup,write_json
from hml_time import roundtrip20_30_20
from score_worker import humanml
from source_lock import sha

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--manifest',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();setup()
    rows=[r for r in read_manifest(a.manifest) if r['dataset']=='HumanML3D' and r['split']=='train' and r.get('frames',0)>=84][:512]
    require(len(rows)==512,'Need512 eligible official TRAIN sources')
    out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
    for folder in ('input20','feature30','back20','score'): (out/folder).mkdir()
    selected=[]; diagnostics=[]
    for r in rows:
        sid=r['source_id'];x=np.load(r['native_path'],allow_pickle=False)
        require(sha(r['native_path'])==r['native_sha256'],'Source digest changed')
        up,back,original=roundtrip20_30_20(x)
        np.save(out/'input20'/(sid+'.npy'),original)
        np.save(out/'feature30'/(sid+'.npy'),up)
        np.save(out/'back20'/(sid+'.npy'),back)
        selected.append(dict(r,native_path=str(out/'input20'/(sid+'.npy')),native_sha256=sha(out/'input20'/(sid+'.npy')),prediction_path=str(out/'back20'/(sid+'.npy'))))
        diagnostics.append(dict(source_id=sid,author_sha256=r['native_sha256'],frames20=len(x),frames30=len(up),common_frames20=len(back),max_abs_channel_error=float(np.abs(original-back).max())))
    args=SimpleNamespace(output=out/'score',arm='time_resample_control',seed=112,replications=1,generated_records=None,condition_plan=None)
    result=humanml(args,selected)
    write_json(out/'result.json',dict(status='DIAGNOSTIC_FINISHED',dataset='HumanML3D',samples=512,split='train',
        protocol='author263D20 -> physical feature transport30 -> author263D20 on common support',
        direct30_as20_scoring=False,formal_official_result=False,verified_at=datetime.now(timezone.utc).isoformat(),
        limitations=['One repetition on TRAIN, not official TEST20rep.','Feature-space interpolation is not skeletal/mesh conversion validation.'],
        result=result,transport=diagnostics))

if __name__=='__main__':main()
