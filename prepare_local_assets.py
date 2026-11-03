"""Arrange legitimately obtained local evaluator assets without changing values."""
import argparse
import json
from pathlib import Path
from configure import link,dump
from source_lock import ROOT,sha

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--t2m-checkpoint',type=Path,required=True)
    p.add_argument('--mean',type=Path,required=True);p.add_argument('--std',type=Path,required=True)
    p.add_argument('--lodge-features',type=Path,required=True)
    a=p.parse_args()
    expected=json.loads((ROOT/'assets.expected.json').read_text())
    paths={'assets/t2m/t2m/text_mot_match/model/finest.tar':a.t2m_checkpoint,
           'assets/t2m/mean.npy':a.mean,'assets/t2m/std.npy':a.std}
    for relative,path in paths.items():
        acceptable=[expected['files'][relative]]+expected.get('tensor_equivalent_alternatives',{}).get(relative,[])
        if sha(path) not in acceptable:raise ValueError('Unexpected evaluator asset: '+str(path))
    reference=json.loads((ROOT/'metadata/lodge_reference.json').read_text())
    for relative,digest in reference['files'].items():
        if sha(a.lodge_features/relative)!=digest:raise ValueError('Published LODGE reference differs: '+relative)
    for relative,path in paths.items():link(ROOT/relative,path)
    for relative in reference['files']:link(ROOT/'assets/lodge_published_reference_v1'/relative,a.lodge_features/relative)
    dump(ROOT/'assets/lodge_published_reference_v1/reference_receipt.json',reference)
    proof=json.loads((ROOT/'metadata/t2m_provenance.json').read_text())
    proof['installed_checkpoint_sha256']=sha(a.t2m_checkpoint)
    dump(ROOT/'assets/t2m_author_verification.json',proof)
    print('LOCAL_ASSETS_HASH_VERIFIED_AND_LINKED')

if __name__=='__main__':main()
