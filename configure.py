"""Materialize a local runtime from public IDs plus legitimately obtained assets."""
import argparse
from collections import Counter
import json
from pathlib import Path
from source_lock import ROOT, sha

def dump(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2)+'\n')

def link(target,source):
    source=Path(source).expanduser().resolve()
    if not source.exists(): raise ValueError('Missing user-provided input: '+str(source))
    if target.exists() or target.is_symlink():
        if target.resolve()!=source: raise ValueError('Refuse changing existing runtime link: '+str(target))
        return
    target.parent.mkdir(parents=True,exist_ok=True); target.symlink_to(source,target_is_directory=source.is_dir())

def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--config',type=Path,required=True); a=p.parse_args()
    config=json.loads(a.config.read_text())
    for name,source in config['links'].items():
        target=ROOT/name
        if name.startswith('/') or '..' in Path(name).parts or not (name.startswith('assets/') or name.startswith('data/') or name=='upstream/T2M/glove'):
            raise ValueError('Unsupported runtime link: '+name)
        link(target,source)
    expected=json.loads((ROOT/'assets.expected.json').read_text())
    frozen={}
    for name,digest in expected['files'].items():
        acceptable=[digest]+expected.get('tensor_equivalent_alternatives',{}).get(name,[])
        actual=sha(ROOT/name)
        if actual not in acceptable: raise ValueError('Not the measured asset: '+name)
        frozen[name]=actual
    dump(ROOT/'assets.lock.json',dict(files=frozen))
    rows=[json.loads(s) for s in (ROOT/'metadata/members.jsonl').read_text().splitlines()]
    for r in rows:
        ds,sid=r['dataset'],r['source_id']
        suffix={'HumanML3D':'.npy','AIST++':'.pkl','FineDance':'.npy','BEAT2':'.npz'}[ds]
        r['native_path']=str(ROOT/'data'/ds/'motions'/(sid+suffix))
        if ds=='BEAT2': r['audio_path']=str(ROOT/'data/BEAT2/audio'/(sid+'.wav'))
        path=Path(r['native_path'])
        if not path.is_file(): raise ValueError('Missing official member: '+str(path))
        actual=sha(path)
        if r.get('native_sha256',actual)!=actual: raise ValueError('Author native bytes changed: '+sid)
        r['native_sha256']=actual
    text=''.join(json.dumps(r)+'\n' for r in rows)
    for relative in ('membership_v1/manifest.jsonl','native_sources_v3/manifest.jsonl'):
        path=ROOT/relative
        if path.exists() and path.read_text()!=text: raise ValueError('Runtime source manifest differs; use a separate checkout')
        path.parent.mkdir(parents=True,exist_ok=True); path.write_text(text)
    dump(ROOT/'native_sources_v3/receipt.json',dict(status='AUTHOR_NATIVE_CATALOG_FROZEN',manifest_sha256=sha(ROOT/'native_sources_v3/manifest.jsonl')))
    profile=json.loads((ROOT/'metadata/beat_speaker2.json').read_text())
    speaker_ids=set(sum(profile.values(),[]))
    selected=[r for r in rows if r['dataset']!='BEAT2' or r['source_id'] in speaker_ids]
    path=ROOT/'protocol_profiles_v1/emage_author_speaker2.jsonl'; path.parent.mkdir(exist_ok=True)
    path.write_text(''.join(json.dumps(r)+'\n' for r in selected))
    dump(path.parent/'receipt.json',dict(author_profile_manifest_sha256=sha(path)))
    print(json.dumps({ds:dict(Counter(r['split'] for r in rows if r['dataset']==ds)) for ds in ('HumanML3D','AIST++','FineDance','BEAT2')},indent=2))

if __name__=='__main__': main()
