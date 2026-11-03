"""Fetch exact files from their authors, retaining license notices. No model download."""
import argparse
import json
from pathlib import Path
import urllib.request
from source_lock import ROOT, sha

REPOS = {'Bailando':('github','lisiyao21/Bailando'), 'LODGE':('github','THUSIGSICLAB/LODGE'),
         'T2M':('github','EricGuo5513/text-to-motion'), 'emage_evaltools':('hf','haiyangliu0/emage_evaltools'),
         'pytorch3d':('github','facebookresearch/pytorch3d'), 'EMAGE_caller':('github','haiyangliu0/character-policy')}

def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--list-only',action='store_true'); args=p.parse_args()
    lock=json.loads((ROOT/'sources.lock.json').read_text())
    for rel,digest in lock['files'].items():
        _,repo,*parts=Path(rel).parts; relative='/'.join(parts)
        kind,origin=REPOS[repo]; rev=lock['revisions'][repo]
        if repo=='EMAGE_caller' and relative not in ('train_emage_audio.py','test_emage_audio.py'): relative='emage_utils/'+relative
        url=(f'https://raw.githubusercontent.com/{origin}/{rev}/{relative}' if kind=='github'
             else f'https://huggingface.co/{origin}/resolve/{rev}/{relative}')
        if args.list_only: print(rel,url,digest); continue
        target=ROOT/rel
        if target.exists():
            if sha(target)!=digest: raise ValueError('Existing file differs; not overwritten: '+rel)
            continue
        target.parent.mkdir(parents=True,exist_ok=True)
        temp=target.with_name(target.name+'.download')
        with urllib.request.urlopen(url,timeout=60) as response, temp.open('wb') as f:
            while block:=response.read(1024*1024): f.write(block)
        if sha(temp)!=digest: raise ValueError('Author file hash differs: '+rel)
        temp.rename(target)
    print('URL_LIST_ONLY' if args.list_only else 'UPSTREAM_BYTES_VERIFIED')

if __name__=='__main__': main()
