#!/usr/bin/env python3
"""Independently hash a completed snapshot, then produce <=1GB Release parts."""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import argparse
import collections
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
from datetime import datetime, timezone


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):
            h.update(block)
    return h.hexdigest()


def pack(root, entries, prefix, zstd_binary):
    tar = subprocess.Popen(['tar','--sort=name','-cf','-','-C',str(root),*entries],stdout=subprocess.PIPE)
    zstd = subprocess.Popen([zstd_binary,'-T1','-3','-c'],stdin=tar.stdout,stdout=subprocess.PIPE)
    tar.stdout.close()
    part = subprocess.Popen(['split','-b','1000000000','--numeric-suffixes=0','--suffix-length=3','-',str(prefix)],stdin=zstd.stdout)
    zstd.stdout.close()
    codes = [part.wait(),zstd.wait(),tar.wait()]
    assert codes == [0,0,0], codes


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--snapshot',type=Path,required=True)
    p.add_argument('--docs',type=Path,required=True)
    p.add_argument('--aist-reference',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--zstd',default='zstd',help='Explicit existing zstd executable; no environment changes')
    a=p.parse_args()
    assert len(os.sched_getaffinity(0))<=2
    acceptance=json.loads((a.snapshot/'acceptance.json').read_text())
    assert acceptance['status']=='FULL_SOURCE_TRANSPORT_INTEGRITY_VERIFIED' and acceptance['completed']==2814
    assert sha(a.snapshot/'manifest.jsonl')==acceptance['manifest_sha256']
    rows=list(map(json.loads,(a.snapshot/'manifest.jsonl').read_text().splitlines()))
    assert len(rows)==2814
    checked=0
    for ds in ('AISTpp','FineDance','BEAT2'):
        expected=set()
        rr=list(map(json.loads,(a.snapshot/ds/'manifest.jsonl').read_text().splitlines()))
        assert rr==[r for r in rows if ('AISTpp' if r['dataset']=='AIST++' else r['dataset'])==ds]
        for r in rr:
            for key in ('soma','smplx','hands'):
                if not r.get(key+'_path'):continue
                rel=r[key+'_path'];path=a.snapshot/rel
                assert path.resolve().is_relative_to(a.snapshot.resolve()) and not path.is_symlink()
                assert sha(path)==r[key+'_sha256'],rel
                expected.add(rel);checked+=1
        actual={str(p.relative_to(a.snapshot)) for p in (a.snapshot/ds).rglob('*.npz')}
        assert actual==expected,(ds,'missing or extra files')
    assert checked==7450
    a.output.mkdir(parents=True,exist_ok=False)
    verification=dict(verified_at=datetime.now(timezone.utc).isoformat(),status='ALL_PAIRED_RELEASE_FILES_INDEPENDENTLY_HASH_VERIFIED',
                      sources=2814,files=checked,manifest_sha256=sha(a.snapshot/'manifest.jsonl'),
                      acceptance_sha256=sha(a.snapshot/'acceptance.json'),packager_sha256=sha(Path(__file__)))
    (a.snapshot/'archive_verification.json').write_text(json.dumps(verification,indent=2)+'\n')
    for child in a.docs.iterdir():
        target=a.snapshot/child.name
        if target.exists():
            # Repack only from the same pinned documentation. Never silently
            # overwrite a prior snapshot's notices or licensing metadata.
            if child.is_dir():
                expected={str(p.relative_to(child)):sha(p) for p in child.rglob('*') if p.is_file()}
                actual={str(p.relative_to(target)):sha(p) for p in target.rglob('*') if p.is_file()}
                assert actual==expected, str(target)
            else:
                assert target.is_file() and sha(target)==sha(child), str(target)
            continue
        if child.is_dir():shutil.copytree(child,target)
        else:shutil.copy2(child,target)
    common=['run.json','acceptance.json','archive_verification.json','membership.json','manifest.jsonl','code','README.md','NOTICE.md','licenses']
    if (a.snapshot/'reuse.json').is_file():common.append('reuse.json')
    if (a.snapshot/'metrics').is_dir():common.append('metrics')
    assets=[]
    for ds in ('AISTpp','FineDance','BEAT2'):
        print('PACKING '+ds,flush=True)
        pack(a.snapshot,[ds,*common],a.output/(ds+'.tar.zst.part-'),a.zstd)
        print('PACKED '+ds,flush=True)
    ref=a.aist_reference
    assert sha(ref/'reference_receipt.json')=='a85be8b8db9db267cb45c9540d2fe5a0c9a139fe305dc0f7ebb2884a3d8e78cd'
    receipt=json.loads((ref/'reference_receipt.json').read_text())
    assert receipt['samples']==1363 and len(receipt['files'])==2726
    assert {str(p.relative_to(ref)) for kind in ('kinetic_features','manual_features_new') for p in (ref/kind).iterdir()}==set(receipt['files'])
    for relative,digest in receipt['files'].items():assert sha(ref/relative)==digest
    pack(ref.parent,[ref.name+'/'+part for part in ('reference_receipt.json','kinetic_features','manual_features_new')],
         a.output/'AISTpp-reference1363.tar.zst.part-',a.zstd)
    for path in sorted(a.output.iterdir()):
        assert path.stat().st_size<=1000000000
        name=path.name
        assets.append(dict(name=name,bytes=path.stat().st_size,sha256=sha(path),dataset='AISTpp' if name.startswith('AISTpp') else name.split('.')[0]))
    (a.output/'SHA256SUMS').write_text(''.join(r['sha256']+'  '+r['name']+'\n' for r in assets))
    sums=a.output/'SHA256SUMS'
    assets.append(dict(name=sums.name,bytes=sums.stat().st_size,sha256=sha(sums),dataset=None))
    result=dict(schema_version=1,release_tag='motion30-20261106',repo='huzhongyyuan/smplx-motion-evaluation',
                created_at=datetime.now(timezone.utc).isoformat(),status='PACKAGED_AND_HASHED_NOT_YET_UPLOADED',
                sources=2814,files=checked,counts=json.loads((a.snapshot/'membership.json').read_text())['counts'],assets=assets,
                acceptance_sha256=sha(a.snapshot/'acceptance.json'),manifest_sha256=sha(a.snapshot/'manifest.jsonl'),
                archive_verification_sha256=sha(a.snapshot/'archive_verification.json'))
    (a.output/'package_index.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
