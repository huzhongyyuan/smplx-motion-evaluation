#!/usr/bin/env python3
"""Maintainer: publish SHA-verified Release parts with bounded resumable staging.

Uses the existing local gh authentication in memory, only for uploads.github.com.
No credentials are sent to the source server or stored in receipts. Each worker
stages at most one1GB part, resumes source interruptions, then verifies SHA
before upload. Existing assets require exact size+GitHub SHA; no overwrite.
"""
import argparse
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
import hashlib
import http.client
import json
from pathlib import Path
import shlex
import shutil
import subprocess
import time
from urllib.parse import quote


def gh_json(args):
    result=subprocess.run(['gh','api',*args],capture_output=True,text=True,check=True)
    return json.loads(result.stdout)


def sha(path):
    digest=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):digest.update(block)
    return digest.hexdigest()


def stage(row,args):
    target=args.cache_dir/row['name']
    partial=target.with_name(target.name+'.partial')
    if target.exists():
        assert target.stat().st_size==row['bytes'] and sha(target)==row['sha256']
        return target
    ssh=['ssh','-F','/dev/null','-o','BatchMode=yes','-o','ConnectTimeout=15',
         '-o','ProxyCommand=none','-o','ServerAliveInterval=15','-o','ServerAliveCountMax=3','-p',str(args.ssh_port)]
    if args.ssh_key:ssh+=['-i',str(args.ssh_key)]
    for attempt in range(1,4):
        offset=partial.stat().st_size if partial.exists() else 0
        assert 0<=offset<=row['bytes']<=1000000000
        assert shutil.disk_usage(args.cache_dir).free>row['bytes']-offset+1000000000
        print(json.dumps(dict(event='stage_started',name=row['name'],attempt=attempt,
                              resume_bytes=offset,total=row['bytes'])),flush=True)
        if offset==row['bytes']:break
        command=['rsync','-a','--partial','--append','--timeout=60',
                 '--rsync-path=taskset -c 0,1 rsync','-e',shlex.join(ssh),
                 args.ssh_host+':'+str(args.remote_dir/row['name']),str(partial)]
        process=subprocess.Popen(command,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
        last=time.monotonic()
        try:
            while process.poll() is None:
                time.sleep(1)
                if time.monotonic()-last>=25:
                    size=partial.stat().st_size if partial.exists() else offset
                    print(json.dumps(dict(event='stage_progress',name=row['name'],staged=size,total=row['bytes'])),flush=True)
                    last=time.monotonic()
            _,stderr=process.communicate()
        finally:
            if process.poll() is None:process.terminate();process.wait()
        if process.returncode==0:break
        error=stderr.decode(errors='replace')
        transient=any(text in error.lower() for text in ('timed out','broken pipe','connection reset',
                      'connection closed','connection unexpectedly closed','no route to host'))
        print(json.dumps(dict(event='stage_failed',name=row['name'],attempt=attempt,
                              exit_code=process.returncode,error=error)),flush=True)
        if not transient or attempt==3:raise RuntimeError('Source transfer failed; partial retained: '+str(partial))
        time.sleep(5)
    assert partial.stat().st_size==row['bytes'] and sha(partial)==row['sha256'],str(partial)
    partial.rename(target)
    print(json.dumps(dict(event='stage_verified',name=row['name'],bytes=row['bytes'])),flush=True)
    return target


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--index',type=Path,required=True)
    p.add_argument('--release-id',type=int,required=True)
    p.add_argument('--ssh-host',required=True)
    p.add_argument('--ssh-port',type=int,default=22)
    p.add_argument('--ssh-key',type=Path)
    p.add_argument('--remote-dir',type=Path,required=True)
    p.add_argument('--receipt',type=Path,required=True)
    p.add_argument('--cache-dir',type=Path,required=True,help='At most one1GB part per worker, removed only after GitHub verifies it')
    p.add_argument('--workers',type=int,choices=(1,2),default=1,
                   help='At most two streams; SSH readers remain bound to CPU0/1')
    a=p.parse_args()
    index=json.loads(a.index.read_text());repo=index['repo']
    assert repo=='huzhongyyuan/smplx-motion-evaluation' and index['release_tag']=='motion30-20261106'
    release=gh_json([f'repos/{repo}/releases/{a.release_id}?asset_snapshot={time.time_ns()}'])
    assert release['tag_name']==index['release_tag'] and release['draft'], 'Only this draft release may be populated'
    existing={x['name']:x for x in release['assets']}
    assert len({r['name'] for r in index['assets']})==len(index['assets'])
    a.cache_dir.mkdir(parents=True,exist_ok=True)
    assert not a.cache_dir.is_symlink()
    token=subprocess.run(['gh','auth','token','--hostname','github.com'],capture_output=True,text=True,check=True).stdout.strip()
    assert token
    def upload(row):
            name=row['name']
            assert Path(name).name==name and len(row['sha256'])==64
            if name in existing:
                asset=existing[name]
                assert asset['size']==row['bytes'] and asset['digest']=='sha256:'+row['sha256']
                assert asset['state']=='uploaded'
            else:
                staged=stage(row,a)
                source=staged.open('rb')
                connection=http.client.HTTPSConnection('uploads.github.com',timeout=180)
                start=time.monotonic();last=start;sent=0;digest=hashlib.sha256()
                print(json.dumps(dict(event='upload_started',name=name,bytes=row['bytes'])),flush=True)
                try:
                    connection.putrequest('POST',f'/repos/{repo}/releases/{a.release_id}/assets?name='+quote(name,safe=''))
                    connection.putheader('Authorization','Bearer '+token)
                    connection.putheader('Accept','application/vnd.github+json')
                    connection.putheader('User-Agent','motion-snapshot-publisher')
                    connection.putheader('Content-Type','application/octet-stream')
                    connection.putheader('Content-Length',str(row['bytes']))
                    connection.endheaders()
                    for block in iter(lambda:source.read(4*1024*1024),b''):
                        sent+=len(block);assert sent<=row['bytes']
                        digest.update(block);connection.send(block)
                        if time.monotonic()-last>=25:
                            print(json.dumps(dict(event='upload_progress',name=name,sent=sent,total=row['bytes'],
                                                  megabytes_per_second=sent/1e6/(time.monotonic()-start))),flush=True)
                            last=time.monotonic()
                    assert sent==row['bytes'] and digest.hexdigest()==row['sha256'], 'Source part changed'
                    response=connection.getresponse();payload=response.read()
                    assert response.status==201,(response.status,payload.decode(errors='replace'))
                    asset=json.loads(payload)
                    assert asset['size']==row['bytes'] and asset['digest']=='sha256:'+row['sha256']
                    assert asset['state']=='uploaded'
                finally:
                    source.close()
                    connection.close()
                staged.unlink()  # Disposable verified part; the original remains on40.
                print(json.dumps(dict(event='upload_verified',name=name,asset_id=asset['id'],
                                      bytes=sent,seconds=time.monotonic()-start)),flush=True)
            rec=dict(name=name,asset_id=asset['id'],bytes=asset['size'],sha256=row['sha256'],
                     github_digest=asset['digest'],url=asset['browser_download_url'],state=asset['state'])
            return rec
    completed=[]
    a.receipt.parent.mkdir(parents=True,exist_ok=True)
    rows=iter(sorted(index['assets'],key=lambda r:(r['name']!='SHA256SUMS',r['name'])))
    with a.receipt.open('x',buffering=1) as receipt, ThreadPoolExecutor(max_workers=a.workers) as pool:
        active=set()
        for _ in range(a.workers):
            row=next(rows,None)
            if row is not None:active.add(pool.submit(upload,row))
        try:
            while active:
                ready,_=wait(active,return_when=FIRST_COMPLETED)
                for future in ready:
                    rec=future.result()
                    receipt.write(json.dumps(rec)+'\n');completed.append(rec)
                    active.remove(future)
                    row=next(rows,None)
                    if row is not None:active.add(pool.submit(upload,row))
        except BaseException:
            for future in active:future.cancel()
            raise
    print(json.dumps(dict(status='ALL_DRAFT_ASSETS_UPLOADED_AND_DIGEST_VERIFIED',assets=len(completed),
                          bytes=sum(r['bytes'] for r in completed))),flush=True)


if __name__=='__main__':main()
