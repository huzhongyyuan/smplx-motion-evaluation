#!/usr/bin/env python3
"""Stream-check the actual compressed archives against the verified snapshot."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path, PurePosixPath
import subprocess
import tarfile


def sha_stream(stream):
    h = hashlib.sha256()
    for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
        h.update(block)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--snapshot', type=Path, required=True)
    p.add_argument('--assets', type=Path, required=True)
    p.add_argument('--aist-reference', type=Path, required=True)
    p.add_argument('--zstd', default='zstd')
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    index = json.loads((a.assets/'package_index.json').read_text())
    checked = {}
    motion_files = 0
    for label in ('AISTpp', 'FineDance', 'BEAT2', 'AISTpp-reference1363'):
        prefix = label+'.tar.zst.part-'
        parts = [a.assets/r['name'] for r in index['assets'] if r['name'].startswith(prefix)]
        assert parts and [x.name for x in parts] == [prefix+f'{i:03}' for i in range(len(parts))]
        source = subprocess.Popen(['cat', '--', *map(str, parts)], stdout=subprocess.PIPE)
        decoder = subprocess.Popen([a.zstd, '-T1', '-d', '-c'], stdin=source.stdout, stdout=subprocess.PIPE)
        source.stdout.close()
        root = a.aist_reference.parent if label.endswith('reference1363') else a.snapshot
        expected_motion = set()
        if not label.endswith('reference1363'):
            for row in map(json.loads, (a.snapshot/label/'manifest.jsonl').read_text().splitlines()):
                expected_motion.update(row[k+'_path'] for k in ('soma', 'smplx', 'hands') if row.get(k+'_path'))
        else:
            receipt = json.loads((a.aist_reference/'reference_receipt.json').read_text())
            expected_motion = {a.aist_reference.name+'/'+rel for rel in receipt['files']}
            expected_motion.add(a.aist_reference.name+'/reference_receipt.json')
        seen = set()
        files = 0
        try:
            with tarfile.open(fileobj=decoder.stdout, mode='r|') as archive:
                for member in archive:
                    path = PurePosixPath(member.name)
                    assert not path.is_absolute() and '..' not in path.parts and not member.issym() and not member.islnk()
                    if member.isdir():
                        continue
                    assert member.isfile() and member.name not in seen
                    seen.add(member.name)
                    original = root/member.name
                    assert original.is_file() and not original.is_symlink()
                    with original.open('rb') as f:
                        expected_sha = sha_stream(f)
                    stream = archive.extractfile(member)
                    assert stream is not None and sha_stream(stream) == expected_sha, member.name
                    files += 1
            # Consume the trailer so zstd's checksum and exit status are checked.
            assert not decoder.stdout.read().strip(b'\0')
            assert decoder.wait() == 0 and source.wait() == 0
        finally:
            decoder.stdout.close()
            for process in (decoder, source):
                if process.poll() is None:
                    process.terminate()
                process.wait()
        if label.endswith('reference1363'):
            assert seen == expected_motion
        else:
            common = set()
            for name in ('run.json', 'acceptance.json', 'archive_verification.json',
                         'membership.json', 'manifest.jsonl', 'code', 'README.md',
                         'NOTICE.md', 'licenses', 'reuse.json', 'metrics'):
                item = a.snapshot/name
                if item.is_dir():
                    common.update(str(x.relative_to(a.snapshot)) for x in item.rglob('*') if x.is_file())
                elif item.is_file():
                    common.add(name)
            assert seen == expected_motion | common | {label+'/manifest.jsonl'}
            motion_files += len(expected_motion)
        checked[label] = files
        print(json.dumps(dict(archive=label, status='PASS', files=files)), flush=True)
    assert motion_files == 7450
    result = dict(status='ALL_ARCHIVED_BYTES_MATCH_VERIFIED_SNAPSHOT',
                  verified_at=datetime.now(timezone.utc).isoformat(),
                  archives=checked, motion_files=motion_files,
                  verifier_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    with a.output.open('x') as f:
        json.dump(result, f, indent=2)
        f.write('\n')


if __name__ == '__main__':
    main()
