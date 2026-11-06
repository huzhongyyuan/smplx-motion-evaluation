#!/usr/bin/env python3
"""Download pinned motion Release assets, or verify an extracted snapshot.

No dataset/model credentials are read. Downloads are checked by SHA256 before
being renamed into place. Extraction is an explicit separate user action.
"""
import argparse
import hashlib
import json
from pathlib import Path
import urllib.request


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def contained(root, relative):
    p = root / relative
    if p.is_symlink() or not p.resolve().is_relative_to(root.resolve()):
        raise ValueError('Unsafe asset path: ' + relative)
    return p


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--index', type=Path, default=Path(__file__).resolve().parents[1]/'releases/motion30-20261106.json')
    p.add_argument('--dest', type=Path, default=Path('motion30-download'))
    p.add_argument('--dataset', choices=['AISTpp','FineDance','BEAT2'])
    p.add_argument('--verify-root', type=Path, help='Check every motion file in extracted per-dataset manifests')
    a = p.parse_args()
    if a.verify_root:
        count = 0
        for ds in ([a.dataset] if a.dataset else ['AISTpp','FineDance','BEAT2']):
            manifest = a.verify_root / ds / 'manifest.jsonl'
            for line in manifest.read_text().splitlines():
                row = json.loads(line)
                for key in ('soma','smplx','hands'):
                    if row.get(key+'_path'):
                        path = contained(a.verify_root, row[key+'_path'])
                        assert sha(path) == row[key+'_sha256'], str(path)
                        count += 1
        print(f'Verified {count} motion/hand files')
        return
    index = json.loads(a.index.read_text())
    a.dest.mkdir(parents=True, exist_ok=True)
    assets = [r for r in index['assets'] if not a.dataset or r.get('dataset') in (a.dataset, None)]
    for row in assets:
        target = contained(a.dest, row['name'])
        if target.exists():
            assert target.stat().st_size == row['bytes'] and sha(target) == row['sha256'], str(target)
            print('Verified existing ' + target.name, flush=True)
            continue
        partial = target.with_name(target.name+'.partial')
        if partial.exists():
            raise FileExistsError(f'Incomplete prior download: {partial}; inspect/remove it before retrying')
        with urllib.request.urlopen(row['url'], timeout=120) as source, partial.open('xb') as f:
            for block in iter(lambda: source.read(8 * 1024 * 1024), b''):
                f.write(block)
        assert partial.stat().st_size == row['bytes'] and sha(partial) == row['sha256'], str(partial)
        partial.rename(target)
        print('Downloaded and verified ' + target.name, flush=True)


if __name__ == '__main__':
    main()
