"""Verify the exact upstream bytes used for the recorded evaluations."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''): h.update(block)
    return h.hexdigest()

def verify():
    lock = json.loads((ROOT/'sources.lock.json').read_text())
    for relative, digest in lock['files'].items():
        if sha(ROOT/relative) != digest: raise ValueError('Upstream bytes changed: '+relative)
    return lock['revisions']

if __name__ == '__main__': print(json.dumps(verify(), indent=2))
