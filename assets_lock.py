"""Check licensed assets against the measured hashes; assets are not distributed."""
import json
from pathlib import Path
from source_lock import ROOT, sha

def verify():
    lock = json.loads((ROOT/'assets.lock.json').read_text())
    for name, digest in lock['files'].items():
        if sha(ROOT/name) != digest: raise ValueError('Asset bytes changed: '+name)
    return lock

if __name__ == '__main__': print(json.dumps(verify(), indent=2))
