"""One frozen actual-SMPL-X30 reference for every current AIST++ entry."""
import json
from pathlib import Path
import numpy as np
from aist30 import protocol
from contracts import require
from source_lock import sha

ROOT = Path(__file__).resolve().parent
DEFAULT_REFERENCE = ROOT/'assets/aist_smplx30_reference1363'
KINDS = ('kinetic_features', 'manual_features_new')
PROTOCOL_ID = 'aist_smplx30_full1363_v1'


def verify_reference(path=DEFAULT_REFERENCE):
    path = Path(path).resolve()
    frozen = json.loads((ROOT/'metadata/aist_reference1363.json').read_text())
    receipt_path = path/'reference_receipt.json'
    require(sha(receipt_path) == frozen['receipt_sha256'], 'AIST reference receipt changed')
    receipt = json.loads(receipt_path.read_text())
    require(receipt['protocol_id'] == PROTOCOL_ID and receipt['samples'] == 1363,
            'AIST requires full1363 reference')
    require(receipt['feature_protocol'] == protocol() and receipt['domain'] == 'SMPL-X',
            'AIST reference time/model mismatch; SMPL or native60 features are invalid')
    ids = set(frozen['source_ids'])
    require(len(ids) == len(frozen['source_ids']) == 1363, 'Invalid frozen AIST membership')
    expected = {str(Path(k)/(sid+'.npy')) for k in KINDS for sid in ids}
    actual = {str(p.relative_to(path)) for k in KINDS for p in (path/k).iterdir()}
    require(set(receipt['files']) == actual == expected, 'Missing/extra AIST reference features')
    for relative, digest in receipt['files'].items():
        p = path/relative
        require(sha(p) == digest, 'AIST reference feature changed: '+relative)
        x = np.load(p, allow_pickle=False)
        require(x.shape == ((72,) if relative.startswith(KINDS[0]) else (32,))
                and np.isfinite(x).all(), 'Invalid AIST feature: '+relative)
    return path, receipt
