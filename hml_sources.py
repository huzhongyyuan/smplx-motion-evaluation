"""Give the unmodified author loader exactly the sources in its frozen manifest."""
from pathlib import Path
import numpy as np
from contracts import require
from evaluator_common import write_json
from source_lock import sha


def stage_hml_sources(rows, output):
    output = Path(output)
    output.mkdir(exist_ok=False)
    inputs = []
    for row in rows:
        sid = row['source_id']
        require(row['dataset'] == 'HumanML3D' and Path(sid).name == sid, 'Invalid HML identity')
        source = Path(row['native_path']).resolve()
        value = np.load(source, allow_pickle=False)
        require(value.ndim == 2 and value.shape[1] == 263 and 40 <= len(value) < 200 and
                np.isfinite(value).all(), f'{sid}: invalid native263D source')
        digest = sha(source)
        require('native_sha256' not in row or digest == row['native_sha256'], f'{sid}: frozen native bytes changed')
        (output/(sid+'.npy')).symlink_to(source)
        inputs.append(dict(source_id=sid, native_path=str(source), native_sha256=digest))
    write_json(output/'receipt.json', dict(status='MANIFEST_SOURCES_STAGED', inputs=inputs,
               policy='Original values and dtype; author loader receives manifest sources, no fallback to production motion_dir'))
    return output
