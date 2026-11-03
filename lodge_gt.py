"""LODGE published202 GT pool, using the pinned author's quantized_metrics."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import sys

import numpy as np

from contracts import require
from evaluator_common import ROOT, UP, setup, load_file, write_json
from source_lock import sha


def verify_reference(root):
    frozen = json.loads((ROOT/'metadata/lodge_reference.json').read_text())
    require(frozen['samples'] == 202 and len(frozen['files']) == 404, 'Require author202 GT pool')
    actual = {str(p.relative_to(root)) for kind in ('kinetic_features', 'manual_features_new')
              for p in (root/kind).iterdir()}
    require(actual == set(frozen['files']), 'Missing/extra LODGE reference feature')
    for name, digest in frozen['files'].items():
        require(sha(root/name) == digest, 'Published LODGE feature changed: '+name)
        require(np.isfinite(np.load(root/name, allow_pickle=False)).all(), 'Nonfinite LODGE feature')
    return frozen


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reference-features', type=Path, default=ROOT/'assets/lodge_published_reference_v1')
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    setup()
    reference = args.reference_features.resolve()
    frozen = verify_reference(reference)
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    sys.path[:0] = [str(UP/'LODGE/metric'), str(UP/'LODGE')]
    source = UP/'LODGE/metric/metrics_finedance.py'
    metric = load_file('lodge_gt_author_metric', source)
    for name in ('PROTOCOL.md', 'sources.lock.json', 'lodge_gt.py'):
        shutil.copy2(ROOT/name, out/name)
    write_json(out/'run.json', dict(status='RUNNING', dataset='FineDance',
        started_at=datetime.now(timezone.utc).isoformat(),
        reference_metadata_sha256=sha(ROOT/'metadata/lodge_reference.json'),
        author_metric_sha256=sha(source), own_code_sha256=sha(__file__)))
    # Identity on the published pool: no motion re-extraction or new equations.
    scores = metric.quantized_metrics(str(reference), str(reference))
    result = dict(status='SCORER_RUN_FINISHED', dataset='FineDance', samples=202,
        profile='lodge_published202_gt', scope='author published feature pool, not TEST19',
        metrics=dict(Divk=scores['div_k_gt'], Divg=scores['div_m_gt']),
        identity_diagnostic=scores, paper_gt_fid=None,
        reference_metadata_sha256=sha(ROOT/'metadata/lodge_reference.json'),
        reference_files=frozen['files'], author_metric_sha256=sha(source),
        own_code_sha256=sha(__file__),
        raw_motion_feature_equality_verified=False)
    write_json(out/'result.json', result)
    write_json(out/'run.json', dict(status='SCORER_RUN_FINISHED', dataset='FineDance',
        finished_at=datetime.now(timezone.utc).isoformat(), result_sha256=sha(out/'result.json'),
        author_metric_sha256=sha(source), own_code_sha256=sha(__file__)))
    print(json.dumps(result['metrics']))


if __name__ == '__main__':
    main()
