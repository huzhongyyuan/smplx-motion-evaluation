"""Unit regressions; these checks do not claim benchmark or conversion acceptance."""
import subprocess
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parent

if __name__=='__main__':
    raise SystemExit(subprocess.call([sys.executable,'-m','unittest','-v',
        'test_contracts','test_aist30','test_hml_sources','test_metric_transport','test_hml_time',
        'test_release_metadata','test_beat_report','test_gt_results','test_current_protocols'],cwd=ROOT))
