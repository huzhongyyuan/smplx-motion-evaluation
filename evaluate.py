"""CPU-only evaluation: python evaluate.py COMMAND --help."""
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
COMMANDS = dict(configure='configure.py', fetch='fetch_upstream.py', check='check_release.py',
                score='score_worker.py', adapt='adapters.py', plan='hml_plan.py',
                report='report.py', gt='gt.py', **{'lodge-gt':'lodge_gt.py',
                'prepare-beat2024':'prepare_beat2024.py','aist-smplx':'aist_smplx.py','hml-time':'hml_time_probe.py'})

def main():
    if len(sys.argv)<2 or sys.argv[1] not in COMMANDS:
        raise SystemExit(__doc__+'\nCommands: '+', '.join(COMMANDS))
    env=os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2',
               MKL_NUM_THREADS='2', NUMBA_NUM_THREADS='2', PYTHONDONTWRITEBYTECODE='1', MPLBACKEND='Agg')
    if hasattr(os,'sched_getaffinity'):
        os.sched_setaffinity(0,set(sorted(os.sched_getaffinity(0))[:2]))
    raise SystemExit(subprocess.call([sys.executable,str(ROOT/COMMANDS[sys.argv[1]]),*sys.argv[2:]],cwd=ROOT,env=env))

if __name__=='__main__': main()
