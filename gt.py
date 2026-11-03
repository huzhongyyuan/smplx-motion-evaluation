"""Frozen GT recipes corresponding to results/gt_index.json; no protocol selection by score."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent


def command(dataset, output):
    if dataset == 'HumanML3D':
        args = ['score', '--dataset', dataset, '--split', 'test', '--arm', 'native_gt',
                '--manifest', 'native_sources_v3/manifest.jsonl', '--seed', '112', '--replications', '20']
    elif dataset == 'AIST++':
        args = ['aist-smplx', '--manifest', 'membership_v1/manifest.jsonl']
    elif dataset == 'FineDance':
        args = ['lodge-gt']
    elif dataset == 'BEAT2':
        args = ['score', '--dataset', dataset, '--split', 'test', '--arm', 'native_gt',
                '--manifest', 'protocol_profiles_v1/emage_author_speaker2.jsonl',
                '--membership-profile', 'emage_author_speaker2', '--beat-profile', 'emage2024_bc_l1']
    else:
        raise ValueError('Unknown dataset: '+dataset)
    return [sys.executable, str(ROOT/'evaluate.py'), *args, '--output', str(output)]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset', choices=('HumanML3D', 'AIST++', 'FineDance', 'BEAT2'))
    p.add_argument('--output', type=Path)
    p.add_argument('--list', action='store_true', help='Print frozen recipes/results without loading any data')
    args = p.parse_args()
    if args.list:
        print((ROOT/'results/gt_index.json').read_text())
        return
    if args.dataset is None or args.output is None:
        p.error('--dataset and --output required unless --list')
    raise SystemExit(subprocess.call(command(args.dataset, args.output), cwd=ROOT))


if __name__ == '__main__':
    main()
