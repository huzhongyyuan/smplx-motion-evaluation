"""Fetch the immutable historical author metric/normalizer; no credentials or retries."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import urllib.request

REVISION = '6ca70b9541285b124da2eeedcd80f7c5a54eb111'
ASSET_REVISION = '94f0ecb26dc66219b6a8216b9999d6b2eda505b5'
METRIC_SHA = 'ab2871867ab5f56eacd8e56b3fa39c488ccaff3df9f49ec8d673c03c3b30e8c5'
NORMALIZER_SHA = '53b5e48f2a7bf78c41a6de6395d6bb4f29018465ca5d0ee2820a2be3eebb7137'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path(__file__).resolve().parent/'assets/emage2024_v2')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    records = {}
    targets = [
        ('source/utils/metric.py', f'https://raw.githubusercontent.com/PantoMatrix/PantoMatrix/{REVISION}/scripts/EMAGE_2024/utils/metric.py', METRIC_SHA),
        ('mean_vel_smplxflame_30.npy', f'https://huggingface.co/datasets/haiyangliu0/BEAT2_Tools/resolve/{ASSET_REVISION}/emage_weights/mean_vel_smplxflame_30.npy', NORMALIZER_SHA)]
    for relative, url, expected in targets:
        with urllib.request.urlopen(url, timeout=30) as response:
            data = response.read()
        digest = hashlib.sha256(data).hexdigest()
        if digest != expected:
            raise ValueError('Author artifact SHA mismatch: '+relative)
        path = args.output/relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        # Only stable author URLs, never temporary signed download URLs.
        records[relative] = dict(sha256=digest, bytes=len(data), url=url)
    receipt = dict(status='AUTHOR_LEGACY_ASSETS_VERIFIED', verified_at=datetime.now(timezone.utc).isoformat(),
        source_repository='https://github.com/PantoMatrix/PantoMatrix', source_revision=REVISION,
        asset_repository='https://huggingface.co/datasets/haiyangliu0/BEAT2_Tools', asset_revision=ASSET_REVISION,
        files=records, scope='BC/L1 only; no legacy FGD weight or score required',
        normalizer_identity='Pinned author legacy archive; not a claim about the archived2024 environment.')
    (args.output/'receipt.json').write_text(json.dumps(receipt, indent=2)+'\n')
    print(json.dumps(receipt, indent=2))


if __name__ == '__main__':
    main()
