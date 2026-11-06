# Paired SOMA30 / SMPL-X30 motion

Download: [motion30-20261106](https://github.com/huzhongyyuan/smplx-motion-evaluation/releases/tag/motion30-20261106).
Machine-readable checksums and asset URLs: [release index](../releases/motion30-20261106.json).
Published2026-11-06; all20 asset sizes/SHA256 and two small anonymous downloads
were verified. See the [publication receipt](../releases/motion30-20261106.publication.json).

| Dataset | TRAIN | VAL | TEST | Additional | Total |
|---|---:|---:|---:|---:|---:|
| AIST++ | 952 | 20 | 20 | 0 | 992 |
| FineDance | 183 | 0 | 19 | 0 | 202 |
| BEAT2 | 1093 | 106 | 265 | 156 | 1620 |

There are2814 paired motions. Official roles stay distinct; BEAT2 additional
is not automatically TRAIN. FineDance130 is included in the frozen LODGE
TEST19. AIST's1363-source FID reference is separate from the952-source TRAIN
and the992-motion admitted dataset. HumanML3D's body adaptation and source
coverage remain open, so it is not part of this motion release.

The project owner confirmed written author permission for all three datasets.
Original dataset terms and citations apply; see [THIRD_PARTY](../THIRD_PARTY.md)
and the archived NOTICE/licenses. The integration code's MIT license does not
relicense data. Source audio/text, models, meshes and weights are not included.

## Download and verify

```bash
python scripts/fetch_motion_snapshot.py --dest motion30-download
cd motion30-download
sha256sum -c SHA256SUMS
mkdir -p ../motion30
cat AISTpp.tar.zst.part-* | zstd -dc | tar -xf - -C ../motion30
cat FineDance.tar.zst.part-* | zstd -dc | tar -xf - -C ../motion30
cat BEAT2.tar.zst.part-* | zstd -dc | tar -xf - -C ../motion30
cat AISTpp-reference1363.tar.zst.part-* | zstd -dc | tar -xf - -C ../motion30
cd ..
python scripts/fetch_motion_snapshot.py --verify-root motion30
```

Parts are <=1GB and must be concatenated in their zero-padded order. Extracting
all three archives shares identical metadata. Per-dataset manifests contain
relative paths and SHA256 for all7450 motion/hand files. For a single dataset,
use `--dataset AISTpp`, `FineDance` or `BEAT2` with the download/verification
script. The aggregate SHA256SUMS also lists the other datasets' parts.

`SOMA30` motion arrays preserve source values. FineDance148 has one stale
estimated-frame flag corrected after native rotation/root verification;
`original_estimated_frame` and original/published SHA256 retain its provenance.
All other SOMA files preserve source bytes. `SMPLX30` has `poses[T,165]`, `trans[T,3]`,
betas, expression defaults,30FPS and actual `source_frame30`. Fingers remain
SMPL-X sidecars: BEAT2 native axis-angle, FineDance native52 row6 converted to
axis-angle, AIST neutral/unavailable. There is no SOMA hand retargeting.
Body uses the matched native transport inverse, composing Neck2 into Head;
there is no fitting, smoothing, endpoint extension or tokenizer inference.
World root is divided by declared scale with its fixed offset retained.
BEAT2 betas300 are retained; AIST/FineDance use explicit neutral zero20.
Face/expression defaults are labelled. These are Y-up meters.

FineDance TRAIN trimming is retained. This snapshot's restored FineDance211
has3261 frames and starts at source-frame0 (0 seconds). The earlier91 benchmark
used a different cropped211 starting at source-frame660 (22 seconds); its
archived scores are historical evidence, not a score of this exact snapshot.
Read each motion's recorded clock instead of assuming either origin. Audio is
obtained separately and must bind to that source clock. Transport integrity
does not establish full audiovisual or whole-corpus mesh quality.

## Use the existing evaluation pipeline

Obtain and configure author assets as described in [DATA_ACCESS](DATA_ACCESS.md).
Set config link `assets/aist_smplx30_reference1363` to the extracted
`aist_smplx30_full1363_v1`; the unchanged reference receipt and all2726 feature
hashes are checked. Generate an export manifest without changing data:

```bash
python scripts/snapshot_exports.py --root motion30 --dataset 'AIST++' --output runs/aist_snapshot_exports
python evaluate.py aist-smplx --manifest native_sources_v3/manifest.jsonl --exports runs/aist_snapshot_exports/exports.jsonl --output runs/aist_snapshot_score

python scripts/snapshot_exports.py --root motion30 --dataset FineDance --output runs/fd_snapshot_exports
python evaluate.py adapt --dataset FineDance --split test --manifest native_sources_v3/manifest.jsonl --exports runs/fd_snapshot_exports/exports.jsonl --arm soma_gt_roundtrip --output runs/fd_snapshot_adapt
python evaluate.py score --dataset FineDance --split test --manifest runs/fd_snapshot_adapt/manifest.jsonl --arm soma_gt_roundtrip --output runs/fd_snapshot_score

python scripts/snapshot_exports.py --root motion30 --dataset BEAT2 --output runs/beat_snapshot_exports
python evaluate.py adapt --dataset BEAT2 --split test --manifest protocol_profiles_v1/emage_author_speaker2.jsonl --exports runs/beat_snapshot_exports/exports.jsonl --arm soma_gt_roundtrip --output runs/beat_snapshot_adapt
python evaluate.py score --dataset BEAT2 --split test --manifest runs/beat_snapshot_adapt/manifest.jsonl --membership-profile emage_author_speaker2 --arm soma_gt_roundtrip --output runs/beat_snapshot_score
```

For AIST TEST20 only, pass `--evaluation-split test` to both the export and
scoring commands. The default40 retains TEST/VAL labels and reports TEST20
separately. BEAT's export helper selects the frozen speaker2 TEST15. Its hand
sidecars are explicitly GT-labelled and cannot be attached to generated arms.
Existing benchmark values cover AIST40, FineDance19 and BEAT15, not all2814
motions; packaging performs complete transport/hash/clock checks rather than
claiming a new whole-corpus perceptual benchmark. Build and verification
receipts travel with the archives. No training run is changed by the release.
