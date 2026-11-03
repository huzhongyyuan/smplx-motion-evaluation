# Metric corrections — 2026-11-03

The fixed physical diagnostic and the frozen paper scorer answer different questions.
Never replace a historical score with a corrected score under the same label.

## FineDance and AIST++

Both pinned author feature implementations compute the first velocity as
`x[i] - x[i-1] / dt` instead of `(x[i] - x[i-1]) / dt`. Consequently, a static
joint at one metre produces an acceleration feature of 3540 at the default
1/60-second step. This channel depends on static geometry as well as motion.
The standard LODGE wrapper makes non-root joints root-relative, but that does
not remove their nonzero rest offsets. It does not establish that this defect
caused any particular SOMA score (including the reported 13-case FIDk133.297).

`evaluate.py corrected-kinetic` uses centered second differences divided by
dt squared and an explicit 1/30-second step for both benchmarks. It retains
the author's two-frame half-window, velocity-energy definitions, benchmark
joint/root convention, FID function and reference normalization. FineDance's
author kinetic default was 1/60 even for 30fps inputs; this diagnostic corrects
that clock as well. It is named **physical_kinetic30_v1**, not an official paper
score. Units are metres, Y-up. FineDance uses body22; AIST++ uses the fixed body24
evaluation map with default fingers.

```bash
python evaluate.py corrected-kinetic --dataset FineDance \
  --predictions /path/aligned_preprocessed_lodge139 \
  --reference /path/frozen_reference_preprocessed_lodge139 \
  --output runs/fd_physical_kinetic
```

Directories contain `.npy` motion139/319 already processed/cropped by LODGE,
or `.npy` joint sequences T×22×3. AIST++ accepts T×24×3 `.npy` or `.npz` with
`joints24` and explicit `fps=30`. Both feature distributions are always rebuilt.
Old one-dimensional feature files are rejected. FineDance caps both sides at
1024 frames; AIST++ caps predictions at600 and retains full reference sources.
Source IDs, file hashes, frame counts and the new protocol are recorded.

The author's released202-feature pool cannot be corrected from its feature
vectors. Re-extract the corresponding frozen motion sources for this diagnostic.
Do not compare the corrected score against the old202 features or published
FID/Div numbers. The published GT result files remain historical evidence.

The production `gt`/`score` paper-compatibility entries remain frozen, and now
report `known_metric_issues`; `report` labels old results too. Original fetched
source is a pinned compatibility dependency, not the physical diagnostic.

## BEAT2

BEAT2 does not use the dance acceleration function. The fixed2024 source has
two separate behaviors: L1div writes absolute mean deviations into its input,
then BC reads that array; BC also compares minima indices relative to a crop
against a velocity mask indexed over the full sequence.

The wrapper now performs historical L1/BC on a private copy, so callers retain
their original positions and repeated scoring is stable. Historical BC/L1
retain their established values. An additional diagnostic,
**BC_unmutated_aligned_threshold**, reads original positions and aligns the
threshold mask with the cropped minima. It retains the same audio onsets,
normalizer, threshold0.3, order7 and temporal support. It is reported separately
from `metrics`, with `paper_comparable=false`; it is not the accepted paper BC.
Only BEAT2 includes generated hands. FGD remains outside the current BC/L1
profile. No SOMA body/hand reconstruction acceptance follows from this fix.

## Acceptance

Regressions cover static nonzero poses, constant velocity, known quadratic
acceleration, 30/60fps time units, coordinate translation, both actual dance
extractors, invalid/old features, historical BC equality, caller immutability,
repeat scoring and crop-threshold indices. Real-input controls and deployment
evidence are linked from VALIDATION.md after execution.

SOMA parity still requires matched source IDs, crop/time/shape/joint conventions
and the same scorer/reference on both arms. Preserve per-source failures and
the full official TEST membership. Fixing an evaluator cannot by itself repair
a converter, tokenizer or dataset.

For the requested native-SMPL-X versus SOMA-GT check, freeze the converter and
export manifest before scoring. Compare three separate arms: native GT,
GT→SOMA→SMPL-X, and tokenizer reconstruction→SMPL-X. Report paired score changes
with the same reference pool, plus per-source body/root rotation, joint position,
velocity, acceleration and foot-contact differences. BEAT2 additionally requires
the matching arm's hand rotations; the other datasets keep default fingers.
Do not reduce19 sources to a convenient13-source intersection, inject native GT
back into a failed conversion, smooth away errors, or change the reference pool
to make the scores close. A13-source reproduction may diagnose the reported
failure, but cannot accept the full test set. Numerical/quality tolerances must
be fixed on TRAIN controls before inspecting the final TEST result; there is
currently no approved threshold or evidence of full SOMA parity.
