# Verification scope

Existing GT evidence is in `results/`; exact recipe/result hashes are indexed in `results/gt_index.json`. The release does not replace source experiment evidence with unit-test success.

## Metric diagnostics, 2026-11-03

The evaluation host completed the original36 core checks plus four physical
kinetic checks, then the extended five-check kinetic suite including rejection
of old feature vectors and incorrect/missing FPS. Four BEAT2024 regressions
passed with the pinned real author assets: exact historical BC compatibility,
caller-array immutability, repeat scoring and correctly cropped threshold indices.
Execution used CPU0/1 with CUDA hidden. See [metric corrections](docs/METRIC_CORRECTIONS.md)
for the distinction between frozen paper scores and corrected diagnostics.

The real FineDance19 identity control is in
[fd19_physical_identity.json](results/fd19_physical_identity.json).
Both arms contain the same19 preprocessed GT motions; both feature sets were
re-extracted. Their Divk values are both11.14637233221043. The unchanged author
FID routine returns −7.83825066719146e−6 − 4.658130792098806e−6i on these identical
features; the numerical residual is retained rather than clipped to zero.
This verifies the diagnostic's identity control, **not SOMA conversion parity**.
This corrected Divk cannot be compared with the author's published202-pool Divk.

The reported13-source SOMA FineDance FIDk133.297 has not been reproduced here:
its exact converted motion files, frozen converter and full source manifest
have not been located. No result in this release establishes that discrepancy's
cause or that SOMA GT/tokenizer outputs match native SMPL-X quality.

## Current single-protocol release

Verified2026-11-02 23:51 PST (2026-11-03T07:51:21Z) on the existing evaluation host, CPU0/1 with CUDA hidden. [current_protocol_validation.json](results/current_protocol_validation.json) binds the tested source hashes and checks. **36 core regressions and3 author-asset regressions passed.** The registry has exactly four recipes with no duplicate datasets. BEAT2 defaults to EMAGE2024/speaker2, rejects the former version/fullCSV selectors, and the current reporter rejects missing/previous BEAT profiles. AIST++ uses only actual SMPL-X30 scoring; its numerical FK/feature functions are unchanged.

Three real AIST sources matched their archived geometry and both feature types exactly. Two real FineDance TRAIN sources (001/002) passed the original LODGE preprocessing and202-source scoring before/after branch removal: feature files were byte-identical and all scores exactly equal. These small controls verify this code migration, not a new full-data or model-quality acceptance. Current archived AIST scores retain exactly the two original SMPL-X branches and all992 source records. Old source/results retain their hashes in [legacy/migration.json](legacy/migration.json).

## Previous full GT migration evidence

Previous release verified2026-11-02 23:00 PST (2026-11-03T07:00:36Z). Its [migration receipt](legacy/previous_release/results/public_gt_migration.json) records that release's code hashes, final source-lock identity, actual metrics and migration checks. **29 core regressions and3 historical-asset regressions passed for that release.** All95 runtime/source-evidence files were also fetched afresh from their pinned author URLs and verified on the packaging machine. The old531-file lock included unused training assets/example motions; all retained author-file hashes are unchanged. GloVe/evaluator weights are checked separately as10 local asset files. This historical receipt is not a hash receipt for subsequently edited source files.

| Check | Scope |
|---|---|
| Archived HumanML3D | Final20 repetitions, sourceTEST4198, loader4648, batch-evaluated4640; means and95%CI independently recomputed from per-replication values |
| Archived AIST++ | Full992 sources, strictTRAIN952, TEST20 and separateTEST+VAL40; current result retains exact SMPL-X scores and sources; full former SMPL comparison is in legacy/ |
| Public AIST geometry control | Three real sources; geometry compared with full SMPL-X, both feature types compared with archived original features; see `results/public_aist_migration_control.json` |
| FineDance | Public runner rechecked404 feature hashes and recomputed202-pool Divk9.728691849938494/Divg7.441712841724804, identical to archived GT |
| BEAT2 | Public runner completed all15 sources,28680 motionframes/26880 BCframes; aggregateBC0.6867171335676296/L1div13.070166015625 and every source's scores/clocks/input hashes exactly matched the accepted archive |

Release checks: `python evaluate.py check`. Historical-asset regression: `python -m unittest -v test_beat2024` after `python evaluate.py prepare-beat2024`. These test source coverage, SO(3)/time/hand contracts, archived statistics and profile separation. They do not establish generation quality.

The portable package reused the existing frozen source catalog and legally obtained datasets/assets. An attempted cold30582-file rescan was stopped with its log preserved; this release does not claim a new full-data audit. A fresh-machine full four-dataset download-and-rerun has not been performed. Full HML20 and AIST992 were reused rather than rerun for packaging. SOMA/tokenizer fidelity, AIST paperSMPL60 equivalence, FineDance final paper inference-member identity and exact historical BEAT paper-environment identity remain outside these verified claims.
