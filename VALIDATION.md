# Verification scope

Existing GT evidence is in `results/`; exact recipe/result hashes are indexed in `results/gt_index.json`. The release does not replace source experiment evidence with unit-test success.

Verified2026-11-02 23:00 PST (2026-11-03T07:00:36Z). [public_gt_migration.json](results/public_gt_migration.json) records code hashes, final source-lock identity, actual metrics and migration checks. **29 core regressions and3 historical-asset regressions passed.** All95 runtime/source-evidence files were also fetched afresh from their pinned author URLs and verified on the packaging machine. The old531-file lock included unused training assets/example motions; all retained author-file hashes are unchanged. GloVe/evaluator weights are checked separately as10 local asset files.

| Check | Scope |
|---|---|
| Archived HumanML3D | Final20 repetitions, sourceTEST4198, loader4648, batch-evaluated4640; means and95%CI independently recomputed from per-replication values |
| Archived AIST++ | Full992 sources, strictTRAIN952, TEST20 and separateTEST+VAL40; original result retained |
| Public AIST geometry control | Three real sources; geometry compared with full SMPL-X, both feature types compared with archived original features; see `results/public_aist_migration_control.json` |
| FineDance | Public runner rechecked404 feature hashes and recomputed202-pool Divk9.728691849938494/Divg7.441712841724804, identical to archived GT |
| BEAT2 | Public runner completed all15 sources,28680 motionframes/26880 BCframes; aggregateBC0.6867171335676296/L1div13.070166015625 and every source's scores/clocks/input hashes exactly matched the accepted archive |

Release checks: `python evaluate.py check`. Historical-asset regression: `python -m unittest -v test_beat2024` after `python evaluate.py prepare-beat2024`. These test source coverage, SO(3)/time/hand contracts, archived statistics and profile separation. They do not establish generation quality.

The portable package reused the existing frozen source catalog and legally obtained datasets/assets. An attempted cold30582-file rescan was stopped with its log preserved; this release does not claim a new full-data audit. A fresh-machine full four-dataset download-and-rerun has not been performed. Full HML20 and AIST992 were reused rather than rerun for packaging. SOMA/tokenizer fidelity, AIST paperSMPL60 equivalence, FineDance final paper inference-member identity and exact historical BEAT paper-environment identity remain outside these verified claims.
