# SOMA / SMPL-X benchmark input contract

All four datasets use the same scorer for direct SMPL-X, SOMA roundtrip,
tokenizer reconstruction and model generation. The manifest identifies the arm,
source hashes, converter version, FPS, time origin, axes, units and checkpoint.
Changing the converter requires new exports and a new scoring directory; old
scores cannot be relabelled.

Preserve the original time origin. For example, an archive with
`source_frame30=[660,661,...]` starts at22s, not0s. The shared contract reads
source-frame/timestamp metadata and rejects a conflicting export origin;
cropped sources must not be shifted again when official preprocessing crops
the evaluation window. Archives without clock arrays require an explicit
origin from the caller.

Use the inverse matching the forward producer. The
[conversion repository](https://github.com/huzhongyyuan/soma-smplx-convert)
provides an explicit `scripts/convert_native.py` for declared native/global
rotation transport, separately from its surface-fitting converter. The native
path preserves mapped global orientations, composes reconstructed Neck2 into
Head and divides root coordinates by the recorded producer scale. It retains
fixed world offsets; it does not use source GT for fitting or frame correction.

Keep actual decoder rotations and hand output. Do not neutralize Neck2 to meet
a different converter's contract. Record unavailable fingers as neutral, and
do not present external trajectory bypass as tokenizer root reconstruction.
Shape is declared separately: the native path transports orientations rather
than preserving SOMA body proportions. Preserve all supplied BEAT2 betas;
the historical BEAT2 caller retains its own official geometry convention.

| Dataset | Current benchmark |
|---|---|
| AIST++ | Actual SMPL-X30; frozen full1363 reference; TEST20 and TEST+VAL40 separate; TRAIN952 unchanged |
| FineDance | LODGE official preprocessing and published202 reference; full TEST19 or explicitly partial diagnostics |
| BEAT2 | Frozen EMAGE2024 speaker2 TEST15 BC/L1, actual fingers; no FGD in this profile |
| HumanML3D | Author20FPS/263D, original T2M evaluator,20 repetitions; incomplete source/time coverage labelled separately |

30FPS AIST scores do not become the original SMPL60 paper scores by changing
the reference pool. FineDance missing130 and an available-source HML subset
are coverage limitations, not complete official test results. Keep failed runs
and old reference/mesh-fit results as historical evidence.
