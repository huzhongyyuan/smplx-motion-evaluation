# Obtain datasets and assets

No raw author archive is stored in Git. A separately authorised three-dataset
converted-motion snapshot is distributed through GitHub Release; see
[motion download and verification](MOTION_RELEASE.md). It contains paired
SOMA30/SMPL-X30 motion and native SMPL-X finger parameters, not source audio,
text, body-model files or weights. Public metadata in `metadata/` identifies
the original sources and preserves official roles. Author assets below remain
necessary for the frozen benchmark adapters and scorers.

1. **HumanML3D**: follow [the authors' instructions](https://github.com/EricGuo5513/HumanML3D) for `new_joint_vecs` and `texts`. Use the exact author20fps263D values, not previously fitted SMPL/SMPL-X. `metadata/members.jsonl` pins all27,924 admitted source hashes. Obtain the T2M epoch28 matching evaluator from [the T2M README](https://github.com/EricGuo5513/text-to-motion); the verified author archive ID is `1IgrFCnxeg4olBtURUHimzS03ZI0df_6W`. Do not substitute a different public evaluator because it has the same filename. T2M GloVe assets are also obtained from that README.
2. **AIST++**: obtain official motion pickles, split and ignore files from [AIST++](https://google.github.io/aistplusplus_dataset/). The wrapper consumes native60fps SMPL parameters and samples every other frame to produce30fps, with official SMPL and SMPL-X assets. It does not require interpolated60fps predictions for scoring.
3. **FineDance**: obtain original52-joint motion and music from [FineDance](https://github.com/li-ronghui/FineDance), or preprocessing assets linked by [LODGE](https://github.com/THUSIGSICLAB/LODGE#readme). The verified `mofea319.rar` author reference archive ID is `1leTSqD6vyZRnM-yGUaAD-R013YlCbUTo`. `metadata/lodge_reference.json` pins its404 precomputed feature files. The nativeGT diagnostic follows LODGE loader filtering,19 sources; the20-source inference list including120 is preserved separately in `metadata/lodge_inference20.json` and is not silently equated to that diagnostic.
4. **BEAT2**: obtain English-v2 data, official CSV, native `smplxflame_30` and `wave16k` from the [authors](https://pantomatrix.github.io/BEAT/). Use actual30fps body+hands; fullTEST265 and current author speaker2TEST15 are separate.
5. **Body models and evaluator assets**: obtain `SMPL_MALE.pkl` and `SMPLX_NEUTRAL_2020.npz` under their original licenses. Obtain EMAGE's `AESKConv_240_100.bin` and `mean_vel_smplxflame_30.npy` from the [author evaltools](https://huggingface.co/haiyangliu0/emage_evaltools). The EMAGE-compatible model includes the author-required300shape/100expression support, whereas the body-only AIST adapter uses default20shape/10expression. A similarly named incompatible model is not accepted.

`assets.expected.json` records the measured hashes. The T2M original checkpoint and the three-network minimal repack were previously verified tensor-identical; both exact file hashes are accepted, and the local choice is frozen. `metadata/t2m_provenance.json` records the provenance without private paths.

## Preparing local asset layout

After legally downloading and extracting the packages, run:

```bash
python prepare_local_assets.py \
  --t2m-checkpoint /your-download/t2m/text_mot_match/model/finest.tar \
  --mean /your-download/t2m/Comp_v6_KLD01/meta/mean.npy \
  --std /your-download/t2m/Comp_v6_KLD01/meta/std.npy \
  --lodge-features /your-download/mofea319
```

The script creates local symlinks and hash-verified receipts in ignored `assets/`. It does not distribute assets or rewrite their numerical contents. Set the remaining data/model/GloVe paths in `config.local.json` using `config.example.json`. The HML reference-joints directory must contain the real author `000021.npy` with the fixed reference hash in `adapters.py`; it is needed only for conversion/preprocessing, not native263D scoring.

Run `python evaluate.py configure --config config.local.json`. Missing sources or mismatched hashes fail explicitly; do not fill, rename or delete cases to pass a gate. Keep generated local manifests and downloaded packages out of Git.

An independent fresh-machine full run has not been completed for this first release. `VALIDATION.md` separates actual migration checks from existing experimental results and open protocol differences.
