# Third-party sources and redistribution

The MIT license in this repository applies to the original integration code. It does not relicense datasets, pretrained weights, body models, upstream implementations, or their derived assets.

| Source | Observed terms / handling |
|---|---|
| [T2M](https://github.com/EricGuo5513/text-to-motion) | MIT source, Copyright2022 ChuanGuo. Fetched with original notices; checkpoint/data terms remain separate. |
| [Bailando](https://github.com/lisiyao21/Bailando) | S-Lab License1.0 with noncommercial restrictions. Not represented as unrestricted MIT/OSI software. |
| [LODGE](https://github.com/THUSIGSICLAB/LODGE) | No explicit license found in the fetched revision. Source/data are obtained from authors; no permission to redistribute is inferred from public availability. |
| [EMAGE evaltools](https://huggingface.co/haiyangliu0/emage_evaltools) | Author model-card metadata specifies Apache2.0. SMPL-X model files retain their separate body-model terms. |
| [EMAGE caller](https://github.com/haiyangliu0/character-policy) | Separate author repository pinned by revision and fileSHA; not vendored. |
| [EMAGE2024 source](https://github.com/PantoMatrix/PantoMatrix/tree/6ca70b9541285b124da2eeedcd80f7c5a54eb111/scripts/EMAGE_2024) / [author legacy assets](https://huggingface.co/datasets/haiyangliu0/BEAT2_Tools/tree/94f0ecb26dc66219b6a8216b9999d6b2eda505b5) | Historical BC/L1 source and normalizer fetched separately at immutable revisions, checked by SHA, not redistributed. Original terms apply. |
| [PyTorch3D](https://github.com/facebookresearch/pytorch3d) | Author implementation dependency, fetched with original notices. Only the required purePython source subset is locked. |
| [HumanML3D](https://github.com/EricGuo5513/HumanML3D) / [AMASS](https://amass.is.tue.mpg.de/) | HumanML3D authors explicitly state AMASS restrictions prevent directly redistributing data. Obtain through their documented process and underlying licenses. |
| [AIST++](https://google.github.io/aistplusplus_dataset/) | Original terms apply. Converted motion and the frozen handcrafted FID reference are in the separately authorised motion Release; music is excluded. |
| [FineDance](https://github.com/li-ronghui/FineDance) | Original noncommercial/No Distribution license is retained. This snapshot's redistribution relies on the project owner's explicit confirmation of prior written author permission; it does not grant further rights. |
| [BEAT2](https://pantomatrix.github.io/BEAT/) | Converted motion and native SMPL-X fingers are in the authorised snapshot; author dataset card declares Apache2.0. Speech and text are excluded. |
| [SMPL](https://smpl.is.tue.mpg.de/) / [SMPL-X](https://smpl-x.is.tue.mpg.de/) | Licensed model assets must be obtained independently. Never included in this repository. |

Git contains split/ID/hash metadata, integration scripts and aggregate results.
The separate [motion snapshot](docs/MOTION_RELEASE.md) contains converted arrays
and AIST handcrafted reference features, published on the project owner's
confirmation of written author authorisations for all three datasets on2026-11-06.
Original notices and FineDance's full license accompany the assets. This is not
a blanket sublicense. Audio, text corpora, learned features, model weights,
meshes and body-model files are excluded. Original terms remain applicable.

The source locks describe the bytes used, not an expanded license grant. All relevant authors should be cited when reporting results. Paper links and attribution appear in the Chinese evaluation document.
