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
| [AIST++](https://google.github.io/aistplusplus_dataset/) | Obtain and use under the dataset's own terms. Motion/music and derived motion arrays are not redistributed here. |
| [FineDance](https://github.com/li-ronghui/FineDance) | Obtain original and LODGE-preprocessed packages from author links; no new redistribution permission assumed. |
| [BEAT2](https://pantomatrix.github.io/BEAT/) | Obtain according to author dataset terms. Speech, text, motion and derived caches are excluded. |
| [SMPL](https://smpl.is.tue.mpg.de/) / [SMPL-X](https://smpl-x.is.tue.mpg.de/) | Licensed model assets must be obtained independently. Never included in this repository. |

This release contains split/ID/hash metadata and aggregate experimental results, not motion arrays, audio, text corpora, learned feature arrays, model weights, meshes, or body-model files. The user remains responsible for the applicable upstream and dataset conditions when fetching or rebuilding assets.

The source locks describe the bytes used, not an expanded license grant. All relevant authors should be cited when reporting results. Paper links and attribution appear in the Chinese evaluation document.
