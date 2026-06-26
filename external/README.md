# External Code

This directory is for third-party source trees used by the project but not owned by this repository. Do not commit cloned source trees, pretrained weights, compiled CUDA artifacts, checkpoints, generated logs, or downloaded archives. Keep only small project-owned manifests and documentation in git.

Current expected source trees:

```text
external/Co-DETR/              # current mainline detector/inference stack
external/InternImage-master/   # InternImage reference/source tree
external/IDEA-Research-DINO/   # legacy RGC-DINO/DINO-R50 fallback and comparison path
```

Current mainline:

- `external/Co-DETR/tools/train.py`
- `external/Co-DETR/tools/test.py`
- `external/Co-DETR/tools/dist_train.sh`
- `external/Co-DETR/mmdet/`
- `external/Co-DETR/projects/configs/` or `external/Co-DETR/configs/`
- `external/Co-DETR/ops_dcnv3/` for InternImage DCNv3 support

Legacy DINO integration checks expect:

- `external/IDEA-Research-DINO/main.py`
- `external/IDEA-Research-DINO/models/dino/`
- `external/IDEA-Research-DINO/models/dino/ops/`

Public pretrained weights must live under `/data1/liuxuan/checkpoints/`, not under `external/` or `/home/`.
