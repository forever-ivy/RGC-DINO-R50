# External Code

This directory is for third-party source trees used by the project but not owned by this repository.

Expected DINO location:

```text
external/IDEA-Research-DINO/
```

Recommended source:

```text
https://github.com/IDEA-Research/DINO
```

Do not commit the cloned DINO source tree, pretrained weights, compiled CUDA artifacts, checkpoints, or generated logs. Keep only small project-owned manifests and documentation in git.

The phase 1 integration checks expect the external DINO tree to contain:

- `main.py`
- `models/dino/`
- `models/dino/ops/`
