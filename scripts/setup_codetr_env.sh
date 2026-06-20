#!/bin/sh
# Create an isolated Co-DETR/OpenMMLab environment.
# This does not modify the existing /data1/liuxuan/envs/py310 RGC-DINO env.

set -eu

ENV_PREFIX=${ENV_PREFIX:-/data1/liuxuan/envs/codetr}
CONDA=${CONDA:-/data1/miniconda3/condabin/conda}

if [ ! -x "$CONDA" ]; then
  echo "conda not found at $CONDA" >&2
  exit 2
fi

CONDA_BASE=$($CONDA info --base)
if [ ! -f "$CONDA_BASE/etc/profile.d/conda.sh" ]; then
  echo "conda activation script not found under $CONDA_BASE" >&2
  exit 2
fi

if [ ! -d "$ENV_PREFIX" ]; then
  "$CONDA" create -y -p "$ENV_PREFIX" python=3.10
fi

. "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$ENV_PREFIX"

python -m pip install --upgrade pip
python -m pip install 'setuptools==60.2.0' wheel

# Co-DETR is based on MMDetection 2.x / MMCV 1.x.  Use an isolated, older
# OpenMMLab stack instead of polluting the main PyTorch 2.12 RGC-DINO env.
# CUDA 11.7 wheels run on the server's newer NVIDIA driver and have matching
# mmcv-full wheels for torch 1.13.
python -m pip install \
  torch==1.13.1+cu117 \
  torchvision==0.14.1+cu117 \
  --extra-index-url https://download.pytorch.org/whl/cu117

python -m pip install openmim
mim install "mmcv-full==1.7.0"

# Keep legacy OpenMMLab dependencies on versions compatible with torch 1.13 / MMCV 1.x.
python -m pip install \
  'numpy<2' \
  'opencv-python<4.9' \
  'yapf==0.32.0' \
  'rich==13.4.2' \
  matplotlib pycocotools six terminaltables fairscale scipy timm fvcore tensorboard einops

# Co-DETR's local mmdet package can be imported through PYTHONPATH=external/Co-DETR;
# avoid editable install/build isolation, which may try to rebuild or fetch deps.
PYTHONPATH=/data1/liuxuan/projects/RGC-DINO-R50/external/Co-DETR:${PYTHONPATH:-} \
  python /data1/liuxuan/projects/RGC-DINO-R50/scripts/check_codetr_environment.py
