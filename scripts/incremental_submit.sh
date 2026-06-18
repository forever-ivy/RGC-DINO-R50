#!/bin/bash
# Thin wrapper for the Python incremental submission pipeline.
# Usage: bash scripts/incremental_submit.sh <train_dir> [extra args...]

set -euo pipefail

if [ "$#" -lt 1 ]; then
    echo "Usage: bash scripts/incremental_submit.sh <train_dir> [extra args...]" >&2
    exit 2
fi

TRAIN_DIR="$1"
shift

python scripts/run_incremental_submission_pipeline.py \
    --train-dir "$TRAIN_DIR" \
    --once \
    --auto-promote \
    "$@"
