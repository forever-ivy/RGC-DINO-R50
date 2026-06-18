#!/bin/bash
# Thin wrapper for evaluating a training directory once and promoting the best candidate.
# Platform submission is handled separately by scripts/monitor_competition.py.
# Usage: bash scripts/auto_submit_best.sh <train_dir> [extra args...]

set -euo pipefail

if [ "$#" -lt 1 ]; then
    echo "Usage: bash scripts/auto_submit_best.sh <train_dir> [extra args...]" >&2
    exit 2
fi

TRAIN_DIR="$1"
shift

python scripts/run_incremental_submission_pipeline.py \
    --train-dir "$TRAIN_DIR" \
    --once \
    --auto-promote \
    "$@"
