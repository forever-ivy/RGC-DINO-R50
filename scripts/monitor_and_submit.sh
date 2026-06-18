#!/bin/bash
# Thin wrapper for continuously polling a training directory, evaluating new checkpoints,
# and promoting improved test-set candidates. Real submission is handled by
# scripts/monitor_competition.py.
# Usage: bash scripts/monitor_and_submit.sh <train_dir> [extra args...]

set -euo pipefail

if [ "$#" -lt 1 ]; then
    echo "Usage: bash scripts/monitor_and_submit.sh <train_dir> [extra args...]" >&2
    exit 2
fi

TRAIN_DIR="$1"
shift

python scripts/run_incremental_submission_pipeline.py \
    --train-dir "$TRAIN_DIR" \
    --auto-promote \
    "$@"
