#!/usr/bin/env bash
# Run Co-DETR test inference, convert to competition TXT/ZIP, and promote only
# when local validation evidence is supplied by the caller.

set -eo pipefail

ROOT=${ROOT:-/data1/liuxuan/projects/RGC-DINO-R50}
cd "$ROOT"

CONFIG=${CONFIG:-configs/codetr_internimage_l_aic2026_test.py}
CHECKPOINT=${CHECKPOINT:?set CHECKPOINT to a Co-DETR .pth file}
VAL_MAP=${VAL_MAP:?set VAL_MAP to the validated fold metric backing this candidate}
VAL_MAP50=${VAL_MAP50:-}
REASON=${REASON:?set REASON explaining why this candidate should be submitted}
CANDIDATE_KIND=${CANDIDATE_KIND:-codetr_internimage_l}
SCORE_THRESHOLD=${SCORE_THRESHOLD:-0.0}
MAX_DETECTIONS=${MAX_DETECTIONS:-100}
RUN_ID=${RUN_ID:-$(date +%Y%m%d_%H%M%S)}
OUT_DIR=${OUT_DIR:-outputs/codetr/aic2026_test_${CANDIDATE_KIND}_${RUN_ID}}
ZIP_PATH=${ZIP_PATH:-outputs/codetr/${CANDIDATE_KIND}_valmap${VAL_MAP}_${RUN_ID}.zip}
COCO_ANN=${COCO_ANN:-outputs/codetr_coco/aic2026_test/annotations/instances_test2017.json}
DATASET_ROOT=${DATASET_ROOT:-source/AIC2026_PHASE_1_1000}
SPLIT_MANIFEST=${SPLIT_MANIFEST:-outputs/splits/split_manifest.json}

. /data1/miniconda3/etc/profile.d/conda.sh
conda activate /data1/liuxuan/envs/codetr

export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-1}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-1}
export NUMEXPR_NUM_THREADS=${NUMEXPR_NUM_THREADS:-1}
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

mkdir -p "$OUT_DIR" outputs/codetr outputs/submissions

python scripts/export_codetr_test_coco.py \
  --dataset-root "$DATASET_ROOT" \
  --output-root outputs/codetr_coco/aic2026_test

PYTHONPATH=external/Co-DETR:${PYTHONPATH:-} python external/Co-DETR/tools/test.py \
  "$CONFIG" \
  "$CHECKPOINT" \
  --out "$OUT_DIR/results.pkl" \
  --format-only \
  --cfg-options data.test_dataloader.samples_per_gpu=1 data.test_dataloader.workers_per_gpu=0 \
  > "$OUT_DIR/test.log" 2>&1

PYTHONPATH=src python scripts/codetr_results_to_submission.py \
  --dataset-root "$DATASET_ROOT" \
  --coco-ann "$COCO_ANN" \
  --results-pkl "$OUT_DIR/results.pkl" \
  --output-dir "$OUT_DIR/submission_txt" \
  --zip-path "$ZIP_PATH" \
  --checkpoint-path "$CHECKPOINT" \
  --config-path "$CONFIG" \
  --split-manifest "$SPLIT_MANIFEST" \
  --score-threshold "$SCORE_THRESHOLD" \
  --max-detections "$MAX_DETECTIONS" \
  | tee "$OUT_DIR/conversion_summary.jsonl"

PREDICTION_OBJECTS=$(python - <<PY
import json
from pathlib import Path
last = None
for line in Path('$OUT_DIR/conversion_summary.jsonl').read_text().splitlines():
    line=line.strip()
    if line.startswith('{'):
        last=json.loads(line)
print(last.get('prediction_objects', 0) if last else 0)
PY
)

PROMOTE_ARGS=(
  "$ZIP_PATH"
  --submissions-dir outputs/submissions
  --reason "$REASON"
  --local-map "$VAL_MAP"
  --leaderboard-baseline "${LEADERBOARD_BASELINE:-48.335}"
  --manifest-path "${ZIP_PATH%.zip}.manifest.json"
  --candidate-kind "$CANDIDATE_KIND"
  --checkpoint-path "$CHECKPOINT"
  --train-dir "${TRAIN_DIR:-$(dirname "$CHECKPOINT")}"
  --val-map-50-95 "$VAL_MAP"
  --prediction-objects "$PREDICTION_OBJECTS"
  --score-threshold "$SCORE_THRESHOLD"
  --config-path "$CONFIG"
  --split-manifest "$SPLIT_MANIFEST"
  --source-sweep-ranking-json "${SOURCE_SWEEP_RANKING_JSON:-outputs/codetr/submission_contract_sweep/epoch11_fold0_val/submission_param_ranking.json}"
  --force
)
if [ -n "$VAL_MAP50" ]; then
  PROMOTE_ARGS+=(--val-map-50 "$VAL_MAP50")
fi
PYTHONPATH=src python scripts/promote_submission_candidate.py "${PROMOTE_ARGS[@]}"

echo "promoted candidate: $ZIP_PATH"
