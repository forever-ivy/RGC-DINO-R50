#!/usr/bin/env bash
# Evaluate a Co-DETR checkpoint under the strict final-TXT validation contract,
# then run test inference and promote it only if the contract metric improves.

set -eo pipefail

ROOT=${ROOT:-/data1/liuxuan/projects/RGC-DINO-R50}
cd "$ROOT"

CHECKPOINT=${CHECKPOINT:?set CHECKPOINT to a Co-DETR .pth file}
CONFIG=${CONFIG:-configs/codetr_internimage_l_mm_config.py}
TEST_CONFIG=${TEST_CONFIG:-configs/codetr_internimage_l_aic2026_test.py}
BASELINE_VAL_MAP=${BASELINE_VAL_MAP:-0.324389}
BASELINE_VAL_MAP50=${BASELINE_VAL_MAP50:-0.489903}
LEADERBOARD_BASELINE=${LEADERBOARD_BASELINE:-45.044}
RUN_ID=${RUN_ID:-$(date +%Y%m%d_%H%M%S)}
VAL_OUT=${VAL_OUT:-outputs/codetr/fasttrack_${RUN_ID}}
CANDIDATE_KIND=${CANDIDATE_KIND:-codetr_internimage_l_fasttrack}

mkdir -p "$VAL_OUT" outputs/codetr
printf 'run_id=%s\ncheckpoint=%s\nconfig=%s\ntest_config=%s\nval_out=%s\n' \
  "$RUN_ID" "$CHECKPOINT" "$CONFIG" "$TEST_CONFIG" "$VAL_OUT" \
  > outputs/codetr/codetr_fasttrack_latest.txt

. /data1/miniconda3/etc/profile.d/conda.sh
conda activate /data1/liuxuan/envs/codetr

export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-1}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-1}
export NUMEXPR_NUM_THREADS=${NUMEXPR_NUM_THREADS:-1}
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-1}

printf '[%s] fast-track validation for %s on CUDA_VISIBLE_DEVICES=%s\n' \
  "$(date --iso-8601=seconds)" "$CHECKPOINT" "$CUDA_VISIBLE_DEVICES" | tee "$VAL_OUT/fasttrack.log"

python scripts/export_codetr_coco.py --fold 0 --output-root outputs/codetr_coco/fold0 --clip-labels | tee -a "$VAL_OUT/fasttrack.log"

PYTHONPATH=external/Co-DETR:${PYTHONPATH:-} python external/Co-DETR/tools/test.py \
  "$CONFIG" \
  "$CHECKPOINT" \
  --out "$VAL_OUT/results.pkl" \
  --eval bbox \
  --work-dir "$VAL_OUT" \
  --cfg-options data.test_dataloader.samples_per_gpu=1 data.test_dataloader.workers_per_gpu=0 \
  > "$VAL_OUT/val_test.log" 2>&1

PYTHONPATH=src python scripts/sweep_codetr_submission_params.py \
  --results-pkl "$VAL_OUT/results.pkl" \
  --coco-ann outputs/codetr_coco/fold0/annotations/instances_val2017.json \
  --labels source/训练集/labels \
  --sample-ids-file outputs/splits/fold0_val_ids.txt \
  --output-dir "$VAL_OUT/submission_contract_sweep" \
  --thresholds 0 0.001 0.003 0.005 0.01 0.02 0.03 0.05 0.08 0.1 \
  --max-detections 80 100 \
  --top-k 10 \
  | tee "$VAL_OUT/sweep.log"

BEST_VALUES=$(python - <<PY
import json
from pathlib import Path
rows=json.loads(Path('$VAL_OUT/submission_contract_sweep/submission_param_ranking.json').read_text())
row=rows[0]
print(row['map_50_95'], row['map_50'], row['score_threshold'], row['max_detections'])
PY
)
read -r NEW_MAP NEW_MAP50 NEW_THR NEW_TOPK <<< "$BEST_VALUES"
printf 'best_submission_contract map=%s map50=%s threshold=%s topK=%s\n' \
  "$NEW_MAP" "$NEW_MAP50" "$NEW_THR" "$NEW_TOPK" | tee -a "$VAL_OUT/fasttrack.log"

python - "$NEW_MAP" "$BASELINE_VAL_MAP" <<'PY'
import sys
new=float(sys.argv[1])
base=float(sys.argv[2])
if new <= base:
    print(f'No promotion: final-TXT val mAP {new:.6f} <= baseline {base:.6f}')
    sys.exit(3)
print(f'Promotion allowed: final-TXT val mAP {new:.6f} > baseline {base:.6f}')
PY

TEST_RUN_ID="${RUN_ID}_map${NEW_MAP}_thr${NEW_THR}_top${NEW_TOPK}"
CHECKPOINT="$CHECKPOINT" \
CONFIG="$TEST_CONFIG" \
VAL_MAP="$NEW_MAP" \
VAL_MAP50="$NEW_MAP50" \
SCORE_THRESHOLD="$NEW_THR" \
MAX_DETECTIONS="$NEW_TOPK" \
RUN_ID="$TEST_RUN_ID" \
CANDIDATE_KIND="$CANDIDATE_KIND" \
TRAIN_DIR="$(dirname "$CHECKPOINT")" \
LEADERBOARD_BASELINE="$LEADERBOARD_BASELINE" \
SOURCE_SWEEP_RANKING_JSON="$VAL_OUT/submission_contract_sweep/submission_param_ranking.json" \
REASON="Co-DETR InternImage-L fast-track checkpoint improved strict final-TXT fold0 val mAP from ${BASELINE_VAL_MAP} to ${NEW_MAP}; threshold=${NEW_THR}, topK=${NEW_TOPK}." \
bash scripts/run_codetr_test_and_promote.sh | tee -a "$VAL_OUT/fasttrack.log"

printf '[%s] fast-track workflow finished\n' "$(date --iso-8601=seconds)" | tee -a "$VAL_OUT/fasttrack.log"
