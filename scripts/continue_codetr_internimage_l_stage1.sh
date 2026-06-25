#!/usr/bin/env bash
# Continue the current Co-DETR + InternImage-L fold0 run from the best validated
# checkpoint in a background/tmux-safe way, then evaluate and promote only if the
# new validation metric beats the supplied baseline.

set -eo pipefail

ROOT=${ROOT:-/data1/liuxuan/projects/RGC-DINO-R50}
cd "$ROOT"

BASELINE_VAL_MAP=${BASELINE_VAL_MAP:-0.4379615851682616}
BASELINE_VAL_MAP50=${BASELINE_VAL_MAP50:-0.6249825407716256}
LEADERBOARD_BASELINE=${LEADERBOARD_BASELINE:-50.353}
BASE_CKPT=${BASE_CKPT:-outputs/codetr/internimage_l_stage1_1gpu_12ep_fold0_20260619_151708/best_bbox_mAP_epoch_11.pth}
CONFIG=${CONFIG:-configs/codetr_internimage_l_mm_config.py}
RUN_ID=${RUN_ID:-$(date +%Y%m%d_%H%M%S)}
WORK_DIR=${WORK_DIR:-outputs/codetr/internimage_l_stage1_continue_ep24_fold0_${RUN_ID}}
LOG_DIR=${LOG_DIR:-/data1/liuxuan/logs}
LOG_FILE=${LOG_FILE:-$LOG_DIR/codetr-internimage-l-continue-${RUN_ID}.log}
MAX_EPOCHS=${MAX_EPOCHS:-24}
LR_STEPS=${LR_STEPS:-[18,22]}

mkdir -p "$WORK_DIR" "$LOG_DIR" outputs/codetr
printf 'run_id=%s\nwork_dir=%s\nlog_file=%s\nbase_ckpt=%s\nconfig=%s\n' \
  "$RUN_ID" "$WORK_DIR" "$LOG_FILE" "$BASE_CKPT" "$CONFIG" \
  > outputs/codetr/internimage_l_stage1_continue_latest.txt

exec > "$LOG_FILE" 2>&1

. /data1/miniconda3/etc/profile.d/conda.sh
conda activate /data1/liuxuan/envs/codetr

export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-1}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-1}
export NUMEXPR_NUM_THREADS=${NUMEXPR_NUM_THREADS:-1}
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

printf '[%s] continuing Co-DETR InternImage-L\n' "$(date --iso-8601=seconds)"
python scripts/check_codetr_integration.py \
  --codetr-root external/Co-DETR \
  --internimage-weights /data1/liuxuan/checkpoints/internimage/internimage_l_public_pretrain.pth \
  --codetr-weights /data1/liuxuan/checkpoints/codetr/codetr_internimage_l_public_pretrain.pth \
  --require-weights

python scripts/export_codetr_coco.py --fold 0 --output-root outputs/codetr_coco/fold0 --clip-labels
PYTHONPATH=external/Co-DETR:${PYTHONPATH:-} python scripts/check_codetr_environment.py \
  --codetr-root external/Co-DETR \
  --config "$CONFIG"

# Resume preserves optimizer/scheduler from BASE_CKPT.  This is intentional for
# a conservative continuation; if it underperforms, next run should load model
# weights only with a fresh optimizer.
PYTHONPATH=external/Co-DETR:${PYTHONPATH:-} python external/Co-DETR/tools/train.py \
  "$CONFIG" \
  --work-dir "$WORK_DIR" \
  --resume-from "$BASE_CKPT" \
  --cfg-options \
    data.workers_per_gpu=0 \
    runner.max_epochs="$MAX_EPOCHS" \
    lr_config.step="$LR_STEPS" \
    checkpoint_config.max_keep_ckpts=4 \
    log_config.interval=50

BEST_LINE=$(grep 'Epoch(val)' "$WORK_DIR"/*.log | tail -1 || true)
printf 'last_val_line=%s\n' "$BEST_LINE"

BEST_CKPT=$(ls -t "$WORK_DIR"/best_bbox_mAP_epoch_*.pth 2>/dev/null | head -1)
if [ -z "$BEST_CKPT" ]; then
  BEST_CKPT=$(ls -t "$WORK_DIR"/epoch_*.pth 2>/dev/null | head -1)
fi
VAL_OUT="$WORK_DIR/final_txt_val_eval"
mkdir -p "$VAL_OUT"
PYTHONPATH=external/Co-DETR:${PYTHONPATH:-} python external/Co-DETR/tools/test.py \
  "$CONFIG" \
  "$BEST_CKPT" \
  --out "$VAL_OUT/results.pkl" \
  --eval bbox \
  --work-dir "$VAL_OUT" \
  --cfg-options data.test_dataloader.samples_per_gpu=1 data.test_dataloader.workers_per_gpu=0

PYTHONPATH=src python scripts/sweep_codetr_submission_params.py \
  --results-pkl "$VAL_OUT/results.pkl" \
  --coco-ann outputs/codetr_coco/fold0/annotations/instances_val2017.json \
  --labels source/训练集/labels \
  --sample-ids-file outputs/splits/fold0_val_ids.txt \
  --output-dir "$VAL_OUT/submission_contract_sweep" \
  --thresholds 0 0.001 0.003 0.005 0.01 0.02 0.03 0.05 0.08 0.1 \
  --max-detections 80 100 \
  --top-k 10

BEST_JSON=$(python - <<PY
import json
from pathlib import Path
ranking=Path('$VAL_OUT/submission_contract_sweep/submission_param_ranking.json')
rows=json.loads(ranking.read_text())
print(json.dumps(rows[0], sort_keys=True))
PY
)
printf 'best_submission_contract=%s\n' "$BEST_JSON"
NEW_MAP=$(python - <<PY
import json
print(json.loads('''$BEST_JSON''')['map_50_95'])
PY
)
NEW_MAP50=$(python - <<PY
import json
print(json.loads('''$BEST_JSON''')['map_50'])
PY
)
NEW_THR=$(python - <<PY
import json
print(json.loads('''$BEST_JSON''')['score_threshold'])
PY
)
NEW_TOPK=$(python - <<PY
import json
print(json.loads('''$BEST_JSON''')['max_detections'])
PY
)

python - <<PY
import sys
new=float('$NEW_MAP')
base=float('$BASELINE_VAL_MAP')
if new <= base:
    print(f'No promotion: final-TXT val mAP {new:.6f} <= baseline {base:.6f}')
    sys.exit(3)
print(f'Promotion allowed: final-TXT val mAP {new:.6f} > baseline {base:.6f}')
PY

RUN_ID_TEST="continue_${RUN_ID}_map${NEW_MAP}"
CHECKPOINT="$BEST_CKPT" \
CONFIG=configs/codetr_internimage_l_aic2026_test.py \
VAL_MAP="$NEW_MAP" \
VAL_MAP50="$NEW_MAP50" \
SCORE_THRESHOLD="$NEW_THR" \
MAX_DETECTIONS="$NEW_TOPK" \
RUN_ID="$RUN_ID_TEST" \
CANDIDATE_KIND=codetr_internimage_l_continue \
TRAIN_DIR="$WORK_DIR" \
SOURCE_SWEEP_RANKING_JSON="$VAL_OUT/submission_contract_sweep/submission_param_ranking.json" \
REASON="Co-DETR InternImage-L continuation improved strict final-TXT fold0 val mAP from ${BASELINE_VAL_MAP} to ${NEW_MAP}; threshold=${NEW_THR}, topK=${NEW_TOPK}." \
bash scripts/run_codetr_test_and_promote.sh

printf '[%s] continuation workflow finished\n' "$(date --iso-8601=seconds)"
