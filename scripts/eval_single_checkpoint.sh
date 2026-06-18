#!/bin/bash
# 评估单个checkpoint
CHECKPOINT=$1
FOLD=${2:-0}

echo "评估: $CHECKPOINT"

CUDA_VISIBLE_DEVICES=0 python scripts/train_rgc_dino.py \
    --config-file configs/swin_l_stage1.yaml \
    --output-dir /tmp/eval_temp \
    --fold $FOLD \
    --resume $CHECKPOINT \
    --quality-cache outputs/cache/quality_features_train.json \
    --val-batches 9999 \
    --amp \
    --smoke-only 2>&1 | tee /tmp/eval_output.txt

# 提取mAP结果
if grep -q "IoU" /tmp/eval_output.txt; then
    grep "IoU" /tmp/eval_output.txt | tail -5
fi
