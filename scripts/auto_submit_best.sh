#!/bin/bash
# 自动评估和提交最佳checkpoint
# 用法: bash scripts/auto_submit_best.sh <训练目录>

set -e

TRAIN_DIR=$1
if [ -z "$TRAIN_DIR" ]; then
    echo "错误: 请提供训练目录"
    echo "用法: bash scripts/auto_submit_best.sh outputs/rgc_dino/swin_l_fold0_20260616_0126"
    exit 1
fi

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "自动评估和提交最佳模型"
echo "训练目录: $TRAIN_DIR"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# 1. 找到所有checkpoint
CHECKPOINTS=$(ls $TRAIN_DIR/checkpoint*.pth 2>/dev/null | grep -v "checkpoint.pth$" | sort)
NUM_CHECKPOINTS=$(echo "$CHECKPOINTS" | wc -l)

echo "找到 $NUM_CHECKPOINTS 个checkpoint"
echo ""

if [ "$NUM_CHECKPOINTS" -eq 0 ]; then
    echo "没有找到checkpoint，退出"
    exit 0
fi

# 2. 评估所有checkpoint并选择最佳
echo "=== 评估所有checkpoint ==="
python scripts/select_best_checkpoint.py \
    --checkpoint-dir $TRAIN_DIR \
    --output-json $TRAIN_DIR/eval_results.json

# 3. 读取最佳checkpoint
BEST_EPOCH=$(python3 << 'EOF'
import json
import sys
with open(sys.argv[1]) as f:
    data = json.load(f)
print(data['best_epoch'])
EOF
"$TRAIN_DIR/eval_results.json")

BEST_MAP=$(python3 << 'EOF'
import json
import sys
with open(sys.argv[1]) as f:
    data = json.load(f)
print(f"{data['best_map']:.4f}")
EOF
"$TRAIN_DIR/eval_results.json")

echo ""
echo "✓ 最佳checkpoint: epoch $BEST_EPOCH, mAP=$BEST_MAP"
echo ""

# 4. 生成提交文件
echo "=== 生成提交文件 ==="
python scripts/infer_rgc_dino.py \
    --checkpoint $TRAIN_DIR/checkpoint$(printf "%04d" $BEST_EPOCH).pth \
    --output outputs/submissions/swin_l_fold0_ep${BEST_EPOCH}_$(date +%Y%m%d_%H%M).zip

SUBMISSION_FILE=$(ls -t outputs/submissions/swin_l_fold0_ep${BEST_EPOCH}_*.zip | head -1)

echo ""
echo "✓ 提交文件已生成: $SUBMISSION_FILE"
echo ""

# 5. 记录结果
echo "=== 记录结果 ==="
cat >> outputs/submissions/submission_log.txt << LOG_EOF
$(date '+%Y-%m-%d %H:%M:%S') | Swin-L fold0 | epoch=$BEST_EPOCH | val_mAP=$BEST_MAP | file=$SUBMISSION_FILE
LOG_EOF

echo "✓ 已记录到 outputs/submissions/submission_log.txt"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "提交文件准备完成，等待手动确认后提交"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
