#!/bin/bash
# 增量评估和提交：每个新checkpoint完成后立即评估并提交最佳
# 用法: bash scripts/incremental_submit.sh <训练目录>

set -e

TRAIN_DIR=$1
if [ -z "$TRAIN_DIR" ]; then
    echo "错误: 请提供训练目录"
    exit 1
fi

STATE_FILE="$TRAIN_DIR/submit_state.json"
SUBMISSION_LOG="outputs/submissions/submission_log.txt"

# 初始化状态文件
if [ ! -f "$STATE_FILE" ]; then
    echo '{"last_processed_epoch": -1, "best_epoch": -1, "best_map": 0, "last_submitted_epoch": -1}' > "$STATE_FILE"
fi

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "增量评估和提交 - $(date '+%Y-%m-%d %H:%M:%S')"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# 读取状态
LAST_PROCESSED=$(python3 -c "import json; print(json.load(open('$STATE_FILE'))['last_processed_epoch'])")
BEST_EPOCH=$(python3 -c "import json; print(json.load(open('$STATE_FILE'))['best_epoch'])")
BEST_MAP=$(python3 -c "import json; print(json.load(open('$STATE_FILE'))['best_map'])")
LAST_SUBMITTED=$(python3 -c "import json; print(json.load(open('$STATE_FILE'))['last_submitted_epoch'])")

echo "当前状态："
echo "  已处理到: epoch $LAST_PROCESSED"
echo "  当前最佳: epoch $BEST_EPOCH, mAP=$BEST_MAP"
echo "  上次提交: epoch $LAST_SUBMITTED"
echo ""

# 查找新的checkpoint
NEW_CHECKPOINTS=$(ls $TRAIN_DIR/checkpoint*.pth 2>/dev/null | grep -v "checkpoint.pth$" | sort)

for ckpt in $NEW_CHECKPOINTS; do
    EPOCH=$(basename $ckpt | grep -o '[0-9]\{4\}')
    EPOCH=$((10#$EPOCH))  # 去掉前导零
    
    if [ "$EPOCH" -le "$LAST_PROCESSED" ]; then
        continue  # 已处理过
    fi
    
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "评估 epoch $EPOCH"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    
    # 评估当前checkpoint
    EVAL_OUTPUT=$(python scripts/eval_checkpoint.py \
        --checkpoint $ckpt \
        --fold 0 2>&1)
    
    CURRENT_MAP=$(echo "$EVAL_OUTPUT" | grep "mAP:" | awk '{print $2}')
    
    echo "  Epoch $EPOCH: mAP=$CURRENT_MAP"
    
    # 更新最佳模型
    IS_NEW_BEST=0
    if (( $(echo "$CURRENT_MAP > $BEST_MAP" | bc -l) )); then
        echo "  ✓ 新的最佳模型！"
        BEST_EPOCH=$EPOCH
        BEST_MAP=$CURRENT_MAP
        IS_NEW_BEST=1
    else
        echo "  - 未超过当前最佳 (epoch $BEST_EPOCH: $BEST_MAP)"
    fi
    
    # 更新状态
    python3 << EOF
import json
with open('$STATE_FILE', 'r') as f:
    state = json.load(f)
state['last_processed_epoch'] = $EPOCH
state['best_epoch'] = $BEST_EPOCH
state['best_map'] = $BEST_MAP
with open('$STATE_FILE', 'w') as f:
    json.dump(state, f)
EOF
    
    # 如果是新的最佳模型，生成提交文件
    if [ "$IS_NEW_BEST" -eq 1 ] && [ "$BEST_EPOCH" != "$LAST_SUBMITTED" ]; then
        echo ""
        echo "=== 生成提交文件 ==="
        
        SUBMISSION_FILE="outputs/submissions/swin_l_fold0_ep${BEST_EPOCH}_$(date +%Y%m%d_%H%M).zip"
        
        python scripts/infer_rgc_dino.py \
            --checkpoint $TRAIN_DIR/checkpoint$(printf "%04d" $BEST_EPOCH).pth \
            --output $SUBMISSION_FILE
        
        echo "✓ 提交文件已生成: $SUBMISSION_FILE"
        
        # 记录
        echo "$(date '+%Y-%m-%d %H:%M:%S') | Swin-L fold0 | epoch=$BEST_EPOCH | val_mAP=$BEST_MAP | file=$SUBMISSION_FILE | status=READY" >> $SUBMISSION_LOG
        
        # 更新已提交状态
        python3 << EOF
import json
with open('$STATE_FILE', 'r') as f:
    state = json.load(f)
state['last_submitted_epoch'] = $BEST_EPOCH
with open('$STATE_FILE', 'w') as f:
    json.dump(state, f)
EOF
        
        echo ""
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        echo "⚠️  新的最佳模型提交文件已准备好"
        echo "    Epoch: $BEST_EPOCH"
        echo "    mAP: $BEST_MAP"
        echo "    文件: $SUBMISSION_FILE"
        echo ""
        echo "    等待手动提交到竞赛平台"
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    fi
    
    echo ""
done

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "增量评估完成"
echo "当前最佳: epoch $BEST_EPOCH, mAP=$BEST_MAP"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
