#!/bin/bash
# Swin-L训练脚本：2×RTX 3090优化版本
# 使用方法: bash scripts/train_swin_l.sh fold0

set -e

# 参数
FOLD=${1:-fold0}
CONFIG=configs/swin_l_stage1.yaml
OUTPUT_BASE=outputs/rgc_dino/swin_l_${FOLD}_$(date +%Y%m%d_%H%M)

# 检查权重文件
WEIGHT_FILE=weights/swin_large_patch4_window7_224_22k.pth
if [ ! -f "$WEIGHT_FILE" ]; then
    echo "错误: 预训练权重不存在: $WEIGHT_FILE"
    echo "请先运行: bash scripts/download_swin_weights.sh"
    exit 1
fi

# 检查fold划分文件
FOLD_FILE=outputs/splits/${FOLD}_val_ids.txt
if [ ! -f "$FOLD_FILE" ]; then
    echo "错误: Fold划分文件不存在: $FOLD_FILE"
    exit 1
fi

# 创建输出目录
mkdir -p $OUTPUT_BASE
mkdir -p logs

# 日志文件
LOG_FILE=logs/train_swin_l_${FOLD}_$(date +%Y%m%d_%H%M).log

echo "=========================================="
echo "Swin-L训练开始"
echo "=========================================="
echo "Fold: $FOLD"
echo "配置: $CONFIG"
echo "输出: $OUTPUT_BASE"
echo "日志: $LOG_FILE"
echo "预训练权重: $WEIGHT_FILE"
echo "显卡数: 2"
echo "=========================================="

# 启动训练（2卡分布式）
python -m torch.distributed.launch \
    --nproc_per_node=2 \
    --master_port=29500 \
    scripts/train_rgc_dino.py \
    --config $CONFIG \
    --output-dir $OUTPUT_BASE \
    --fold $FOLD \
    --train-ids outputs/splits/${FOLD}_train_ids.txt \
    --val-ids outputs/splits/${FOLD}_val_ids.txt \
    --quality-cache outputs/cache/quality_features_train.json \
    --epochs 36 \
    --lr 2e-5 \
    --lr-backbone 5e-6 \
    --batch-size 1 \
    --grad-accumulation 16 \
    --amp \
    --clip-labels \
    2>&1 | tee $LOG_FILE

echo "=========================================="
echo "训练完成！"
echo "检查输出: $OUTPUT_BASE"
echo "查看日志: $LOG_FILE"
echo "=========================================="

# 自动评估最佳checkpoint
echo "开始评估checkpoint..."
python scripts/select_best_checkpoint.py \
    --train-dir $OUTPUT_BASE \
    --val-ids outputs/splits/${FOLD}_val_ids.txt \
    --quality-cache outputs/cache/quality_features_train.json \
    --clip-labels \
    --amp

echo "=========================================="
echo "全部完成！"
echo "=========================================="
