#!/bin/bash
# 监控训练完成并自动评估提交
# 用法: bash scripts/monitor_and_submit.sh

TRAIN_DIR="outputs/rgc_dino/swin_l_fold0_20260616_0126"
CHECK_INTERVAL=1800  # 30分钟检查一次

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "训练监控和自动提交脚本"
echo "训练目录: $TRAIN_DIR"
echo "检查间隔: $CHECK_INTERVAL 秒 (30分钟)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

LAST_CHECKPOINT_COUNT=0

while true; do
    # 检查训练进程是否还在运行
    if ! ps aux | grep "train_rgc_dino.py" | grep -v grep | grep -q "$TRAIN_DIR"; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] 训练进程已退出或完成"
        
        # 等待一段时间确保checkpoint写入完成
        sleep 60
        
        # 检查是否有新的checkpoint
        CHECKPOINT_COUNT=$(ls $TRAIN_DIR/checkpoint*.pth 2>/dev/null | grep -v "checkpoint.pth$" | wc -l)
        
        if [ "$CHECKPOINT_COUNT" -gt "$LAST_CHECKPOINT_COUNT" ]; then
            echo "[$(date '+%Y-%m-%d %H:%M:%S')] 发现新checkpoint，开始评估..."
            bash scripts/auto_submit_best.sh $TRAIN_DIR
            LAST_CHECKPOINT_COUNT=$CHECKPOINT_COUNT
        fi
        
        # 如果训练完成（有36个checkpoint），退出监控
        if [ "$CHECKPOINT_COUNT" -ge 36 ]; then
            echo "[$(date '+%Y-%m-%d %H:%M:%S')] 训练完全完成（36个epochs）"
            break
        fi
    else
        # 训练还在进行中，检查是否有新checkpoint
        CHECKPOINT_COUNT=$(ls $TRAIN_DIR/checkpoint*.pth 2>/dev/null | grep -v "checkpoint.pth$" | wc -l)
        
        if [ "$CHECKPOINT_COUNT" -gt "$LAST_CHECKPOINT_COUNT" ]; then
            echo "[$(date '+%Y-%m-%d %H:%M:%S')] 训练进行中，已完成 $CHECKPOINT_COUNT 个epochs"
            
            # 如果有足够的checkpoint，评估并提交最佳的
            if [ "$CHECKPOINT_COUNT" -ge 12 ]; then
                echo "[$(date '+%Y-%m-%d %H:%M:%S')] 已有12+个checkpoint，评估最佳模型..."
                bash scripts/auto_submit_best.sh $TRAIN_DIR
            fi
            
            LAST_CHECKPOINT_COUNT=$CHECKPOINT_COUNT
        fi
    fi
    
    sleep $CHECK_INTERVAL
done

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "监控完成"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
