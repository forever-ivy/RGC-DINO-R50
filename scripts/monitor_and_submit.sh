#!/bin/bash
# 监控训练并增量评估提交
# 每完成一个epoch就评估，发现更好的就生成提交文件

TRAIN_DIR="outputs/rgc_dino/swin_l_fold0_20260616_0126"
CHECK_INTERVAL=600  # 10分钟检查一次

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "增量监控和提交脚本"
echo "训练目录: $TRAIN_DIR"
echo "检查间隔: $CHECK_INTERVAL 秒 (10分钟)"
echo "策略: 每个新epoch立即评估，最佳模型生成提交文件"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

while true; do
    # 运行增量评估
    bash scripts/incremental_submit.sh $TRAIN_DIR
    
    # 检查训练是否完成
    CHECKPOINT_COUNT=$(ls $TRAIN_DIR/checkpoint*.pth 2>/dev/null | grep -v "checkpoint.pth$" | wc -l)
    
    if [ "$CHECKPOINT_COUNT" -ge 36 ]; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] 训练完全完成（36个epochs）"
        
        # 最后一次评估
        bash scripts/incremental_submit.sh $TRAIN_DIR
        break
    fi
    
    if ! ps aux | grep "train_rgc_dino.py" | grep -v grep | grep -q "$TRAIN_DIR"; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] 训练进程已退出"
        
        # 最后一次评估
        bash scripts/incremental_submit.sh $TRAIN_DIR
        break
    fi
    
    sleep $CHECK_INTERVAL
done

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "监控完成"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
