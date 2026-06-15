#!/bin/bash
# 下载Swin-L预训练权重

set -e

WEIGHT_DIR=weights
WEIGHT_FILE=swin_large_patch4_window7_224_22k.pth
WEIGHT_URL="https://github.com/microsoft/Swin-Transformer/releases/download/v1.0.0/${WEIGHT_FILE}"

echo "=========================================="
echo "下载Swin-L预训练权重"
echo "=========================================="

# 创建目录
mkdir -p $WEIGHT_DIR

# 检查是否已存在
if [ -f "$WEIGHT_DIR/$WEIGHT_FILE" ]; then
    echo "权重文件已存在: $WEIGHT_DIR/$WEIGHT_FILE"
    echo "文件大小: $(du -h $WEIGHT_DIR/$WEIGHT_FILE | awk '{print $1}')"
    echo "如需重新下载，请先删除该文件"
    exit 0
fi

# 下载
echo "开始下载..."
echo "URL: $WEIGHT_URL"
echo "保存到: $WEIGHT_DIR/$WEIGHT_FILE"
echo ""

cd $WEIGHT_DIR
wget --progress=bar:force:noscroll $WEIGHT_URL

echo ""
echo "=========================================="
echo "下载完成！"
echo "文件: $WEIGHT_DIR/$WEIGHT_FILE"
echo "大小: $(du -h $WEIGHT_FILE | awk '{print $1}')"
echo "=========================================="

# 验证文件
if [ -f "$WEIGHT_FILE" ]; then
    file_size=$(stat -f%z "$WEIGHT_FILE" 2>/dev/null || stat -c%s "$WEIGHT_FILE" 2>/dev/null)
    if [ $file_size -lt 100000000 ]; then
        echo "警告: 文件大小异常，可能下载不完整"
        echo "请检查网络连接后重试"
        exit 1
    fi
fi

echo "✓ 权重文件就绪"
