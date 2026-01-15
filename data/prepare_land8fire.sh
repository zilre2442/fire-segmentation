#!/bin/bash
# Land8Fire 数据集预处理一键脚本
# 
# 使用方法:
#   chmod +x data/prepare_land8fire.sh
#   ./data/prepare_land8fire.sh

echo "=========================================="
echo "Land8Fire 数据集预处理"
echo "=========================================="
echo ""

# 检查 Python 环境
if ! command -v python3 &> /dev/null; then
    echo "错误: 未找到 python3，请先安装 Python"
    exit 1
fi

echo "[步骤 1/2] 裁剪影像和掩膜为 256x256 patches..."
echo "--------------------------------------"
python3 data/crop_land8fire.py

if [ $? -ne 0 ]; then
    echo ""
    echo "错误: 裁剪脚本执行失败"
    exit 1
fi

echo ""
echo "[步骤 2/2] 划分训练集、验证集和测试集..."
echo "--------------------------------------"
python3 data/split_land8fire.py

if [ $? -ne 0 ]; then
    echo ""
    echo "错误: 划分脚本执行失败"
    exit 1
fi

echo ""
echo "=========================================="
echo "✓ Land8Fire 数据集预处理完成!"
echo "=========================================="
echo ""
echo "生成的文件:"
echo "  - dataset/Land8Fire/images/patches/"
echo "  - dataset/Land8Fire/masks/patches/"
echo "  - data/splits_land8fire/Land8Fire_train.csv"
echo "  - data/splits_land8fire/Land8Fire_val.csv"
echo "  - data/splits_land8fire/Land8Fire_test.csv"
echo ""
echo "可以使用以下配置训练模型:"
echo "  DATA_ROOT = 'data/splits_land8fire'"
echo "  ALGORITHM = 'Land8Fire'"
echo ""
