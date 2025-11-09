#!/usr/bin/env python3
"""
Land8Fire 数据集划分脚本

功能：
1. 读取 Land8Fire/images/patches 和 Land8Fire/masks/patches 中的 patch
2. 将数据集划分为训练集、验证集和测试集
3. 生成 CSV 文件保存到 data/splits_land8fire 目录

划分比例：
- 训练集: 70%
- 验证集: 15%
- 测试集: 15%

使用方法:
    python data/split_land8fire.py
"""

import os
import random
import csv
from tqdm import tqdm


# ============================================================================
# 配置参数
# ============================================================================

# 数据路径
DATASET_ROOT = "dataset/Land8Fire"
IMAGE_DIR = os.path.join(DATASET_ROOT, "images", "patches")
MASK_DIR = os.path.join(DATASET_ROOT, "masks", "patches")

# 输出目录
OUTPUT_DIR = "data/splits_land8fire"

# 划分比例
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# 随机种子（保证可重复性）
RANDOM_SEED = 42


# ============================================================================
# 辅助函数
# ============================================================================

def get_base_name(filename):
    """
    从文件名中提取基础名称（去除扩展名）
    """
    return os.path.splitext(filename)[0]


def verify_data_pairing(image_files, mask_files):
    """
    验证影像和掩膜文件是否一一对应
    
    Args:
        image_files: 影像文件列表
        mask_files: 掩膜文件列表
    
    Returns:
        配对的 (影像文件, 掩膜文件) 列表
    """
    # 创建掩膜文件名集合（去除扩展名）
    mask_basenames = {get_base_name(m) for m in mask_files}
    
    # 验证配对
    paired_data = []
    unmatched_images = []
    
    for img in image_files:
        img_base = get_base_name(img)
        
        # 查找对应的掩膜（.png 扩展名）
        if img_base in mask_basenames:
            paired_data.append((img, f"{img_base}.png"))
        else:
            unmatched_images.append(img)
    
    if unmatched_images:
        print(f"警告: {len(unmatched_images)} 个影像没有对应的掩膜")
        # 可选：打印前几个不匹配的文件
        # for img in unmatched_images[:5]:
        #     print(f"  - {img}")
    
    return paired_data


def split_dataset(paired_data, train_ratio, val_ratio, test_ratio, seed):
    """
    将数据集划分为训练集、验证集和测试集
    
    Args:
        paired_data: 配对的数据列表
        train_ratio: 训练集比例
        val_ratio: 验证集比例
        test_ratio: 测试集比例
        seed: 随机种子
    
    Returns:
        (train_set, val_set, test_set) 元组
    """
    # 设置随机种子
    random.seed(seed)
    
    # 打乱数据
    data_copy = paired_data.copy()
    random.shuffle(data_copy)
    
    # 计算划分点
    total = len(data_copy)
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)
    
    # 划分数据集
    train_set = data_copy[:train_end]
    val_set = data_copy[train_end:val_end]
    test_set = data_copy[val_end:]
    
    return train_set, val_set, test_set


def save_to_csv(data, image_dir, mask_dir, output_path):
    """
    将数据集保存为 CSV 文件
    
    Args:
        data: 数据列表 [(image_file, mask_file), ...]
        image_dir: 影像目录路径
        mask_dir: 掩膜目录路径
        output_path: 输出 CSV 文件路径
    """
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['image_path', 'mask_path'])  # 表头
        
        for img_file, mask_file in data:
            img_path = os.path.join(image_dir, img_file)
            mask_path = os.path.join(mask_dir, mask_file)
            writer.writerow([img_path, mask_path])


def main():
    """主函数"""
    print("=" * 80)
    print("Land8Fire 数据集划分脚本")
    print("=" * 80)
    print(f"影像目录: {IMAGE_DIR}")
    print(f"掩膜目录: {MASK_DIR}")
    print(f"输出目录: {OUTPUT_DIR}")
    print(f"划分比例 - 训练集: {TRAIN_RATIO*100:.0f}%, 验证集: {VAL_RATIO*100:.0f}%, 测试集: {TEST_RATIO*100:.0f}%")
    print(f"随机种子: {RANDOM_SEED}")
    print()
    
    # 检查目录是否存在
    if not os.path.exists(IMAGE_DIR):
        print(f"错误: 影像目录不存在: {IMAGE_DIR}")
        print("请先运行 crop_land8fire.py 脚本生成 patches")
        return
    
    if not os.path.exists(MASK_DIR):
        print(f"错误: 掩膜目录不存在: {MASK_DIR}")
        print("请先运行 crop_land8fire.py 脚本生成 patches")
        return
    
    # 创建输出目录
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 获取所有文件
    print("读取文件列表...")
    image_files = [f for f in os.listdir(IMAGE_DIR) if f.endswith('.tif')]
    mask_files = [f for f in os.listdir(MASK_DIR) if f.endswith('.png')]
    
    print(f"找到 {len(image_files)} 个影像 patches")
    print(f"找到 {len(mask_files)} 个掩膜 patches")
    print()
    
    # 验证数据配对
    print("验证影像-掩膜配对...")
    paired_data = verify_data_pairing(image_files, mask_files)
    
    if len(paired_data) == 0:
        print("错误: 没有找到有效的影像-掩膜配对，请检查文件")
        return
    
    print(f"成功配对: {len(paired_data)} 对")
    print()
    
    # 划分数据集
    print("划分数据集...")
    train_set, val_set, test_set = split_dataset(
        paired_data,
        TRAIN_RATIO,
        VAL_RATIO,
        TEST_RATIO,
        RANDOM_SEED
    )
    
    print(f"训练集: {len(train_set)} 对 ({len(train_set)/len(paired_data)*100:.1f}%)")
    print(f"验证集: {len(val_set)} 对 ({len(val_set)/len(paired_data)*100:.1f}%)")
    print(f"测试集: {len(test_set)} 对 ({len(test_set)/len(paired_data)*100:.1f}%)")
    print()
    
    # 保存 CSV 文件
    print("保存 CSV 文件...")
    
    train_csv = os.path.join(OUTPUT_DIR, "Land8Fire_train.csv")
    val_csv = os.path.join(OUTPUT_DIR, "Land8Fire_val.csv")
    test_csv = os.path.join(OUTPUT_DIR, "Land8Fire_test.csv")
    
    save_to_csv(train_set, IMAGE_DIR, MASK_DIR, train_csv)
    save_to_csv(val_set, IMAGE_DIR, MASK_DIR, val_csv)
    save_to_csv(test_set, IMAGE_DIR, MASK_DIR, test_csv)
    
    print(f"训练集 CSV: {train_csv}")
    print(f"验证集 CSV: {val_csv}")
    print(f"测试集 CSV: {test_csv}")
    print()
    
    # 生成统计信息
    print("=" * 80)
    print("数据集划分完成!")
    print("=" * 80)
    print(f"总样本数: {len(paired_data)}")
    print(f"  训练集: {len(train_set)} ({len(train_set)/len(paired_data)*100:.1f}%)")
    print(f"  验证集: {len(val_set)} ({len(val_set)/len(paired_data)*100:.1f}%)")
    print(f"  测试集: {len(test_set)} ({len(test_set)/len(paired_data)*100:.1f}%)")
    print()
    print("可以使用以下配置训练模型:")
    print(f"  DATA_ROOT = '{OUTPUT_DIR}'")
    print(f"  ALGORITHM = 'Land8Fire'")
    print()


if __name__ == "__main__":
    main()
