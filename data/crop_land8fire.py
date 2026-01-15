#!/usr/bin/env python3
"""
Land8Fire 数据集裁剪脚本

功能：
1. 将 Land8Fire/images_raw 中的 TIF 影像裁剪为 256x256 的小块
2. 将 Land8Fire/masks_raw 中的 PNG 掩膜裁剪为 256x256 的小块
3. 保存到 Land8Fire/images/patches 和 Land8Fire/masks/patches 目录
4. 只处理有对应掩膜的影像

使用方法:
    python data/crop_land8fire.py
"""

import os
import numpy as np
from PIL import Image
import rasterio
from rasterio.windows import Window
from tqdm import tqdm


# ============================================================================
# 配置参数
# ============================================================================

# 数据路径
DATASET_ROOT = "dataset/Land8Fire"
IMAGES_RAW_DIR = os.path.join(DATASET_ROOT, "images_raw")
MASKS_RAW_DIR = os.path.join(DATASET_ROOT, "masks_raw")
IMAGES_OUTPUT_DIR = os.path.join(DATASET_ROOT, "images", "patches")
MASKS_OUTPUT_DIR = os.path.join(DATASET_ROOT, "masks", "patches")

# 裁剪参数
PATCH_SIZE = 256
STRIDE = 256  # 步长，等于 PATCH_SIZE 表示无重叠
MIN_FIRE_PIXELS = 1  # 最小火点像素数，小于此值的 patch 将被丢弃（可选）


# ============================================================================
# 辅助函数
# ============================================================================

def get_base_name(filename):
    """
    从文件名中提取基础名称（去除扩展名和 _BQA 后缀）
    例如: LC08_L1TP_046021_20200908_20200908_01_RT.TIF -> LC08_L1TP_046021_20200908_20200908_01_RT
    """
    base = os.path.splitext(filename)[0]
    # 移除 _BQA 后缀（如果存在）
    if base.endswith("_BQA"):
        base = base[:-4]
    return base


def find_mask_for_image(image_name, masks_dict):
    """
    为影像查找对应的掩膜
    
    Args:
        image_name: 影像文件名
        masks_dict: 掩膜基础名称到文件名的映射
    
    Returns:
        对应的掩膜文件名，如果没有则返回 None
    """
    base = get_base_name(image_name)
    return masks_dict.get(base, None)


def crop_image_with_mask(image_path, mask_path, output_img_dir, output_mask_dir):
    """
    将影像和掩膜同时裁剪成小块
    
    Args:
        image_path: 影像文件路径
        mask_path: 掩膜文件路径
        output_img_dir: 影像输出目录
        output_mask_dir: 掩膜输出目录
    
    Returns:
        裁剪的 patch 数量
    """
    base_name = get_base_name(os.path.basename(image_path))
    
    # 读取影像
    with rasterio.open(image_path) as src:
        height, width = src.height, src.width
        n_bands = src.count
        
        # 读取掩膜
        mask = Image.open(mask_path)
        mask_arr = np.array(mask)
        
        # 确保掩膜尺寸与影像匹配
        if mask_arr.shape[:2] != (height, width):
            print(f"警告: {base_name} 的掩膜尺寸 {mask_arr.shape[:2]} 与影像尺寸 {(height, width)} 不匹配")
            # 尝试调整掩膜大小
            mask = mask.resize((width, height), Image.NEAREST)
            mask_arr = np.array(mask)
        
        patch_count = 0
        
        # 遍历影像进行裁剪
        for i in range(0, height - PATCH_SIZE + 1, STRIDE):
            for j in range(0, width - PATCH_SIZE + 1, STRIDE):
                # 裁剪掩膜块
                mask_patch = mask_arr[i:i+PATCH_SIZE, j:j+PATCH_SIZE]
                
                # 可选：跳过没有火点或火点过少的 patch
                fire_pixels = np.sum(mask_patch > 0)
                # if fire_pixels < MIN_FIRE_PIXELS:
                #     continue  # 取消注释此行以启用过滤
                
                # 裁剪影像块
                window = Window(j, i, PATCH_SIZE, PATCH_SIZE)
                image_patch = src.read(window=window)
                
                # 生成输出文件名
                patch_name = f"{base_name}_patch_{i:05d}_{j:05d}"
                
                # 保存影像块 (TIF 格式)
                image_patch_path = os.path.join(output_img_dir, f"{patch_name}.tif")
                with rasterio.open(
                    image_patch_path,
                    'w',
                    driver='GTiff',
                    height=PATCH_SIZE,
                    width=PATCH_SIZE,
                    count=n_bands,
                    dtype=image_patch.dtype,
                    compress='lzw'
                ) as dst:
                    dst.write(image_patch)
                
                # 保存掩膜块 (PNG 格式)
                mask_patch_path = os.path.join(output_mask_dir, f"{patch_name}.png")
                Image.fromarray(mask_patch).save(mask_patch_path)
                
                patch_count += 1
        
        return patch_count


def main():
    """主函数"""
    print("=" * 80)
    print("Land8Fire 数据集裁剪脚本")
    print("=" * 80)
    print(f"影像输入目录: {IMAGES_RAW_DIR}")
    print(f"掩膜输入目录: {MASKS_RAW_DIR}")
    print(f"影像输出目录: {IMAGES_OUTPUT_DIR}")
    print(f"掩膜输出目录: {MASKS_OUTPUT_DIR}")
    print(f"裁剪尺寸: {PATCH_SIZE}x{PATCH_SIZE}")
    print(f"步长: {STRIDE}")
    print()
    
    # 创建输出目录
    os.makedirs(IMAGES_OUTPUT_DIR, exist_ok=True)
    os.makedirs(MASKS_OUTPUT_DIR, exist_ok=True)
    
    # 获取所有影像文件（排除 BQA 和 .log 文件）
    image_files = [
        f for f in os.listdir(IMAGES_RAW_DIR)
        if f.endswith('.TIF') and '_BQA' not in f
    ]
    
    # 获取所有掩膜文件
    mask_files = [f for f in os.listdir(MASKS_RAW_DIR) if f.endswith('.png')]
    
    # 创建掩膜基础名称到文件名的映射
    masks_dict = {get_base_name(m): m for m in mask_files}
    
    print(f"找到 {len(image_files)} 个影像文件")
    print(f"找到 {len(mask_files)} 个掩膜文件")
    print()
    
    # 筛选有对应掩膜的影像
    valid_images = []
    for img in image_files:
        mask = find_mask_for_image(img, masks_dict)
        if mask:
            valid_images.append((img, mask))
        else:
            print(f"警告: {img} 没有对应的掩膜，跳过")
    
    print(f"共有 {len(valid_images)} 对有效的影像-掩膜配对")
    print()
    
    if len(valid_images) == 0:
        print("错误: 没有找到有效的影像-掩膜配对，请检查文件命名")
        return
    
    # 处理每对影像和掩膜
    total_patches = 0
    
    print("开始裁剪...")
    for img_file, mask_file in tqdm(valid_images, desc="处理进度"):
        img_path = os.path.join(IMAGES_RAW_DIR, img_file)
        mask_path = os.path.join(MASKS_RAW_DIR, mask_file)
        
        try:
            patch_count = crop_image_with_mask(
                img_path,
                mask_path,
                IMAGES_OUTPUT_DIR,
                MASKS_OUTPUT_DIR
            )
            total_patches += patch_count
        except Exception as e:
            print(f"\n错误: 处理 {img_file} 时发生异常: {e}")
            continue
    
    print()
    print("=" * 80)
    print("裁剪完成!")
    print("=" * 80)
    print(f"成功处理: {len(valid_images)} 对影像-掩膜")
    print(f"生成 patch 总数: {total_patches}")
    print(f"影像 patches: {IMAGES_OUTPUT_DIR}")
    print(f"掩膜 patches: {MASKS_OUTPUT_DIR}")
    print()


if __name__ == "__main__":
    main()
