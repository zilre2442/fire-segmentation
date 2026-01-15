#!/usr/bin/env python3
"""
按火点像素数量划分数据集

将数据集按火点像素数量分为四个子集：
- very_few: 1-10 像素
- few: 10-100 像素
- many: 100-1000 像素
- very_many: >1000 像素

生成的新CSV文件将保存在原split目录下的子目录中。

使用示例:
    # 划分 ActiveFire voting 数据集
    python data/split_by_fire_pixels.py --dataset activefire --algorithm voting
    
    # 划分 Land8Fire 数据集
    python data/split_by_fire_pixels.py --dataset land8fire
    
    # 划分所有数据集
    python data/split_by_fire_pixels.py --dataset all
"""

import os
import sys
import argparse
import random
from typing import Dict, List, Tuple
from tqdm import tqdm

import numpy as np
import pandas as pd
import rasterio

# 确保项目根目录在 sys.path 中
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def categorize_fire_pixels(count: int) -> str:
    """根据火点像素数分类"""
    if count == 0:
        return "none"
    elif count <= 10:
        return "very_few"
    elif count <= 100:
        return "few"
    elif count <= 1000:
        return "many"
    else:
        return "very_many"


def split_csv_by_fire_pixels(
    csv_path: str,
    output_dir: str,
    dataset_name: str
) -> Dict[str, int]:
    """
    读取CSV并按火点数量分类，生成四个子数据集CSV
    
    Args:
        csv_path: 输入CSV文件路径
        output_dir: 输出目录
        dataset_name: 数据集名称（用于文件命名）
    
    Returns:
        各类别的样本计数字典
    """
    if not os.path.exists(csv_path):
        print(f"⚠ 警告: CSV文件不存在: {csv_path}")
        return {}
    
    # 读取CSV文件
    df = pd.read_csv(csv_path)
    
    if df.empty:
        print(f"⚠ 警告: CSV文件为空: {csv_path}")
        return {}
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 按类别分组存储
    categories = {
        "very_few": [],
        "few": [],
        "many": [],
        "very_many": []
    }
    
    skipped = 0
    none_count = 0
    
    print(f"\n处理数据集: {dataset_name}")
    print(f"  CSV路径: {csv_path}")
    print(f"  总样本数: {len(df)}")
    
    # 遍历每一行
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="  分析进度"):
        # 获取mask路径
        if 'mask' in df.columns:
            mask_path = row['mask']
            image_path = row.get('image', row.iloc[0] if len(row) > 0 else None)
        elif len(df.columns) >= 2:
            image_path = row.iloc[0]
            mask_path = row.iloc[1]
        else:
            skipped += 1
            continue
        
        # 检查mask文件是否存在
        if not os.path.exists(mask_path):
            skipped += 1
            continue
        
        try:
            # 读取mask并统计火点像素数
            with rasterio.open(mask_path) as src:
                mask = src.read(1)
                fire_pixel_count = int(np.sum(mask > 0))
                
                # 分类
                category = categorize_fire_pixels(fire_pixel_count)
                
                if category == "none":
                    none_count += 1
                    continue  # 跳过无火点样本
                
                # 添加到对应类别
                if category in categories:
                    categories[category].append({
                        'image': image_path,
                        'mask': mask_path,
                        'fire_pixels': fire_pixel_count
                    })
        
        except Exception as e:
            skipped += 1
            continue
    
    # 保存各类别的CSV文件
    category_counts = {}
    for cat_name, samples in categories.items():
        if samples:
            # 创建DataFrame
            cat_df = pd.DataFrame(samples)
            
            # 保存CSV（保持与原始格式一致）
            output_csv = os.path.join(output_dir, f"{dataset_name}_{cat_name}.csv")
            cat_df[['image', 'mask']].to_csv(output_csv, index=False)
            
            category_counts[cat_name] = len(samples)
            print(f"  ✓ 已保存 {cat_name}: {len(samples)} 个样本 -> {output_csv}")
        else:
            category_counts[cat_name] = 0
            print(f"  ✗ {cat_name}: 无样本")
    
    if skipped > 0:
        print(f"  ⚠ 跳过样本数: {skipped}")
    if none_count > 0:
        print(f"  ℹ 无火点样本数: {none_count}")
    
    return category_counts


def process_activefire_dataset(
    algorithm: str,
    splits_dir: str = "data/splits_activefire"
) -> Dict[str, Dict[str, int]]:
    """处理 ActiveFire 数据集的 train/val/test 分割"""
    results = {}
    
    # 创建输出根目录
    output_root = os.path.join(splits_dir, f"by_fire_pixels_{algorithm}")
    
    for split in ["train", "val", "test"]:
        csv_path = os.path.join(splits_dir, f"{algorithm}_{split}.csv")
        dataset_name = f"{algorithm}_{split}"
        
        # 创建该split的输出目录
        split_output_dir = os.path.join(output_root, split)
        
        category_counts = split_csv_by_fire_pixels(
            csv_path=csv_path,
            output_dir=split_output_dir,
            dataset_name=dataset_name
        )
        
        if category_counts:
            results[dataset_name] = category_counts
    
    return results


def process_land8fire_dataset(
    splits_dir: str = "data/splits_land8fire"
) -> Dict[str, Dict[str, int]]:
    """处理 Land8Fire 数据集的 train/val/test 分割"""
    results = {}
    
    # 创建输出根目录
    output_root = os.path.join(splits_dir, "by_fire_pixels")
    
    for split in ["train", "val", "test"]:
        csv_path = os.path.join(splits_dir, f"Land8Fire_{split}.csv")
        dataset_name = f"Land8Fire_{split}"
        
        # 创建该split的输出目录
        split_output_dir = os.path.join(output_root, split)
        
        category_counts = split_csv_by_fire_pixels(
            csv_path=csv_path,
            output_dir=split_output_dir,
            dataset_name=dataset_name
        )
        
        if category_counts:
            results[dataset_name] = category_counts
    
    return results


def process_manual_dataset(
    splits_dir: str = "data/splits_manual"
) -> Dict[str, Dict[str, int]]:
    """处理 Manual Annotations 数据集的 train/val/test 分割"""
    results = {}
    
    # 创建输出根目录
    output_root = os.path.join(splits_dir, "by_fire_pixels")
    
    for split in ["train", "val", "test"]:
        csv_path = os.path.join(splits_dir, f"manual_{split}.csv")
        dataset_name = f"manual_{split}"
        
        # 创建该split的输出目录
        split_output_dir = os.path.join(output_root, split)
        
        category_counts = split_csv_by_fire_pixels(
            csv_path=csv_path,
            output_dir=split_output_dir,
            dataset_name=dataset_name
        )
        
        if category_counts:
            results[dataset_name] = category_counts
    
    return results


def merge_and_split_custom(
    activefire_dir: str,
    land8fire_dir: str,
    output_dir: str,
    activefire_algo: str = "voting"
):
    """
    合并 ActiveFire 和 Land8Fire 数据集，并按火点像素数量划分
    规则: <100 (small) vs >=100 (large)
    比例: 4:1:5
    """
    print(f"\n{'='*70}")
    print("执行合并与自定义划分 (ActiveFire + Land8Fire)")
    print(f"划分规则: <100 像素 (small) vs >=100 像素 (large)")
    print(f"划分比例: Train:Val:Test = 4:1:5")
    print(f"{'='*70}")

    os.makedirs(output_dir, exist_ok=True)
    
    all_samples = []
    
    # Helper to read CSV
    def read_samples(csv_path):
        samples = []
        if not os.path.exists(csv_path):
            return samples
        df = pd.read_csv(csv_path)
        for _, row in df.iterrows():
            if 'mask' in df.columns:
                mask = row['mask']
                image = row.get('image', row.iloc[0] if len(row) > 0 else None)
            elif len(df.columns) >= 2:
                image = row.iloc[0]
                mask = row.iloc[1]
            else:
                continue
            samples.append({'image': image, 'mask': mask})
        return samples

    # Load ActiveFire
    print("正在加载 ActiveFire 数据...")
    for split in ["train", "val", "test"]:
        path = os.path.join(activefire_dir, f"{activefire_algo}_{split}.csv")
        all_samples.extend(read_samples(path))
        
    # Load Land8Fire
    print("正在加载 Land8Fire 数据...")
    for split in ["train", "val", "test"]:
        path = os.path.join(land8fire_dir, f"Land8Fire_{split}.csv")
        all_samples.extend(read_samples(path))
        
    print(f"总样本数 (合并后): {len(all_samples)}")
    
    # Categorize
    small_samples = []
    large_samples = []
    skipped = 0
    
    print("正在分析火点像素数量...")
    for sample in tqdm(all_samples):
        mask_path = sample['mask']
        if not os.path.exists(mask_path):
            skipped += 1
            continue
            
        try:
            with rasterio.open(mask_path) as src:
                mask = src.read(1)
                count = int(np.sum(mask > 0))
                
                if count == 0:
                    continue # Skip no fire
                elif count < 100:
                    small_samples.append(sample)
                else:
                    large_samples.append(sample)
        except:
            skipped += 1
            
    print(f"分类结果:")
    print(f"  Small (<100): {len(small_samples)}")
    print(f"  Large (>=100): {len(large_samples)}")
    
    # Split and Save
    def save_split(samples, prefix):
        random.shuffle(samples)
        total = len(samples)
        n_train = int(total * 0.4)
        n_val = int(total * 0.1)
        # n_test = rest
        
        train_set = samples[:n_train]
        val_set = samples[n_train:n_train+n_val]
        test_set = samples[n_train+n_val:]
        
        # Save
        pd.DataFrame(train_set).to_csv(os.path.join(output_dir, f"{prefix}_train.csv"), index=False)
        pd.DataFrame(val_set).to_csv(os.path.join(output_dir, f"{prefix}_val.csv"), index=False)
        pd.DataFrame(test_set).to_csv(os.path.join(output_dir, f"{prefix}_test.csv"), index=False)
        
        print(f"  {prefix} -> Train: {len(train_set)}, Val: {len(val_set)}, Test: {len(test_set)}")

    save_split(small_samples, "small")
    save_split(large_samples, "large")
    
    print(f"\n结果已保存至: {output_dir}")
    print(f"使用方法示例:")
    print(f"  python exp/train_scripts/train_baseline.py --data-root {output_dir} --algo small")
    print(f"  python exp/train_scripts/train_baseline.py --data-root {output_dir} --algo large")
    
    return {} # Return empty dict to satisfy main loop structure if needed, or just exit


def print_summary(all_results: Dict[str, Dict[str, int]]):
    """打印汇总统计"""
    print(f"\n{'='*70}")
    print("按火点数量划分数据集 - 汇总统计")
    print(f"{'='*70}")
    print(f"{'数据集':<30} {'very_few':>10} {'few':>10} {'many':>10} {'very_many':>10}")
    print(f"{'-'*70}")
    
    for dataset_name, counts in all_results.items():
        print(f"{dataset_name:<30} {counts.get('very_few', 0):>10} "
              f"{counts.get('few', 0):>10} {counts.get('many', 0):>10} "
              f"{counts.get('very_many', 0):>10}")
    
    print(f"{'='*70}\n")


def main():
    parser = argparse.ArgumentParser(
        description="按火点像素数量将数据集划分为四个子集",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["activefire", "land8fire", "manual", "all", "merge_custom"],
        default="all",
        help="选择要处理的数据集。'merge_custom' 将合并 ActiveFire 和 Land8Fire 并按 <100 和 >=100 划分。"
    )
    
    parser.add_argument(
        "--algorithm",
        type=str,
        choices=["voting", "Kumar-Roy", "Murphy", "Schroeder", "intersection"],
        default="voting",
        help="ActiveFire数据集使用的算法（仅当--dataset=activefire时有效）"
    )
    
    parser.add_argument(
        "--activefire-dir",
        type=str,
        default="data/splits_activefire",
        help="ActiveFire数据集CSV所在目录"
    )
    
    parser.add_argument(
        "--land8fire-dir",
        type=str,
        default="data/splits_land8fire",
        help="Land8Fire数据集CSV所在目录"
    )

    parser.add_argument(
        "--manual-dir",
        type=str,
        default="data/splits_manual",
        help="Manual Annotations数据集CSV所在目录"
    )

    parser.add_argument(
        "--merge-output-dir",
        type=str,
        default="data/splits_merged_pixels",
        help="合并划分后的输出目录"
    )
    
    args = parser.parse_args()
    
    print("="*70)
    print("按火点像素数量划分数据集工具")
    print("="*70)
    print("\n类别定义:")
    print("  - very_few (1-10):    1-10 像素")
    print("  - few (10-100):       10-100 像素")
    print("  - many (100-1000):    100-1000 像素")
    print("  - very_many (>1000):  >1000 像素")
    print()
    
    all_results = {}
    
    # 处理数据集
    if args.dataset == "merge_custom":
        merge_and_split_custom(
            args.activefire_dir,
            args.land8fire_dir,
            args.merge_output_dir,
            args.algorithm
        )
        return

    if args.dataset in ["activefire", "all"]:
        print(f"\n{'#'*70}")
        print(f"# 处理 ActiveFire 数据集 (算法: {args.algorithm})")
        print(f"{'#'*70}")
        activefire_results = process_activefire_dataset(
            args.algorithm,
            args.activefire_dir
        )
        all_results.update(activefire_results)
    
    if args.dataset in ["land8fire", "all"]:
        print(f"\n{'#'*70}")
        print(f"# 处理 Land8Fire 数据集")
        print(f"{'#'*70}")
        land8fire_results = process_land8fire_dataset(args.land8fire_dir)
        all_results.update(land8fire_results)

    if args.dataset in ["manual", "all"]:
        print(f"\n{'#'*70}")
        print(f"# 处理 Manual Annotations 数据集")
        print(f"{'#'*70}")
        manual_results = process_manual_dataset(args.manual_dir)
        all_results.update(manual_results)
    
    # 打印汇总
    if all_results:
        print_summary(all_results)
        print(f"✓ 划分完成！共处理 {len(all_results)} 个数据集分割")
        print(f"✓ 新的CSV文件已保存在对应的 'by_fire_pixels_*' 子目录中\n")


if __name__ == "__main__":
    main()
