#!/usr/bin/env python3
"""
火点像素数量分布分析工具

分析各数据集中样本的火点像素数量分布，按以下类别统计：
- 类别1: 1-10 像素
- 类别2: 10-100 像素  
- 类别3: 100-1000 像素
- 类别4: >1000 像素
- 类别0: 无火点 (0像素)

支持的数据集:
- ActiveFire (voting, Kumar-Roy, Murphy, Schroeder, intersection)
- Land8Fire

使用示例:
    python data/analyze_fire_pixel_distribution.py
    python data/analyze_fire_pixel_distribution.py --dataset activefire --algorithm voting
    python data/analyze_fire_pixel_distribution.py --dataset manual
    python data/analyze_fire_pixel_distribution.py --dataset all --save-csv data/fire_pixel_distribution.csv
"""

import os
import sys
import argparse
from collections import defaultdict
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
        return "none (0)"
    elif count <= 10:
        return "very_few (1-10)"
    elif count <= 100:
        return "few (10-100)"
    elif count <= 1000:
        return "many (100-1000)"
    else:
        return "very_many (>1000)"


def analyze_csv_dataset(csv_path: str, dataset_name: str) -> Dict[str, int]:
    """
    分析单个CSV文件对应的数据集
    
    Args:
        csv_path: CSV文件路径
        dataset_name: 数据集名称（用于显示）
    
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
    
    # 统计各类别数量
    category_counts = defaultdict(int)
    skipped = 0
    
    print(f"\n分析数据集: {dataset_name}")
    print(f"  CSV路径: {csv_path}")
    print(f"  总样本数: {len(df)}")
    
    # 遍历每一行
    for idx, row in tqdm(df.iterrows(), total=len(df), desc=f"  处理进度"):
        # CSV格式: image_path, mask_path
        # 根据实际CSV结构调整列名
        if 'mask' in df.columns:
            mask_path = row['mask']
        elif len(df.columns) >= 2:
            mask_path = row.iloc[1]  # 假设第二列是mask路径
        else:
            print(f"  ⚠ 无法识别mask列，跳过该样本")
            skipped += 1
            continue
        
        # 检查mask文件是否存在
        if not os.path.exists(mask_path):
            skipped += 1
            continue
        
        try:
            # 读取mask并统计火点像素数
            with rasterio.open(mask_path) as src:
                mask = src.read(1)  # 读取第一个波段
                fire_pixel_count = int(np.sum(mask > 0))
                
                # 分类
                category = categorize_fire_pixels(fire_pixel_count)
                category_counts[category] += 1
        
        except Exception as e:
            skipped += 1
            continue
    
    if skipped > 0:
        print(f"  ⚠ 跳过样本数: {skipped}")
    
    return dict(category_counts)


def print_category_stats(category_counts: Dict[str, int], dataset_name: str):
    """打印类别统计信息"""
    total = sum(category_counts.values())
    
    print(f"\n{'='*60}")
    print(f"数据集: {dataset_name}")
    print(f"{'='*60}")
    print(f"{'类别':<25} {'样本数':>10} {'占比':>10}")
    print(f"{'-'*60}")
    
    # 按预定义顺序显示
    order = [
        "none (0)",
        "very_few (1-10)", 
        "few (10-100)",
        "many (100-1000)",
        "very_many (>1000)"
    ]
    
    for cat in order:
        count = category_counts.get(cat, 0)
        percentage = (count / total * 100) if total > 0 else 0
        print(f"{cat:<25} {count:>10} {percentage:>9.2f}%")
    
    print(f"{'-'*60}")
    print(f"{'总计':<25} {total:>10} {100.0:>9.2f}%")
    print(f"{'='*60}\n")


def analyze_activefire_dataset(algorithm: str, splits_dir: str = "data/splits_activefire"):
    """分析ActiveFire数据集的train/val/test分割"""
    results = {}
    
    for split in ["train", "val", "test"]:
        csv_path = os.path.join(splits_dir, f"{algorithm}_{split}.csv")
        dataset_name = f"ActiveFire-{algorithm}-{split}"
        
        category_counts = analyze_csv_dataset(csv_path, dataset_name)
        if category_counts:
            results[dataset_name] = category_counts
            print_category_stats(category_counts, dataset_name)
    
    return results


def analyze_land8fire_dataset(splits_dir: str = "data/splits_land8fire"):
    """分析Land8Fire数据集的train/val/test分割"""
    results = {}
    
    for split in ["train", "val", "test"]:
        csv_path = os.path.join(splits_dir, f"Land8Fire_{split}.csv")
        dataset_name = f"Land8Fire-{split}"
        
        category_counts = analyze_csv_dataset(csv_path, dataset_name)
        if category_counts:
            results[dataset_name] = category_counts
            print_category_stats(category_counts, dataset_name)
    
    return results


def analyze_manual_dataset(splits_dir: str = "data/splits_manual"):
    """分析 Manual Annotations 数据集的 train/val/test 分割"""
    results = {}
    
    for split in ["train", "val", "test"]:
        csv_path = os.path.join(splits_dir, f"manual_{split}.csv")
        dataset_name = f"Manual-{split}"
        
        category_counts = analyze_csv_dataset(csv_path, dataset_name)
        if category_counts:
            results[dataset_name] = category_counts
            print_category_stats(category_counts, dataset_name)
    
    return results


def save_results_to_csv(all_results: Dict[str, Dict[str, int]], output_path: str):
    """将统计结果保存为CSV文件"""
    # 准备数据
    rows = []
    categories = ["none (0)", "very_few (1-10)", "few (10-100)", "many (100-1000)", "very_many (>1000)"]
    
    for dataset_name, category_counts in all_results.items():
        row = {"dataset": dataset_name}
        total = sum(category_counts.values())
        
        for cat in categories:
            count = category_counts.get(cat, 0)
            percentage = (count / total * 100) if total > 0 else 0
            row[f"{cat}_count"] = count
            row[f"{cat}_percent"] = f"{percentage:.2f}%"
        
        row["total"] = total
        rows.append(row)
    
    # 创建DataFrame并保存
    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    print(f"\n✓ 统计结果已保存至: {output_path}\n")


def main():
    parser = argparse.ArgumentParser(
        description="分析数据集中火点像素数量的分布",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["activefire", "land8fire", "manual", "all"],
        default="all",
        help="选择要分析的数据集"
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
        "--save-csv",
        type=str,
        default="data/fire_pixel_distribution.csv",
        help="保存统计结果的CSV文件路径（默认：当前目录下的fire_pixel_distribution.csv）"
    )
    
    args = parser.parse_args()
    
    print("="*60)
    print("火点像素数量分布分析工具")
    print("="*60)
    print("\n类别定义:")
    print("  - none (0):           无火点")
    print("  - very_few (1-10):    1-10 像素")
    print("  - few (10-100):       10-100 像素")
    print("  - many (100-1000):    100-1000 像素")
    print("  - very_many (>1000):  >1000 像素")
    print()
    
    all_results = {}
    
    # 分析数据集
    if args.dataset in ["activefire", "all"]:
        print(f"\n{'#'*60}")
        print(f"# 分析 ActiveFire 数据集 (算法: {args.algorithm})")
        print(f"{'#'*60}")
        activefire_results = analyze_activefire_dataset(args.algorithm, args.activefire_dir)
        all_results.update(activefire_results)
    
    if args.dataset in ["land8fire", "all"]:
        print(f"\n{'#'*60}")
        print(f"# 分析 Land8Fire 数据集")
        print(f"{'#'*60}")
        land8fire_results = analyze_land8fire_dataset(args.land8fire_dir)
        all_results.update(land8fire_results)

    if args.dataset in ["manual", "all"]:
        print(f"\n{'#'*60}")
        print(f"# 分析 Manual Annotations 数据集")
        print(f"{'#'*60}")
        manual_results = analyze_manual_dataset(args.manual_dir)
        all_results.update(manual_results)
    
    # 保存结果到CSV
    if all_results:
        # 如果未指定保存路径，使用默认路径（当前目录）
        if args.save_csv is None:
            # 根据分析的数据集生成文件名
            if args.dataset == "activefire":
                csv_filename = f"fire_pixel_distribution_activefire_{args.algorithm}.csv"
            elif args.dataset == "land8fire":
                csv_filename = "fire_pixel_distribution_land8fire.csv"
            elif args.dataset == "manual":
                csv_filename = "fire_pixel_distribution_manual.csv"
            else:  # all
                csv_filename = "fire_pixel_distribution_all.csv"
            output_path = csv_filename
        else:
            output_path = args.save_csv
        
        save_results_to_csv(all_results, output_path)
    
    # 汇总统计
    if all_results:
        print(f"\n{'='*60}")
        print(f"分析完成！共处理 {len(all_results)} 个数据集分割")
        print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
