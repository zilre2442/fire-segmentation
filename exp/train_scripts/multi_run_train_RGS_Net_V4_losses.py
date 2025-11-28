#!/usr/bin/env python3
"""RGS-Net V4 多次训练脚本 (分割损失对比)。

按顺序运行五次训练, seg_loss_fn 依次取 FocalTverskyLoss、FocalLoss、TverskyLoss、BCELoss、DiceLoss。

使用方式示例:
    CUDA_VISIBLE_DEVICES=1,4,5,6,7 \
    python exp/train_scripts/multi_run_train_RGS_Net_V4_losses.py \
        --nproc-per-node 5 --master-port 65530 \
        --base-save-root output/RGS_Net_V4/loss_ablation \
        -- --data-root data/splits_activefire --algo voting --bands 7 6 5 \
           --epochs 80 --batch-size 16 --lr 3e-4 --base-filters 64 --num-workers 4 --dist
"""

from __future__ import annotations

import argparse
import os
import subprocess
from datetime import datetime
from typing import List

LOSS_RUNS = [
    ("focal_tversky", "FocalTverskyLoss"),
    ("focal", "FocalLoss"),
    ("tversky", "TverskyLoss"),
    ("bce", "BCEWithLogitsLoss"),
    ("dice", "DiceLoss"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Multi-run training for RGS-Net V4 seg loss ablation")
    parser.add_argument("--train-script", type=str, default="train_RGS_Net_V4.py",
                        help="入口训练脚本路径")
    parser.add_argument("--nproc-per-node", type=int, default=1, help="torchrun 进程数")
    parser.add_argument("--master-port", type=int, default=29500, help="torchrun 端口")
    parser.add_argument("--base-save-root", type=str, default="output/RGS_Net_V4/loss_ablation",
                        help="各实验结果保存根目录")
    parser.add_argument("--losses", nargs="*", choices=[name for name, _ in LOSS_RUNS],
                        default=[name for name, _ in LOSS_RUNS], help="需要运行的损失列表")
    parser.add_argument("--dry-run", action="store_true", help="仅打印命令不执行")
    parser.add_argument("train_args", nargs=argparse.REMAINDER,
                        help="其余参数透传给训练脚本 (请使用 -- 分隔)")
    return parser.parse_args()


def sanitize_extra_args(extra: List[str]) -> List[str]:
    if not extra:
        return []
    if extra and extra[0] == "--":
        return extra[1:]
    return extra


def run_experiment(cmd: List[str], env) -> int:
    print("=" * 80)
    print("运行命令:", " ".join(cmd))
    print("=" * 80)
    if env:
        for key in ("CUDA_VISIBLE_DEVICES",):
            if key in env:
                print(f"env {key}={env[key]}")
    if env is None:
        env = os.environ.copy()
    result = subprocess.run(cmd, env=env)
    return result.returncode


def main() -> None:
    args = parse_args()
    extra_args = sanitize_extra_args(args.train_args)
    os.makedirs(args.base_save_root, exist_ok=True)
    env = os.environ.copy()
    summary = []

    for loss_flag, label in LOSS_RUNS:
        if args.losses and loss_flag not in args.losses:
            continue
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        save_dir = os.path.abspath(os.path.join(args.base_save_root, f"{loss_flag}_{timestamp}"))
        base_cmd = [
            "torchrun",
            f"--nproc_per_node={args.nproc_per_node}",
            f"--master_port={args.master_port}",
            args.train_script,
            "--seg-loss", loss_flag,
            "--save-dir", save_dir,
        ] + extra_args
        print(f"[multi-run] 准备执行 {label} -> 输出 {save_dir}")
        if args.dry_run:
            print("DRY-RUN:", " ".join(base_cmd))
            summary.append((loss_flag, save_dir, 0))
            continue
        return_code = run_experiment(base_cmd, env)
        summary.append((loss_flag, save_dir, return_code))
        if return_code != 0:
            print(f"[multi-run] 实验 {loss_flag} 失败, 终止后续任务")
            break

    print("\n===== 多次训练结果汇总 =====")
    for loss_flag, save_dir, code in summary:
        status = "成功" if code == 0 else f"失败({code})"
        print(f"{loss_flag:>14}: {status} -> {save_dir}")
    print("================================")


if __name__ == "__main__":
    main()
