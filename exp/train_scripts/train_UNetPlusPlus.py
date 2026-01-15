"""训练脚本: UNet++ (对齐 Dual-Sight Fire 工作流)

- 支持单机/分布式训练 (`--dist` 或 torchrun 环境变量自动检测)
- 记录 `train_log.txt`、`hyperparameters.txt` 以及 `loss_curves.png`

示例：
CUDA_VISIBLE_DEVICES=2,7 torchrun --nproc_per_node=2 --master_port=65530 exp/train_scripts/train_UNetPlusPlus.py --data-root data/splits_merged_pixels --algo large
CUDA_VISIBLE_DEVICES=7 python exp/train_scripts/train_UNetPlusPlus.py --data-root data/splits_merged_pixels --algo small
python exp/train_scripts/train_UNetPlusPlus.py --data-root data/splits_merged_pixels --algo small
python exp/train_scripts/train_UNetPlusPlus.py --data-root data/splits_merged_pixels --algo large
"""

from __future__ import annotations

import os
import sys
import time
import argparse
from typing import Tuple, Optional

import random
import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Ensure project root is on sys.path so top-level modules (dataset, loss, utils, models) can be imported
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from dataset import LandsatFireDataset
from models.unet_plusplus import UNetPlusPlus
from loss import FocalLoss, SpatialFocalLoss


def parse_args():
    p = argparse.ArgumentParser(description="Train UNet++ (aligned with Dual-Sight Fire script)")
    p.add_argument("--data-root", type=str, default="data/splits_activefire")
    p.add_argument("--algo", type=str, default="voting")
    p.add_argument("--bands", type=int, nargs=3, default=(7, 6, 2))
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--save-dir", type=str, default=None)
    p.add_argument("--dist", action="store_true", help="启用分布式 (torchrun 设置 env 后自动检测)")
    p.add_argument("--backend", type=str, default="nccl", choices=["nccl", "gloo"], help="分布式后端")
    return p.parse_args()


def get_csv_paths(root: str, algo: str) -> Tuple[str, str]:
    train_csv = os.path.join(root, f"{algo}_train.csv")
    val_csv = os.path.join(root, f"{algo}_val.csv")
    return train_csv, val_csv


def is_dist_requested(args) -> bool:
    return args.dist or int(os.environ.get("WORLD_SIZE", "1")) > 1


def setup_ddp(args) -> Tuple[bool, int, int, Optional[int]]:
    dist_flag = is_dist_requested(args)
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if not dist_flag or world_size <= 1:
        return False, 0, 1, None
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    if not dist.is_initialized():
        dist.init_process_group(backend=args.backend, init_method="env://")
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    return True, rank, world_size, local_rank


def cleanup_ddp():
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


def main():
    args = parse_args()
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

    dist_enabled, rank, world_size, local_rank = setup_ddp(args)
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu") if dist_enabled else torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_csv, val_csv = get_csv_paths(args.data_root, args.algo)
    save_dir = args.save_dir or f"output/UNetPlusPlus/{args.algo}_{time.strftime('%Y%m%d%H%M')}"
    if (not dist_enabled) or rank == 0:
        os.makedirs(save_dir, exist_ok=True)
    weights_dir = os.path.join(save_dir, "weights")
    if (not dist_enabled) or rank == 0:
        os.makedirs(weights_dir, exist_ok=True)

    train_ds = LandsatFireDataset(train_csv, bands=tuple(args.bands))
    val_ds = LandsatFireDataset(val_csv, bands=tuple(args.bands))

    if dist_enabled:
        train_sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=rank, shuffle=True, drop_last=True)
        val_sampler = DistributedSampler(val_ds, num_replicas=world_size, rank=rank, shuffle=False, drop_last=False)
    else:
        train_sampler = None
        val_sampler = None

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=dist_enabled,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        sampler=val_sampler,
        num_workers=max(1, args.num_workers // 2),
        pin_memory=True,
    )

    model = UNetPlusPlus(n_channels=len(args.bands), n_classes=1).to(device)
    if dist_enabled:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[local_rank] if torch.cuda.is_available() else None,
            output_device=local_rank if torch.cuda.is_available() else None,
            find_unused_parameters=False,
        )

    # 使用 BCEWithLogitsLoss 直接对 logits 计算，数值更稳定
    criterion = nn.BCEWithLogitsLoss()
    # criterion = FocalLoss(alpha=0.75, gamma=1.5, reduction="mean")
    # criterion = SpatialFocalLoss(
    #     gamma_0=2.0,
    #     alpha_bg=0.5,
    #     connectivity=8,
    # )
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.8, patience=5)

    best_val = float("inf")
    log_path = os.path.join(save_dir, "train_log.txt")
    hparam_path = os.path.join(save_dir, "hyperparameters.txt")
    if (not dist_enabled) or rank == 0:
        with open(log_path, "w") as log:
            log.write("epoch,train_loss,train_acc,val_loss,val_acc\n")
        with open(hparam_path, "w") as f:
            f.write("===== UNet++ Hyperparameters =====\n")
            f.write(f"time: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"algo: {args.algo}\n")
            f.write(f"data_root: {args.data_root}\n")
            f.write(f"bands: {args.bands}\n")
            f.write(f"epochs: {args.epochs}\n")
            f.write(f"batch_size: {args.batch_size}\n")
            f.write(f"lr: {args.lr}\n")
            f.write(f"dist_enabled: {dist_enabled} world_size: {world_size}\n")
            f.write("\nModel Info:\n")
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            f.write(f"total_params: {total_params}\n")
            f.write(f"trainable_params: {trainable_params}\n")
            f.write("\nLoss\n")
            f.write(f"  name: {type(criterion).__name__}\n")
            if hasattr(criterion, 'reduction'):
                f.write(f"  reduction: {criterion.reduction}\n")
            f.write("=======================================\n")

    def reduce_scalar(v: float) -> float:
        if not dist_enabled:
            return v
        t = torch.tensor([v], dtype=torch.float, device=device)
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
        return (t / world_size).item()

    train_curve, val_curve = [], []

    for epoch in range(args.epochs):
        if dist_enabled:
            train_sampler.set_epoch(epoch)
        model.train()
        train_sum = 0.0
        train_acc_sum = 0.0
        iterator = train_loader
        if (not dist_enabled) or rank == 0:
            iterator = tqdm(train_loader, desc=f"Train {epoch + 1}/{args.epochs}", leave=False)
        for images, masks in iterator:
            images = images.to(device, non_blocking=True)
            masks = (masks > 0).float().to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            preds = model(images)
            loss = criterion(preds, masks)
            loss.backward()
            optimizer.step()
            train_sum += loss.item() * images.size(0)
            
            # Accuracy
            preds_bin = (torch.sigmoid(preds) > 0.5).float()
            acc = (preds_bin == masks).float().mean()
            train_acc_sum += acc.item() * images.size(0)

            if ((not dist_enabled) or rank == 0) and isinstance(iterator, tqdm):
                iterator.set_postfix({"loss": f"{loss.item():.3f}"})

        avg_train_local = train_sum / len(train_ds if not dist_enabled else train_sampler.dataset)
        avg_train_acc_local = train_acc_sum / len(train_ds if not dist_enabled else train_sampler.dataset)
        avg_train = reduce_scalar(avg_train_local)
        avg_train_acc = reduce_scalar(avg_train_acc_local)

        model.eval()
        val_sum = 0.0
        val_acc_sum = 0.0
        with torch.inference_mode():
            v_iterator = val_loader
            if (not dist_enabled) or rank == 0:
                v_iterator = tqdm(val_loader, desc=f"Val   {epoch + 1}/{args.epochs}", leave=False)
            for images, masks in v_iterator:
                images = images.to(device, non_blocking=True)
                masks = (masks > 0).float().to(device, non_blocking=True)
                preds = model(images)
                val_loss = criterion(preds, masks)
                val_sum += val_loss.item() * images.size(0)
                
                # Accuracy
                preds_bin = (torch.sigmoid(preds) > 0.5).float()
                acc = (preds_bin == masks).float().mean()
                val_acc_sum += acc.item() * images.size(0)

                if ((not dist_enabled) or rank == 0) and isinstance(v_iterator, tqdm):
                    v_iterator.set_postfix({"loss": f"{val_loss.item():.3f}"})

        avg_val_local = val_sum / len(val_ds if not dist_enabled else val_sampler.dataset)
        avg_val_acc_local = val_acc_sum / len(val_ds if not dist_enabled else val_sampler.dataset)
        avg_val = reduce_scalar(avg_val_local)
        avg_val_acc = reduce_scalar(avg_val_acc_local)

        if (not dist_enabled) or rank == 0:
            print(f"Epoch {epoch + 1:03d} | train_loss {avg_train:.4f} | train_acc {avg_train_acc:.4f} | val_loss {avg_val:.4f} | val_acc {avg_val_acc:.4f}")
            with open(log_path, "a") as log:
                log.write(f"{epoch + 1},{avg_train:.6f},{avg_train_acc:.6f},{avg_val:.6f},{avg_val_acc:.6f}\n")
            
            if avg_val < best_val:
                # best_val updated below for all ranks
                torch.save((model.module if dist_enabled else model).state_dict(), os.path.join(weights_dir, "model_best.pth"))

            if (epoch + 1) % 5 == 0:
                torch.save((model.module if dist_enabled else model).state_dict(), os.path.join(weights_dir, f"model_epoch_{epoch + 1}.pth"))

        # Update best_val on all ranks
        if avg_val < best_val:
            best_val = avg_val

        train_curve.append(avg_train)
        val_curve.append(avg_val)


    if (not dist_enabled) or rank == 0:
        torch.save((model.module if dist_enabled else model).state_dict(), os.path.join(weights_dir, "model_final.pth"))
        plt.figure(figsize=(8, 5))
        plt.plot(train_curve, label="train")
        plt.plot(val_curve, label="val")
        plt.xlabel("epoch"); plt.ylabel("loss"); plt.title("UNet++ Loss")
        plt.legend(); plt.grid(True)
        plt.savefig(os.path.join(save_dir, "loss_curves.png"), dpi=150, bbox_inches="tight")
        plt.close()
        print("训练结束。最优验证损失:", best_val)

    cleanup_ddp()


if __name__ == "__main__":
    main()
