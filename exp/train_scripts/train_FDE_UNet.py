"""
训练脚本: FDE-UNet (基于 HaarWavelet + ACmix + CBAM)

- 与 baseline/RGS V4 脚本共享 CLI 结构，便于统一调度
- 默认使用 BCEWithLogitsLoss；模型返回 logits
- 记录 train_log.txt / hyperparameters.txt / loss_curves.png 与权重
- 训练参数与论文保持一致: Epochs=100, BatchSize=16, Optimizer=Adam, LR=1e-3, EarlyStopping=5

示例：
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 --master_port=65531 exp/train_scripts/train_FDE_UNet.py --data-root data/splits_merged_pixels --algo small
CUDA_VISIBLE_DEVICES=1 python exp/train_scripts/train_FDE_UNet.py --data-root data/splits_merged_pixels --algo large
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
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from dataset import LandsatFireDataset
from models.FDE_Net import FDE_UNet
from loss import FocalLoss

def parse_args():
    p = argparse.ArgumentParser(description="Train FDE-UNet (single/multi GPU)")
    p.add_argument("--data-root", type=str, default="data/splits_activefire")
    p.add_argument("--algo", type=str, default="voting")
    p.add_argument("--bands", type=int, nargs='+', default=(1, 2, 3, 4, 5, 6, 7, 8, 9, 10))
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--save-dir", type=str, default=None)
    p.add_argument("--dist", action="store_true", help="启用分布式 (torchrun 环境变量自动检测)")
    p.add_argument("--backend", type=str, default="nccl", choices=["nccl", "gloo"], help="分布式后端")
    p.add_argument("--find-unused", action="store_true", help="若模型含条件分支，启用以避免 DDP 未使用参数报错")
    return p.parse_args()


def get_csv_paths(root: str, algo: str) -> Tuple[str, str]:
    train_csv = os.path.join(root, f"{algo}_train.csv")
    val_csv = os.path.join(root, f"{algo}_val.csv")
    return train_csv, val_csv


def is_dist_enabled(args) -> bool:
    return args.dist or int(os.environ.get("WORLD_SIZE", "1")) > 1


def setup_ddp(args) -> Tuple[bool, int, int, Optional[int]]:
    dist_flag = is_dist_enabled(args)
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
    torch.backends.cudnn.benchmark = True

    dist_enabled, rank, world_size, local_rank = setup_ddp(args)
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu") if dist_enabled else torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_csv, val_csv = get_csv_paths(args.data_root, args.algo)
    save_dir = args.save_dir or f"output/FDE_UNet/{args.algo}_{time.strftime('%Y%m%d%H%M')}"
    if (not dist_enabled) or rank == 0:
        os.makedirs(save_dir, exist_ok=True)
        os.makedirs(os.path.join(save_dir, "weights"), exist_ok=True)

    train_ds = LandsatFireDataset(train_csv, bands=tuple(args.bands))
    val_ds = LandsatFireDataset(val_csv, bands=tuple(args.bands))

    if dist_enabled:
        train_sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=rank, shuffle=True, drop_last=True)
        val_sampler = DistributedSampler(val_ds, num_replicas=world_size, rank=rank, shuffle=False, drop_last=False)
    else:
        train_sampler = val_sampler = None

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

    model = FDE_UNet(in_channels=len(args.bands), num_classes=1).to(device)
    if dist_enabled:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[local_rank] if torch.cuda.is_available() else None,
            output_device=local_rank if torch.cuda.is_available() else None,
            find_unused_parameters=True,
        )

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    # No scheduler mentioned, but Early Stopping is used.

    weights_dir = os.path.join(save_dir, "weights")
    log_path = os.path.join(save_dir, "train_log.txt")
    hparam_path = os.path.join(save_dir, "hyperparameters.txt")
    best_val = float("inf")
    patience = 5
    counter = 0

    if (not dist_enabled) or rank == 0:
        with open(log_path, "w") as f:
            f.write("epoch,train_loss,val_loss\n")
        with open(hparam_path, "w") as f:
            f.write("===== FDE-UNet Hyperparameters =====\n")
            f.write(f"time: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"algo: {args.algo}\n")
            f.write(f"data_root: {args.data_root}\n")
            f.write(f"bands: {args.bands}\n")
            f.write(f"epochs: {args.epochs}\n")
            f.write(f"batch_size: {args.batch_size}\n")
            f.write(f"lr: {args.lr}\n")
            f.write(f"dist_enabled: {dist_enabled} world_size: {world_size}\n")
            total_params = sum(p.numel() for p in model.parameters())
            train_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            f.write(f"total_params: {total_params}\n")
            f.write(f"trainable_params: {train_params}\n")
            f.write(f"criterion: {type(criterion).__name__}\n")
            f.write(f"early_stopping_patience: {patience}\n")
            f.write("====================================\n")

    def reduce_mean(value: float) -> float:
        if not dist_enabled:
            return value
        tensor = torch.tensor([value], device=device)
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        return (tensor / world_size).item()

    train_curve, val_curve = [], []

    for epoch in range(args.epochs):
        if dist_enabled:
            train_sampler.set_epoch(epoch)

        model.train()
        epoch_sum = 0.0
        iterator = train_loader if ((not dist_enabled) and not torch.cuda.is_available()) else train_loader
        if (not dist_enabled) or rank == 0:
            iterator = tqdm(train_loader, desc=f"Train {epoch + 1}/{args.epochs}", leave=False)

        for images, masks in iterator:
            images = images.to(device, non_blocking=True)
            masks = (masks > 0).float().to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, masks)
            loss.backward()
            optimizer.step()

            epoch_sum += loss.item() * images.size(0)
            if ((not dist_enabled) or rank == 0) and isinstance(iterator, tqdm):
                iterator.set_postfix({"loss": f"{loss.item():.3f}"})

        train_loss_local = epoch_sum / len(train_ds if not dist_enabled else train_sampler.dataset)
        train_loss = reduce_mean(train_loss_local)

        model.eval()
        val_sum = 0.0
        with torch.inference_mode():
            v_iter = val_loader
            if (not dist_enabled) or rank == 0:
                v_iter = tqdm(val_loader, desc=f"Val   {epoch + 1}/{args.epochs}", leave=False)
            for images, masks in v_iter:
                images = images.to(device, non_blocking=True)
                masks = (masks > 0).float().to(device, non_blocking=True)
                logits = model(images)
                val_loss = criterion(logits, masks)
                val_sum += val_loss.item() * images.size(0)
                if ((not dist_enabled) or rank == 0) and isinstance(v_iter, tqdm):
                    v_iter.set_postfix({"loss": f"{val_loss.item():.3f}"})

        val_loss_local = val_sum / len(val_ds if not dist_enabled else val_sampler.dataset)
        val_loss = reduce_mean(val_loss_local)

        if (not dist_enabled) or rank == 0:
            print(f"Epoch {epoch + 1:03d} | train {train_loss:.4f} | val {val_loss:.4f}")
            with open(log_path, "a") as f:
                f.write(f"{epoch + 1},{train_loss:.6f},{val_loss:.6f}\n")
            
            if val_loss < best_val:
                best_val = val_loss
                counter = 0
                torch.save((model.module if dist_enabled else model).state_dict(), os.path.join(weights_dir, "model_best.pth"))
            else:
                counter += 1
                print(f"EarlyStopping counter: {counter} out of {patience}")
            
            if (epoch + 1) % 10 == 0:
                torch.save((model.module if dist_enabled else model).state_dict(), os.path.join(weights_dir, f"model_epoch_{epoch + 1}.pth"))

        train_curve.append(train_loss)
        val_curve.append(val_loss)
        
        # Broadcast early stopping decision
        stop_signal = torch.tensor(1 if counter >= patience else 0, device=device)
        if dist_enabled:
            dist.broadcast(stop_signal, src=0)
        
        if stop_signal.item() == 1:
            if (not dist_enabled) or rank == 0:
                print("Early stopping triggered.")
            break

    if (not dist_enabled) or rank == 0:
        torch.save((model.module if dist_enabled else model).state_dict(), os.path.join(weights_dir, "model_final.pth"))
        plt.figure(figsize=(8, 5))
        plt.plot(train_curve, label="train")
        plt.plot(val_curve, label="val")
        plt.xlabel("epoch")
        plt.ylabel("loss")
        plt.title("FDE-UNet BCEWithLogitsLoss")
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(save_dir, "loss_curves.png"), dpi=150, bbox_inches="tight")
        plt.close()
        print("训练结束。最优验证损失:", best_val)

    cleanup_ddp()


if __name__ == "__main__":
    main()
