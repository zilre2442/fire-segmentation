"""训练脚本: RGS-Net V4

主损失: fused_logits 使用 SpatialFocalTverskyLoss 对火点分割监督。
辅助损失: seg_logits(同上) + rec_logits(背景监督, L1Loss 对 sigmoid(rec_logits) vs 1 - fire_mask)。
辅助损失权重分三阶段: 1.0 -> 0.5 -> 0.0。

为保持简洁：仅单卡/普通训练，支持选择波段与基础参数。可扩展时再加 DDP。

CUDA_VISIBLE_DEVICES=1,4,5,6,7 torchrun --nproc_per_node=5 --master_port=65531 train_RGS_Net_V4.py --data-root data/splits_activefire --algo voting --bands 7 6 5 --epochs 60 --batch-size 16 --lr 1e-3 --base-filters 64 --num-workers 4 --dist
"""

from __future__ import annotations

import os
import time
import argparse
from typing import Tuple, Optional

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import random
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from dataset import LandsatFireDataset
from models.RGS_Net_V4 import RGSNetV4
from loss import (
    SpatialFocalTverskyLoss,
    FocalTverskyLoss,
    FocalLoss,
    TverskyLoss,
    DiceLoss,
    get_criterion_info,
)


SEG_LOSS_CHOICES = ("focal_tversky", "focal", "tversky", "bce", "dice")


def parse_args():
    p = argparse.ArgumentParser(description="Train RGS-Net V4 (single or multi-GPU)")
    p.add_argument("--data-root", type=str, default="data/splits_activefire")
    p.add_argument("--algo", type=str, default="voting")
    p.add_argument("--bands", type=int, nargs=3, default=(7, 6, 5))
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--base-filters", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--save-dir", type=str, default=None)
    p.add_argument("--stage1", type=int, default=20, help="aux weight=1.0 结束 epoch (exclusive)")
    p.add_argument("--stage2", type=int, default=40, help="aux weight=0.5 结束 epoch (exclusive); >=stage2 ->0.0")
    p.add_argument("--dist", action="store_true", help="启用分布式 (torchrun 设置 env 后自动检测)")
    p.add_argument("--backend", type=str, default="nccl", choices=["nccl", "gloo"], help="分布式后端")
    p.add_argument("--seg-loss", type=str, default="focal_tversky", choices=SEG_LOSS_CHOICES,
                   help="分割损失类型")
    return p.parse_args()


def get_csv_paths(root: str, algo: str) -> Tuple[str, str]:
    train_csv = os.path.join(root, f"{algo}_train.csv")
    val_csv = os.path.join(root, f"{algo}_val.csv")
    return train_csv, val_csv


def aux_weight(epoch: int, s1: int, s2: int) -> float:
    if epoch < s1:
        return 1.0
    if epoch < s2:
        return 0.5
    return 0.0


def is_dist_requested(args) -> bool:
    # 显式 --dist 或已有 torchrun 环境变量
    return args.dist or int(os.environ.get("WORLD_SIZE", "1")) > 1


def setup_ddp(args) -> Tuple[bool, int, int, Optional[int]]:
    dist_flag = is_dist_requested(args)
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    # 若仅单进程且缺少必要 env，回退为非分布式
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


def build_seg_loss(loss_name: str) -> nn.Module:
    name = loss_name.lower()
    if name == "focal_tversky":
        return FocalTverskyLoss(alpha=0.4, beta=0.6, gamma=1.6, focal_alpha=0.85, lambda_focal=0.4, lambda_tversky=0.6)
    if name == "focal":
        return FocalLoss(gamma=2.0, alpha=0.25)
    if name == "tversky":
        return TverskyLoss(alpha=0.6, beta=0.4)
    if name == "bce":
        return nn.BCEWithLogitsLoss()
    if name == "dice":
        return DiceLoss()
    raise ValueError(f"未知的 seg loss: {loss_name}")


def main():
    args = parse_args()
    # Seeding & backend
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    dist_enabled, rank, world_size, local_rank = setup_ddp(args)
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu") if dist_enabled else torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_csv, val_csv = get_csv_paths(args.data_root, args.algo)
    save_dir = args.save_dir or f"output/RGS_Net_V4/{args.algo}_{time.strftime('%Y%m%d%H%M')}"
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
        drop_last=dist_enabled,  # 保持各进程 batch 尺寸一致
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        sampler=val_sampler,
        num_workers=max(1, args.num_workers // 2),
        pin_memory=True,
    )

    model = RGSNetV4(n_channels=len(args.bands), base_filters=args.base_filters).to(device)
    if dist_enabled:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank] if torch.cuda.is_available() else None, output_device=local_rank if torch.cuda.is_available() else None, find_unused_parameters=False)

    seg_loss_fn = SpatialFocalTverskyLoss(
        alpha_tversky=0.4,
        beta_tversky=0.6,
        gamma_focal=1.6,
        focal_alpha=0.85,
        lambda_focal=0.4,
        lambda_tversky=0.6,
        weight_min=100.0,
        weight_max=1000.0,
        area_gamma=0.75,
        background_weight=1.0,
        connectivity=8,
        )
    # seg_loss_fn = build_seg_loss(args.seg_loss)
    recon_loss_fn = nn.L1Loss()
    # recon_loss_fn = FocalTverskyLoss(
    #     alpha=0.6,
    #     beta=0.4,
    #     gamma=1.6,
    #     focal_alpha=0.85,
    #     lambda_focal=0.7,
    #     lambda_tversky=0.3,
    # )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.8, patience=5)

    best_val = float("inf")
    log_path = os.path.join(save_dir, "train_log.txt")
    hparam_path = os.path.join(save_dir, "hyperparameters.txt")
    if (not dist_enabled) or rank == 0:
        with open(log_path, "w") as log:
            log.write("epoch,train_loss,val_loss,aux_w,fused_loss,seg_loss,rec_loss\n")
        # 简洁版超参数记录
        with open(hparam_path, "w") as f:
            f.write("===== RGS-Net V4 Hyperparameters =====\n")
            f.write(f"time: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"algo: {args.algo}\n")
            f.write(f"data_root: {args.data_root}\n")
            f.write(f"bands: {args.bands}\n")
            f.write(f"epochs: {args.epochs}\n")
            f.write(f"batch_size: {args.batch_size}\n")
            f.write(f"lr: {args.lr}\n")
            f.write(f"base_filters: {args.base_filters}\n")
            f.write(f"stage1: {args.stage1} stage2: {args.stage2}\n")
            f.write(f"aux_weights: 0-<{args.stage1}:0.2 | {args.stage1}-<{args.stage2}:0.6 | >= {args.stage2}:1.0\n")
            f.write(f"dist_enabled: {dist_enabled} world_size: {world_size}\n")
            f.write("\nModel Info:\n")
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            f.write(f"total_params: {total_params}\n")
            f.write(f"trainable_params: {trainable_params}\n")
            if hasattr(model, 'module'):
                f.write(f"rec_weight: {model.module.rec_weight.item():.4f}\n")
            else:
                f.write(f"rec_weight: {model.rec_weight.item():.4f}\n")
            # 损失函数与参数记录
            f.write("\n[Losses]\n")
            f.write("[Segmentation Loss]\n")
            f.write(f"  name_flag: {args.seg_loss}\n")
            try:
                f.write(get_criterion_info(seg_loss_fn) + "\n")
            except Exception:
                f.write(f"  name: {type(seg_loss_fn).__name__}\n")
            f.write("[Reconstruction Loss]\n")
            # nn.L1Loss 没有在 get_criterion_info 中专门实现，手动记录关键参数
            try:
                f.write(f"  name: {type(recon_loss_fn).__name__}\n")
                if hasattr(recon_loss_fn, 'reduction'):
                    f.write(f"  reduction: {recon_loss_fn.reduction}\n")
                f.write("  target: 1 - mask (background)\n")
                f.write("  pred: sigmoid(rec_logits)\n")
            except Exception:
                f.write("  (custom or unknown parameters)\n")
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
            train_sampler.set_epoch(epoch)  # 保持 shuffle
        model.train()
        aw = aux_weight(epoch, args.stage1, args.stage2)
        train_sum = 0.0
        fused_sum = 0.0
        seg_sum = 0.0
        rec_sum = 0.0
        iterator = train_loader
        if (not dist_enabled) or rank == 0:
            iterator = tqdm(train_loader, desc=f"Train {epoch+1}/{args.epochs}", leave=False)
        for images, masks in iterator:
            images = images.to(device, non_blocking=True)
            masks = (masks > 0).float().to(device, non_blocking=True)
            seg_logits, rec_logits, fused_logits = model(images)
            fused_loss = seg_loss_fn(fused_logits, masks)
            seg_loss = seg_loss_fn(seg_logits, masks)
            rec_target = 1.0 - masks
            rec_loss = recon_loss_fn(torch.sigmoid(rec_logits), rec_target)
            loss = fused_loss + aw * (seg_loss + rec_loss)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            train_sum += loss.item() * images.size(0)
            fused_sum += fused_loss.item() * images.size(0)
            seg_sum += seg_loss.item() * images.size(0)
            rec_sum += rec_loss.item() * images.size(0)
            if ((not dist_enabled) or rank == 0) and isinstance(iterator, tqdm):
                iterator.set_postfix({
                    'loss': f"{loss.item():.3f}",
                    'fused': f"{fused_loss.item():.3f}",
                    'seg': f"{seg_loss.item():.3f}",
                    'rec': f"{rec_loss.item():.3f}",
                    'aux_w': f"{aw:.2f}",
                    'rw': f"{(model.module.rec_weight if hasattr(model,'module') else model.rec_weight).item():.2f}"})
        avg_train_local = train_sum / len(train_ds if not dist_enabled else train_sampler.dataset)
        avg_train = reduce_scalar(avg_train_local)
        avg_fused = reduce_scalar(fused_sum / len(train_ds if not dist_enabled else train_sampler.dataset))
        avg_seg = reduce_scalar(seg_sum / len(train_ds if not dist_enabled else train_sampler.dataset))
        avg_rec = reduce_scalar(rec_sum / len(train_ds if not dist_enabled else train_sampler.dataset))

        # 验证
        model.eval()
        val_sum = 0.0
        val_fused_sum = 0.0
        val_seg_sum = 0.0
        val_rec_sum = 0.0
        with torch.inference_mode():
            v_iterator = val_loader
            if (not dist_enabled) or rank == 0:
                v_iterator = tqdm(val_loader, desc=f"Val   {epoch+1}/{args.epochs}", leave=False)
            for images, masks in v_iterator:
                images = images.to(device, non_blocking=True)
                masks = (masks > 0).float().to(device, non_blocking=True)
                seg_logits, rec_logits, fused_logits = model(images)
                fused_loss = seg_loss_fn(fused_logits, masks)
                seg_loss = seg_loss_fn(seg_logits, masks)
                rec_target = 1.0 - masks
                rec_loss = recon_loss_fn(torch.sigmoid(rec_logits), rec_target)
                val_loss = fused_loss + aw * (seg_loss + rec_loss)
                val_sum += val_loss.item() * images.size(0)
                val_fused_sum += fused_loss.item() * images.size(0)
                val_seg_sum += seg_loss.item() * images.size(0)
                val_rec_sum += rec_loss.item() * images.size(0)
                if ((not dist_enabled) or rank == 0) and isinstance(v_iterator, tqdm):
                    v_iterator.set_postfix({
                        'loss': f"{val_loss.item():.3f}",
                        'fused': f"{fused_loss.item():.3f}",
                        'seg': f"{seg_loss.item():.3f}",
                        'rec': f"{rec_loss.item():.3f}",
                        'aux_w': f"{aw:.2f}"})
        avg_val_local = val_sum / len(val_ds if not dist_enabled else val_sampler.dataset)
        avg_val = reduce_scalar(avg_val_local)
        avg_val_fused = reduce_scalar(val_fused_sum / len(val_ds if not dist_enabled else val_sampler.dataset))
        avg_val_seg = reduce_scalar(val_seg_sum / len(val_ds if not dist_enabled else val_sampler.dataset))
        avg_val_rec = reduce_scalar(val_rec_sum / len(val_ds if not dist_enabled else val_sampler.dataset))

        if (not dist_enabled) or rank == 0:
            print(f"Epoch {epoch:03d} | train {avg_train:.4f} (fused {avg_fused:.4f} seg {avg_seg:.4f} rec {avg_rec:.4f}) | "
                  f"val {avg_val:.4f} (fused {avg_val_fused:.4f} seg {avg_val_seg:.4f} rec {avg_val_rec:.4f}) | aux_w {aw:.2f}")
            with open(log_path, "a") as log:
                log.write(f"{epoch},{avg_train:.6f},{avg_val:.6f},{aw:.2f},{avg_fused:.6f},{avg_seg:.6f},{avg_rec:.6f}\n")
            # Scheduler + checkpoints
            scheduler.step(avg_val)
            if avg_val < best_val:
                best_val = avg_val
                torch.save((model.module if dist_enabled else model).state_dict(), os.path.join(weights_dir, "model_best.pth"))
            if (epoch + 1) % 10 == 0:
                torch.save((model.module if dist_enabled else model).state_dict(), os.path.join(weights_dir, f"model_epoch_{epoch+1}.pth"))

        train_curve.append(avg_train)
        val_curve.append(avg_val)

    if (not dist_enabled) or rank == 0:
        # Save final model and curves
        torch.save((model.module if dist_enabled else model).state_dict(), os.path.join(weights_dir, "model_final.pth"))
        plt.figure(figsize=(8,5))
        plt.plot(train_curve, label="train")
        plt.plot(val_curve, label="val")
        plt.xlabel("epoch"); plt.ylabel("loss"); plt.title("RGS-Net V4 Loss")
        plt.legend(); plt.grid(True)
        plt.savefig(os.path.join(save_dir, "loss_curves.png"), dpi=150, bbox_inches="tight")
        plt.close()
        print("训练结束。最优验证损失:", best_val)
    cleanup_ddp()


if __name__ == "__main__":
    main()