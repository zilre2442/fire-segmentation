"""训练脚本: RGS-Net V4 StageFusion 变体

特性概述:
- 双解码器结构与 RGS-Net V4 相同，但在两个解码器的中间三次 UpBlock 之后执行一次可学习权重的差分融合。
- 主损失使用 fused_logits 进行 SpatialFocalTverskyLoss 监督；辅助损失为 seg_logits 与 rec_logits。

示例命令:
CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 --master_port=65530 exp/train_scripts/train_RGS_Net_V4_stagefusion.py --data-root data/splits_activefire --algo voting
CUDA_VISIBLE_DEVICES=7 python exp/train_scripts/train_RGS_Net_V4_stagefusion.py --data-root data/splits_activefire --algo voting
"""

from __future__ import annotations

import os
import sys
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

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from dataset import LandsatFireDataset
from models.RGS_Net_V4 import RGSNetV4StageFusion
from loss import SpatialFocalTverskyLoss, get_criterion_info, FocalTverskyLoss, FocalLoss, TverskyLoss, DiceLoss, SpatialFocalLoss


def parse_args():
    p = argparse.ArgumentParser(description="Train RGS-Net V4 StageFusion (single or multi-GPU)")
    p.add_argument("--data-root", type=str, default="data/splits_activefire")
    p.add_argument("--algo", type=str, default="voting")
    p.add_argument("--bands", type=int, nargs=3, default=(7, 6, 5))
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--base-filters", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--save-dir", type=str, default=None)
    p.add_argument("--stage1", type=int, default=20, help="aux weight=1.0 结束 epoch (exclusive)")
    p.add_argument("--stage2", type=int, default=40, help="aux weight=0.5 结束 epoch (exclusive); >=stage2 ->0.0")
    p.add_argument("--mid-fuse-init", type=float, default=1.0, help="中间融合权重初始化")
    p.add_argument("--dist", action="store_true", help="启用分布式 (torchrun 设置 env 后自动检测)")
    p.add_argument("--backend", type=str, default="nccl", choices=["nccl", "gloo"], help="分布式后端")
    return p.parse_args()


def get_csv_paths(root: str, algo: str) -> Tuple[str, str]:
    train_csv = os.path.join(root, f"{algo}_train.csv")
    val_csv = os.path.join(root, f"{algo}_val.csv")
    return train_csv, val_csv


def aux_weight(epoch: int, s1: int, s2: int) -> float:
    """已废弃的阶段性辅助权重接口，现不再使用。"""
    raise RuntimeError("aux_weight is deprecated; use fixed loss weights instead.")


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
    device = (
        torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
        if dist_enabled
        else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )

    train_csv, val_csv = get_csv_paths(args.data_root, args.algo)
    save_dir = args.save_dir or f"output/RGS_Net_V4_stagefusion/{args.algo}_{time.strftime('%Y%m%d%H%M')}"
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

    model = RGSNetV4StageFusion(
        n_channels=len(args.bands),
        base_filters=args.base_filters,
        fuse_init=args.mid_fuse_init,
    ).to(device)
    if dist_enabled:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[local_rank] if torch.cuda.is_available() else None,
            output_device=local_rank if torch.cuda.is_available() else None,
            find_unused_parameters=False,
        )

    # seg_loss_fn = SpatialFocalTverskyLoss(
    #     alpha_tversky=0.4,
    #     beta_tversky=0.6,
    #     gamma_focal=1.6,
    #     focal_alpha=0.85,
    #     lambda_focal=0.4,
    #     lambda_tversky=0.6,
    #     weight_min=100.0,
    #     weight_max=1000.0,
    #     area_gamma=0.75,
    #     background_weight=1.0,
    #     connectivity=8,
    # )
    # seg_loss_fn = nn.BCEWithLogitsLoss()
    seg_loss_fn = SpatialFocalLoss(
        gamma_0=2.0,
    )
    recon_loss_fn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.8, patience=5)

    best_val = float("inf")
    log_path = os.path.join(save_dir, "train_log.txt")
    hparam_path = os.path.join(save_dir, "hyperparameters.txt")
    if (not dist_enabled) or rank == 0:
        with open(log_path, "w") as log:
            log.write("epoch,train_loss,val_loss,fused_loss,seg_loss,rec_loss\n")
        with open(hparam_path, "w") as f:
            f.write("===== RGS-Net V4 StageFusion Hyperparameters =====\n")
            f.write(f"time: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"algo: {args.algo}\n")
            f.write(f"data_root: {args.data_root}\n")
            f.write(f"bands: {args.bands}\n")
            f.write(f"epochs: {args.epochs}\n")
            f.write(f"batch_size: {args.batch_size}\n")
            f.write(f"lr: {args.lr}\n")
            f.write(f"base_filters: {args.base_filters}\n")
            f.write(f"stage1: {args.stage1} stage2: {args.stage2}\n")
            f.write(f"mid_fuse_init: {args.mid_fuse_init}\n")
            f.write("loss_weights: fused(0.7) seg(0.15) rec(0.15) [fixed]\n")
            f.write(f"dist_enabled: {dist_enabled} world_size: {world_size}\n")
            f.write("\nModel Info:\n")
            model_to_log = model.module if hasattr(model, "module") else model
            total_params = sum(p.numel() for p in model_to_log.parameters())
            trainable_params = sum(p.numel() for p in model_to_log.parameters() if p.requires_grad)
            f.write(f"total_params: {total_params}\n")
            f.write(f"trainable_params: {trainable_params}\n")
            f.write(f"rec_weight: {model_to_log.rec_weight.item():.4f}\n")
            mid_weights = ", ".join(f"{w.item():.4f}" for w in model_to_log.mid_fuse_weights)
            f.write(f"mid_fuse_weights: [{mid_weights}]\n")
            f.write("\n[Losses]\n")
            f.write("[Segmentation Loss]\n")
            f.write(f"  name_flag: {type(seg_loss_fn).__name__}\n")
            try:
                f.write(get_criterion_info(seg_loss_fn) + "\n")
            except Exception:
                f.write(f"  name: {type(seg_loss_fn).__name__}\n")
            f.write("[Reconstruction Loss]\n")
            try:
                f.write(f"  name: {type(recon_loss_fn).__name__}\n")
                if hasattr(recon_loss_fn, "reduction"):
                    f.write(f"  reduction: {recon_loss_fn.reduction}\n")
                f.write("  target: 1 - mask (background)\n")
                f.write("  pred: rec_logits (BCEWithLogitsLoss)\n")
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
            train_sampler.set_epoch(epoch)

        model.train()
        train_sum = fused_sum = seg_sum = rec_sum = 0.0
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
            rec_loss = recon_loss_fn(rec_logits, rec_target)
            # 固定损失权重：主损失 0.6，两个辅助损失各 0.2
            loss = 0.6 * fused_loss + 0.2 * seg_loss + 0.2 * rec_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            batch = images.size(0)
            train_sum += loss.item() * batch
            fused_sum += fused_loss.item() * batch
            seg_sum += seg_loss.item() * batch
            rec_sum += rec_loss.item() * batch

            if ((not dist_enabled) or rank == 0) and isinstance(iterator, tqdm):
                model_to_show = model.module if hasattr(model, "module") else model
                mid_vals = ",".join(f"{w.item():.2f}" for w in model_to_show.mid_fuse_weights)
                iterator.set_postfix({
                    'loss': f"{loss.item():.3f}",
                    'fused': f"{fused_loss.item():.3f}",
                    'seg': f"{seg_loss.item():.3f}",
                    'rec': f"{rec_loss.item():.3f}",
                    'rw': f"{model_to_show.rec_weight.item():.2f}",
                    'mid': mid_vals,
                })

        denom_train = len(train_ds if not dist_enabled else train_sampler.dataset)
        avg_train_local = train_sum / denom_train
        avg_train = reduce_scalar(avg_train_local)
        avg_fused = reduce_scalar(fused_sum / denom_train)
        avg_seg = reduce_scalar(seg_sum / denom_train)
        avg_rec = reduce_scalar(rec_sum / denom_train)

        model.eval()
        val_sum = val_fused_sum = val_seg_sum = val_rec_sum = 0.0
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
                rec_loss = recon_loss_fn(rec_logits, rec_target)
                val_loss = fused_loss
                batch = images.size(0)
                val_sum += val_loss.item() * batch
                val_fused_sum += fused_loss.item() * batch
                val_seg_sum += seg_loss.item() * batch
                val_rec_sum += rec_loss.item() * batch
                if ((not dist_enabled) or rank == 0) and isinstance(v_iterator, tqdm):
                    v_iterator.set_postfix({
                        'loss': f"{val_loss.item():.3f}",
                        'fused': f"{fused_loss.item():.3f}",
                        'seg': f"{seg_loss.item():.3f}",
                        'rec': f"{rec_loss.item():.3f}",
                    })

        denom_val = len(val_ds if not dist_enabled else val_sampler.dataset)
        avg_val_local = val_sum / denom_val
        avg_val = reduce_scalar(avg_val_local)
        avg_val_fused = reduce_scalar(val_fused_sum / denom_val)
        avg_val_seg = reduce_scalar(val_seg_sum / denom_val)
        avg_val_rec = reduce_scalar(val_rec_sum / denom_val)

        if (not dist_enabled) or rank == 0:
            print(
                f"Epoch {epoch + 1:03d} | train {avg_train:.4f} (fused {avg_fused:.4f} seg {avg_seg:.4f} rec {avg_rec:.4f}) | "
                f"val {avg_val:.4f} (fused {avg_val_fused:.4f} seg {avg_val_seg:.4f} rec {avg_val_rec:.4f})"
            )
            with open(log_path, "a") as log:
                log.write(
                    f"{epoch + 1},{avg_train:.6f},{avg_val:.6f},{avg_fused:.6f},{avg_seg:.6f},{avg_rec:.6f}\n"
                )
            scheduler.step(avg_val)
            model_to_save = model.module if dist_enabled else model
            if avg_val < best_val:
                best_val = avg_val
                torch.save(model_to_save.state_dict(), os.path.join(weights_dir, "model_best.pth"))
            if (epoch + 1) % 10 == 0:
                torch.save(model_to_save.state_dict(), os.path.join(weights_dir, f"model_epoch_{epoch+1}.pth"))

        train_curve.append(avg_train)
        val_curve.append(avg_val)

    if (not dist_enabled) or rank == 0:
        model_to_save = model.module if dist_enabled else model
        torch.save(model_to_save.state_dict(), os.path.join(weights_dir, "model_final.pth"))
        plt.figure(figsize=(8, 5))
        plt.plot(train_curve, label="train")
        plt.plot(val_curve, label="val")
        plt.xlabel("epoch")
        plt.ylabel("loss")
        plt.title("RGS-Net V4 StageFusion Loss")
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(save_dir, "loss_curves.png"), dpi=150, bbox_inches="tight")
        plt.close()
        print("训练结束。最优验证损失:", best_val)
    cleanup_ddp()


if __name__ == "__main__":
    main()
