import os
import sys
import argparse
from datetime import datetime
from typing import Tuple
import logging

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from dataset import LandsatFireDataset
from models.activefire_unet_baseline import ActiveFireUNetBaseline
from loss import FocalTverskyLoss, SpatialFocalTverskyLoss, MaskedL1Loss, get_criterion_info

# CUDA_VISIBLE_DEVICES=1,4,5,6,7 torchrun --nproc_per_node=5 --master_port=65510 exp/train_scripts/train_activefire_baseline_unet.py

RUN_ID = datetime.now().strftime("%Y%m%d%H%M")
DEFAULT_SAVE_DIR = f"output/ActiveFireBaseline/voting_{RUN_ID}"

DATA_ROOT = "data/splits_activefire"
ALGORITHM = "voting"
# DATA_ROOT = "data/splits_land8fire"
# ALGORITHM = "Land8Fire"
BANDS = (7,6,5)
BATCH_SIZE = 16
NUM_WORKERS = 4 
EPOCHS = 50
LR = 1e-3
VAL_INTERVAL = 1
SAVE_INTERVAL = 10

logger = logging.getLogger("af_baseline_train")


def parse_args():
    p = argparse.ArgumentParser(description="ActiveFire UNet Baseline Training")
    p.add_argument("--data-root", type=str, default=DATA_ROOT)
    p.add_argument("--algorithm", type=str, default=ALGORITHM,
                   choices=["voting", "Kumar-Roy", "Murphy", "Schroeder", "intersection", "Land8Fire"])
    p.add_argument("--fire-category", type=str, default=None,
                   choices=["very_few", "few", "many", "very_many"])
    p.add_argument("--bands", type=int, nargs=3, default=BANDS)
    p.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    p.add_argument("--num-workers", type=int, default=NUM_WORKERS)
    p.add_argument("--epochs", type=int, default=EPOCHS)
    p.add_argument("--lr", type=float, default=LR)
    p.add_argument("--base-filters", type=int, default=64)
    p.add_argument("--save-dir", type=str, default=None)
    p.add_argument("--loss", type=str, default="bce", choices=["bce","focal_tversky","spatial_focal_tversky"])
    return p.parse_args()


def get_csv_paths(data_root: str, algorithm: str, fire_category: str = None) -> Tuple[str,str]:
    if fire_category is None:
        return os.path.join(data_root, f"{algorithm}_train.csv"), os.path.join(data_root, f"{algorithm}_val.csv")
    if algorithm == "Land8Fire":
        subdir = os.path.join(data_root, "by_fire_pixels")
    else:
        subdir = os.path.join(data_root, f"by_fire_pixels_{algorithm}")
    return (os.path.join(subdir, "train", f"{algorithm}_train_{fire_category}.csv"),
            os.path.join(subdir, "val", f"{algorithm}_val_{fire_category}.csv"))


def setup_logging(save_dir: str):
    os.makedirs(save_dir, exist_ok=True)
    log_path = os.path.join(save_dir, "train.log")
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(log_path, mode="w")
    fh.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(fh)
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
    logger.addHandler(ch)
    logger.info("Logging initialized: %s", log_path)


def init_dist():
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        dist.init_process_group(backend="nccl", init_method="env://")
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)
        return True, dist.get_rank(), dist.get_world_size(), local_rank
    return False, 0, 1, 0


def main():
    args = parse_args()
    distributed, rank, world_size, local_rank = init_dist()

    save_dir = args.save_dir or f"output/ActiveFireBaseline/{args.algorithm}{'_' + args.fire_category if args.fire_category else ''}_{RUN_ID}"
    if rank == 0:
        setup_logging(save_dir)
        logger.info("ActiveFire UNet Baseline Training Start")
        logger.info("Args: %s", vars(args))

    train_csv, val_csv = get_csv_paths(args.data_root, args.algorithm, args.fire_category)
    train_ds = LandsatFireDataset(train_csv, bands=tuple(args.bands))
    val_ds = LandsatFireDataset(val_csv, bands=tuple(args.bands))

    train_sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=rank, shuffle=True) if distributed else None
    val_sampler = DistributedSampler(val_ds, num_replicas=world_size, rank=rank, shuffle=False) if distributed else None

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=train_sampler, shuffle=(train_sampler is None),
                              num_workers=args.num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, sampler=val_sampler, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)

    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    model = ActiveFireUNetBaseline(n_channels=len(args.bands), base_filters=args.base_filters)
    model.to(device)

    if distributed:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)

    if args.loss == "bce":
        criterion = nn.BCEWithLogitsLoss()
    elif args.loss == "focal_tversky":
        criterion = FocalTverskyLoss()
    else:
        criterion = SpatialFocalTverskyLoss()

    if rank == 0:
        logger.info("Loss: %s", get_criterion_info(criterion))

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.8, patience=5)

    best_val = float("inf")

    for epoch in range(args.epochs):
        if distributed:
            train_sampler.set_epoch(epoch)
        model.train()
        epoch_loss = 0.0
        iter_loader = tqdm(train_loader, disable=(rank != 0), desc=f"Train {epoch+1}/{args.epochs}")
        for images, masks in iter_loader:
            images = images.to(device, non_blocking=True)
            masks = (masks > 0).float()
            if masks.dim() == 3:
                masks = masks.unsqueeze(1)
            masks = masks.to(device, non_blocking=True)

            optimizer.zero_grad()
            logits = model(images)
            if logits.shape[2:] != masks.shape[2:]:
                logits = nn.functional.interpolate(logits, size=masks.shape[2:], mode="bilinear", align_corners=False)
            loss = criterion(logits, masks)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()
            if rank == 0:
                iter_loader.set_postfix(loss=f"{loss.item():.4f}")
        epoch_loss /= max(1, len(train_loader))

        # Validation
        if (epoch % VAL_INTERVAL) == 0:
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for images, masks in val_loader:
                    images = images.to(device, non_blocking=True)
                    masks = (masks > 0).float()
                    if masks.dim() == 3:
                        masks = masks.unsqueeze(1)
                    masks = masks.to(device, non_blocking=True)
                    logits = model(images)
                    if logits.shape[2:] != masks.shape[2:]:
                        logits = nn.functional.interpolate(logits, size=masks.shape[2:], mode="bilinear", align_corners=False)
                    loss = criterion(logits, masks)
                    val_loss += loss.item() * images.size(0)
            val_loss /= max(1, len(val_ds))
            scheduler.step(val_loss)

            if rank == 0:
                logger.info("Epoch %d | Train Loss: %.6f | Val Loss: %.6f", epoch+1, epoch_loss, val_loss)
                if val_loss < best_val:
                    best_val = val_loss
                    weights_dir = os.path.join(save_dir, "weights")
                    os.makedirs(weights_dir, exist_ok=True)
                    torch.save(model.module.state_dict() if isinstance(model, DDP) else model.state_dict(), os.path.join(weights_dir, "model_best.pth"))
                    logger.info("Saved best model (val=%.6f)", best_val)
                if (epoch + 1) % SAVE_INTERVAL == 0:
                    weights_dir = os.path.join(save_dir, "weights")
                    os.makedirs(weights_dir, exist_ok=True)
                    torch.save(model.module.state_dict() if isinstance(model, DDP) else model.state_dict(), os.path.join(weights_dir, f"model_epoch_{epoch+1}.pth"))
        
    if rank == 0:
        logger.info("Training finished. Best Val Loss: %.6f", best_val)

    if distributed:
        dist.barrier()
        dist.destroy_process_group()

if __name__ == "__main__":
    main()
