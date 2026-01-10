"""评估脚本: FDE-UNet

- 支持单机/分布式评估，与 baseline/V4 指标输出一致
- 默认从 save-dir/weights/model_best.pth 读取参数，按需可指定 --model-path
- 计算 Precision/Recall/F1/IoU/mIoU，并保存报告与样例

示例：
CUDA_VISIBLE_DEVICES=0,1,2,7 torchrun --nproc_per_node=4 --master_port=65533 exp/eval_scripts/eval_FDE_UNet.py --data-root data/splits_merged_pixels --algo large --save-dir output/FDE_UNet/large_202512261511
python exp/eval_scripts/eval_FDE_UNet.py --data-root data/splits_activefire --algo voting --bands 1 2 3 4 5 6 7 8 9 10 --threshold 0.5 --save-dir output/FDE_UNet/voting_202512041154
"""

from __future__ import annotations

import os
import sys
import time
import argparse
from typing import Optional, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, RandomSampler, DistributedSampler
from tqdm import tqdm

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from dataset import LandsatFireDataset
from models.FDE_Net import FDE_UNet


def _is_preinit_main() -> bool:
    return os.environ.get("RANK", "0") == "0"


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate FDE-UNet")
    p.add_argument("--dist", action="store_true", help="启用分布式评估")
    p.add_argument("--data-root", type=str, default="data/splits_activefire")
    p.add_argument("--algo", type=str, default="voting")
    p.add_argument("--bands", type=int, nargs=10, default=(1, 2, 3, 4, 5, 6, 7, 8, 9, 10))
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--save-dir", type=str, default="output/FDE_UNet/voting_date")
    p.add_argument("--model-path", type=str, default=None)
    p.add_argument("--backend", type=str, default="nccl", choices=["nccl", "gloo"])
    p.add_argument("--samples", type=int, default=5)
    return p.parse_args()


def get_csv(root: str, algo: str) -> str:
    return os.path.join(root, f"{algo}_test.csv")


def should_dist(args) -> bool:
    return args.dist or int(os.environ.get("WORLD_SIZE", "1")) > 1


def setup_dist(backend: str) -> Optional[int]:
    if int(os.environ.get("WORLD_SIZE", "1")) <= 1 and "LOCAL_RANK" not in os.environ:
        return None
    if not dist.is_initialized():
        dist.init_process_group(backend=backend, init_method="env://")
    return dist.get_rank()


def cleanup_dist():
    if dist.is_initialized():
        dist.destroy_process_group()


def main():
    args = parse_args()
    if _is_preinit_main():
        print("========== FDE-UNet 评估 ==========")
        print(f"时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")

    test_csv = get_csv(args.data_root, args.algo)
    if _is_preinit_main():
        print(f"测试 CSV: {test_csv}")

    dist_flag = should_dist(args)
    rank = setup_dist(args.backend) if dist_flag else None
    local_rank = int(os.environ.get("LOCAL_RANK", "0")) if dist_flag else 0

    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu") if dist_flag else torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = LandsatFireDataset(test_csv, bands=tuple(args.bands))
    sampler = DistributedSampler(dataset, shuffle=False, drop_last=False) if dist.is_initialized() else None
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    model = FDE_UNet(in_channels=len(args.bands), num_classes=1).to(device)
    weights_dir = os.path.join(args.save_dir, "weights")
    default_path = os.path.join(weights_dir, "model_best.pth")
    weight_path = args.model_path or default_path
    if (not dist_flag) or rank in (None, 0):
        print(f"加载权重: {weight_path}")
    state = torch.load(weight_path, map_location=device)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    if all(k.startswith("module.") for k in state.keys()):
        state = {k[len("module."):]: v for k, v in state.items()}
    model.load_state_dict(state)

    if dist.is_initialized():
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank] if torch.cuda.is_available() else None)
    model.eval()
    eval_model = model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model

    totals = torch.zeros(5, dtype=torch.long, device=device)
    with torch.inference_mode():
        iterator = loader if not (dist_flag and rank != 0) else loader
        if (not dist_flag) or rank == 0:
            iterator = tqdm(loader, desc=f"Test [{len(dataset)}]")
        for images, masks in iterator:
            images = images.to(device, non_blocking=True)
            masks = (masks > 0).to(device, non_blocking=True)
            logits = eval_model(images)
            probs = torch.sigmoid(logits)
            preds = probs >= args.threshold
            gt = masks.bool()
            totals[0] += torch.logical_and(preds, gt).sum()
            totals[1] += torch.logical_and(preds, ~gt).sum()
            totals[2] += torch.logical_and(~preds, gt).sum()
            totals[3] += images.size(0)
            totals[4] += gt.numel()

    if dist.is_initialized():
        dist.all_reduce(totals, op=dist.ReduceOp.SUM)

    tp, fp, fn, sample_cnt, total_px = totals.tolist()
    tn = max(0, total_px - tp - fp - fn)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    iou_fire = tp / (tp + fp + fn) if (tp + fp + fn) else 0.0
    iou_bg = tn / (tn + fp + fn) if (tn + fp + fn) else 0.0
    miou = (iou_fire + iou_bg) / 2.0

    os.makedirs(args.save_dir, exist_ok=True)
    report_path = os.path.join(args.save_dir, "eval_report.txt")
    if (not dist_flag) or rank == 0:
        with open(report_path, "w") as f:
            f.write("===== FDE-UNet Evaluation =====\n")
            f.write(f"timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"threshold: {args.threshold}\n")
            f.write(f"samples: {sample_cnt}\n")
            f.write(f"precision: {precision:.4f}\n")
            f.write(f"recall: {recall:.4f}\n")
            f.write(f"f1: {f1:.4f}\n")
            f.write(f"fire_iou: {iou_fire:.4f}\n")
            f.write(f"back_iou: {iou_bg:.4f}\n")
            f.write(f"miou: {miou:.4f}\n")
        print("Precision/Recall/F1:", f"{precision:.4f}", f"{recall:.4f}", f"{f1:.4f}")
        print("IoU (fire/bg/mIoU):", f"{iou_fire:.4f}", f"{iou_bg:.4f}", f"{miou:.4f}")

        sample_dir = os.path.join(args.save_dir, "pred_samples")
        os.makedirs(sample_dir, exist_ok=True)
        sample_loader = DataLoader(
            loader.dataset,
            batch_size=max(1, args.samples),
            sampler=RandomSampler(loader.dataset),
            num_workers=loader.num_workers,
            pin_memory=torch.cuda.is_available(),
        )
        images, masks = next(iter(sample_loader))
        images = images.to(device)
        masks = (masks > 0).float().to(device)
        with torch.inference_mode():
            logits = eval_model(images)
            probs = torch.sigmoid(logits)
            preds = (probs >= args.threshold).float().cpu().numpy()
        imgs_np = images.cpu().numpy()
        gts_np = masks.cpu().numpy()
        for idx in range(imgs_np.shape[0]):
            rgb = imgs_np[idx][:3]
            rgb_img = (rgb * 255).clip(0, 255).astype(np.uint8)
            rgb_img = np.transpose(rgb_img, (1, 2, 0))
            __import__('PIL').Image.fromarray(rgb_img).save(os.path.join(sample_dir, f"sample_{idx:02d}_rgb.png"))
            __import__('PIL').Image.fromarray((gts_np[idx][0] * 255).astype(np.uint8)).save(os.path.join(sample_dir, f"sample_{idx:02d}_gt.png"))
            __import__('PIL').Image.fromarray((preds[idx][0] * 255).astype(np.uint8)).save(os.path.join(sample_dir, f"sample_{idx:02d}_pred.png"))

        rows = images.size(0)
        fig, axes = plt.subplots(rows, 3, figsize=(12, 4 * rows))
        if rows == 1:
            axes = np.expand_dims(axes, 0)
        for i in range(rows):
            rgb = imgs_np[i][:3]
            rgb_img = (rgb * 255).clip(0, 255).astype(np.uint8)
            rgb_img = np.transpose(rgb_img, (1, 2, 0))
            axes[i, 0].imshow(rgb_img)
            axes[i, 0].set_title("Input")
            axes[i, 0].axis("off")
            axes[i, 1].imshow(gts_np[i][0], cmap="gray")
            axes[i, 1].set_title("GT")
            axes[i, 1].axis("off")
            axes[i, 2].imshow(preds[i][0], cmap="gray")
            axes[i, 2].set_title("Pred")
            axes[i, 2].axis("off")
        plt.tight_layout()
        vis_path = os.path.join(args.save_dir, "prediction_grid.png")
        plt.savefig(vis_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"评估报告: {report_path}")
        print(f"样例目录: {sample_dir}")

    cleanup_dist()


if __name__ == "__main__":
    main()
