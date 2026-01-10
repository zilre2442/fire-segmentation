import os
import sys
import time
import argparse
from typing import Optional

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, RandomSampler, DistributedSampler

# Ensure project root is on sys.path so top-level modules (dataset, loss, utils, models) can be imported
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from dataset import LandsatFireDataset
from models.baseline import UNet

# CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 --master_port=65530 exp/eval_scripts/eval_baseline.py
# CUDA_VISIBLE_DEVICES=6 python exp/eval_scripts/eval_baseline.py

# ---- 预初始化阶段的主进程判定（在 DDP 尚未 init 时使用） ----
def _is_preinit_main() -> bool:
    return os.environ.get("LOCAL_RANK", "0") == "0"


if _is_preinit_main():
    print("========== Baseline 火灾检测评估启动 ==========")
    print(f"当前时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"计算设备: {'GPU可用' if torch.cuda.is_available() else '仅限CPU'}")

# DATA_ROOT = "data/splits_activefire"
# ALGORITHM = "voting"  # 可选: 'Kumar-Roy', 'Murphy', 'Schroeder', 'intersection', 'voting'
DATA_ROOT = "data/splits_merged_pixels"
ALGORITHM = "large"
SAVE_DIR = "output/baseline/large_202512260930"  # 指向已训练模型的目录
TH_FIRE = 0.5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BANDS = (7, 6, 2)

# --------- DDP/CLI 实用函数 ---------
def parse_args():
    parser = argparse.ArgumentParser(description="Distributed evaluation for Baseline UNet")
    parser.add_argument("--dist", action="store_true", help="启用分布式评估 (检测到 LOCAL_RANK 也会自动启用)")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--bands", type=int, nargs=3, default=BANDS, help="选择用于评估的三个波段索引")
    parser.add_argument("--save-dir", type=str, default=SAVE_DIR)
    parser.add_argument("--algo", type=str, default=ALGORITHM)
    return parser.parse_args()


def is_dist_env() -> bool:
    return int(os.environ.get("WORLD_SIZE", "1")) > 1 or "LOCAL_RANK" in os.environ


def setup_distributed(backend: str = "nccl") -> Optional[int]:
    if not is_dist_env():
        return None
    if not dist.is_initialized():
        dist.init_process_group(backend=backend)
    return dist.get_rank()


def cleanup_distributed() -> None:
    if dist.is_initialized():
        dist.destroy_process_group()


def get_local_rank_default() -> int:
    return int(os.environ.get("LOCAL_RANK", "0"))


def is_main_process() -> bool:
    return (not dist.is_initialized()) or dist.get_rank() == 0

if _is_preinit_main():
    print("\n[阶段 1/4] 准备测试数据")
args = parse_args()
ALGORITHM = args.algo
SAVE_DIR = args.save_dir
BANDS = tuple(args.bands)
test_data_csv = os.path.join(DATA_ROOT, f"{ALGORITHM}_test.csv")
if _is_preinit_main():
    print(f"├─ 算法标签: {ALGORITHM}")
    print(f"├─ 测试集CSV: {test_data_csv}")

test_dataset = LandsatFireDataset(test_data_csv, bands=BANDS)
if _is_preinit_main():
    print(f"├─ 测试集样本数: {len(test_dataset)}")

# 初始化 DDP (如需要)，并设置设备
rank = setup_distributed()
local_rank = get_local_rank_default() if is_dist_env() else 0
if DEVICE == 'cuda' and torch.cuda.is_available():
    torch.cuda.set_device(local_rank)

sampler = DistributedSampler(test_dataset, shuffle=False, drop_last=False) if dist.is_initialized() else None
pin_mem = torch.cuda.is_available()
test_loader = DataLoader(
    test_dataset,
    batch_size=args.batch_size,
    shuffle=(sampler is None),
    num_workers=args.num_workers,
    pin_memory=pin_mem,
    sampler=sampler,
)
if is_main_process():
    print(f"└─ 数据加载器创建完成: {len(test_loader)} batches")


if is_main_process():
    print("\n[阶段 2/4] 加载预训练模型")
model_name = "model_best"
param_path = os.path.join(SAVE_DIR, "weights", f"{model_name}.pth")
if is_main_process():
    print(f"├─ 参数路径: {param_path}")

model = UNet(n_channels=len(BANDS), n_classes=1, n_filters=64)
if is_main_process():
    print(f"├─ 网络架构: {model.__class__.__name__}")
try:
    map_loc = None if DEVICE == 'cuda' else 'cpu'
    state_dict = torch.load(param_path, map_location=map_loc)
    if isinstance(state_dict, dict) and 'state_dict' in state_dict:
        state_dict = state_dict['state_dict']
    if all(isinstance(k, str) and k.startswith('module.') for k in state_dict.keys()):
        state_dict = {k[len('module.'):]: v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    if is_main_process():
        print(f"└─ 成功加载模型参数 (参数数量: {sum(p.numel() for p in model.parameters())})")
except Exception as exc:
    print(f"!! 模型加载错误: {exc}")
    raise SystemExit(1)


if is_main_process():
    print("\n[阶段 3/4] 配置评估参数")
    visible = os.environ.get('CUDA_VISIBLE_DEVICES', '(未设置)')
    gpu_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
    print(f"├─ 推理设备: {DEVICE}")
    print(f"├─ 可见GPU: {visible} | 实际可用数量: {gpu_count}")
    print(f"├─ 火点阈值: {TH_FIRE}")
    print(f"└─ 输出目录: {SAVE_DIR}")


if is_main_process():
    print("\n[阶段 4/4] 开始模型评估")
model.to('cuda' if torch.cuda.is_available() else 'cpu')
if dist.is_initialized():
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank] if torch.cuda.is_available() else None)
model.eval()

num_samples = 0

total_tp = 0
total_fp = 0
total_fn = 0
total_px = 0

with torch.inference_mode():
    test_iter = test_loader
    test_bar = tqdm(test_iter, desc=f"Test Size [{len(test_dataset)}]") if is_main_process() else test_iter
    for images, masks in test_bar:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        images = images.to(device, non_blocking=True)
        masks = (masks > 0).to(device, non_blocking=True)

        probs = model(images)
        preds = (probs > TH_FIRE)

        batch_size = images.size(0)
        num_samples += batch_size
        total_px += masks.numel()

        pred_np = preds.cpu().numpy()
        mask_np = masks.cpu().numpy()

        tp = np.logical_and(pred_np, mask_np).sum()
        fp = np.logical_and(pred_np, np.logical_not(mask_np)).sum()
        fn = np.logical_and(np.logical_not(pred_np), mask_np).sum()

        total_tp += int(tp)
        total_fp += int(fp)
        total_fn += int(fn)

result_filename = f"eval_{model_name}"

# DDP 聚合 totals
totals = torch.tensor([total_tp, total_fp, total_fn, num_samples, total_px], dtype=torch.long, device='cuda' if torch.cuda.is_available() else 'cpu')
if dist.is_initialized():
    dist.all_reduce(totals, op=dist.ReduceOp.SUM)
total_tp_g, total_fp_g, total_fn_g, num_samples_g, total_px_g = totals.tolist()

precision = total_tp_g / (total_tp_g + total_fp_g) if (total_tp_g + total_fp_g) > 0 else 0.0
recall = total_tp_g / (total_tp_g + total_fn_g) if (total_tp_g + total_fn_g) > 0 else 0.0
f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

# IoU Calculations
den_fire = (total_tp_g + total_fp_g + total_fn_g)
iou_fire = (total_tp_g / den_fire) if den_fire > 0 else 0.0
tn = max(0, total_px_g - total_tp_g - total_fp_g - total_fn_g)
den_bg = (tn + total_fp_g + total_fn_g)
iou_bg = (tn / den_bg) if den_bg > 0 else 0.0
miou = (iou_fire + iou_bg) / 2.0

report_path = os.path.join(SAVE_DIR, f"{result_filename}.txt")
if is_main_process():
    os.makedirs(SAVE_DIR, exist_ok=True)
    with open(report_path, "w") as f:
        f.write("=" * 50 + "\n")
        f.write("Baseline 火点检测模型评估报告\n")
        f.write("=" * 50 + "\n\n")

        f.write(f"设备: {DEVICE}\n")
        f.write(f"阈值: {TH_FIRE}\n")
        f.write(f"测试样本数: {num_samples_g}\n\n")

        f.write("=" * 50 + "\n")
        f.write("评估指标 (微平均)\n")
        f.write("=" * 50 + "\n")
        f.write(f"精确率: {precision:.4f}\n")
        f.write(f"召回率: {recall:.4f}\n")
        f.write(f"F1分数: {f1:.4f}\n\n")

        f.write("IoU 指标\n")
        f.write("-" * 50 + "\n")
        f.write(f"Fire IoU: {iou_fire:.4f}  (TP={total_tp_g}, FP={total_fp_g}, FN={total_fn_g})\n")
        f.write(f"Back IoU: {iou_bg:.4f}   (TN={tn}, FP={total_fp_g}, FN={total_fn_g})\n")
        f.write(f"mIoU: {miou:.4f}\n")

    print("\n" + "=" * 50)
    print("Baseline 火灾检测模型评估结果:")
    print(f" - 精确率: {precision:.4f}")
    print(f" - 召回率: {recall:.4f}")
    print(f" - F1分数: {f1:.4f}")
    print(f" - Fire IoU: {iou_fire:.4f}")
    print(f" - Back IoU: {iou_bg:.4f}")
    print(f" - mIoU: {miou:.4f}")
    print(f"\n评估结果已保存至: {SAVE_DIR}")
    print(f"- 文本报告: {report_path}")
    print("=" * 50 + "\n")


# 抽样可视化
if is_main_process():
    sample_loader = DataLoader(
        test_loader.dataset,
        batch_size=5,
        sampler=RandomSampler(test_loader.dataset),
        num_workers=test_loader.num_workers,
        pin_memory=pin_mem,
    )

    images, true_masks = next(iter(sample_loader))
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    images = images.to(device, non_blocking=True)
    true_masks = true_masks.to(device, non_blocking=True)

    with torch.inference_mode():
        pred_probs = model(images)
        pred_masks = (pred_probs > TH_FIRE).float()

    rows = images.size(0)
    cols = images.size(1) + 2
    fig, axes = plt.subplots(rows, cols, figsize=(12, 4 * rows))

    for i in range(rows):
        sample_img = images[i].cpu().numpy()
        sample_gt = true_masks[i].cpu().numpy().squeeze()
        sample_pred = pred_masks[i].cpu().numpy().squeeze()

        for band in range(images.shape[1]):
            axes[i, band].imshow(sample_img[band], cmap="gray")
            axes[i, band].set_title(f"C{band + 1}", fontsize=10)
            axes[i, band].axis("off")

        axes[i, -2].imshow(sample_gt, cmap="gray")
        axes[i, -2].set_title(f"GT | {np.sum(sample_gt):.0f} px", fontsize=10)
        axes[i, -2].axis("off")

        axes[i, -1].imshow(sample_pred, cmap="gray")
        axes[i, -1].set_title(f"Pred | {np.sum(sample_pred):.0f} px", fontsize=10)
        axes[i, -1].axis("off")

    plt.tight_layout()
    vis_path = os.path.join(SAVE_DIR, f"{model_name}_prediction.png")
    plt.savefig(vis_path, dpi=300, bbox_inches="tight")

    print(f"\n可视化样例已保存: {vis_path}")
    print("\n========== Baseline 评估流程结束 ==========")

# 清理分布式
cleanup_distributed()