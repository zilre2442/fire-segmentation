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

# ---- 预初始化阶段的主进程判定（在 DDP 尚未 init 时使用） ----
def _is_preinit_main() -> bool:
    return os.environ.get("LOCAL_RANK", "0") == "0"


if _is_preinit_main():
    print("========== Baseline 样本级准确率评估启动 ==========")
    print(f"当前时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"计算设备: {'GPU可用' if torch.cuda.is_available() else '仅限CPU'}")

DATA_ROOT = "data/splits_activefire"
ALGORITHM = "voting"
SAVE_DIR = "output/baseline/voting_202512182150"  # 指向已训练模型的目录
TH_FIRE = 0.5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BANDS = (7, 6, 2)

# --------- DDP/CLI 实用函数 ---------
def parse_args():
    parser = argparse.ArgumentParser(description="Distributed evaluation for Baseline UNet (Sample Accuracy)")
    parser.add_argument("--dist", action="store_true", help="启用分布式评估 (检测到 LOCAL_RANK 也会自动启用)")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--bands", type=int, nargs=3, default=BANDS, help="选择用于评估的三个波段索引")
    parser.add_argument("--save-dir", type=str, default=SAVE_DIR)
    parser.add_argument("--algo", type=str, default=ALGORITHM)
    parser.add_argument("--threshold", type=float, default=TH_FIRE)
    parser.add_argument("--data-root", type=str, default=DATA_ROOT)
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
DATA_ROOT = args.data_root
TH_FIRE = args.threshold
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
    print("\n[阶段 4/4] 开始样本级准确率评估")
model.to('cuda' if torch.cuda.is_available() else 'cpu')
if dist.is_initialized():
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank] if torch.cuda.is_available() else None)
model.eval()

num_samples = 0
total_correct = 0

# 统计混淆矩阵 (样本级)
# TP: GT有火，Pred有火
# TN: GT无火，Pred无火
# FP: GT无火，Pred有火
# FN: GT有火，Pred无火
sample_tp = 0
sample_tn = 0
sample_fp = 0
sample_fn = 0

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
        
        # 判断每个样本是否有火 (Max pooling)
        # GT
        gt_has_fire = masks.view(batch_size, -1).max(dim=1).values > 0
        # Pred
        pred_has_fire = preds.view(batch_size, -1).max(dim=1).values > 0
        
        # 计算正确预测的样本数
        correct_mask = (gt_has_fire == pred_has_fire)
        batch_correct = correct_mask.sum().item()
        
        total_correct += batch_correct
        num_samples += batch_size

        # 详细统计
        sample_tp += ((gt_has_fire == 1) & (pred_has_fire == 1)).sum().item()
        sample_tn += ((gt_has_fire == 0) & (pred_has_fire == 0)).sum().item()
        sample_fp += ((gt_has_fire == 0) & (pred_has_fire == 1)).sum().item()
        sample_fn += ((gt_has_fire == 1) & (pred_has_fire == 0)).sum().item()

result_filename = f"eval_{model_name}_sample_acc"

# DDP 聚合 totals
totals = torch.tensor([total_correct, num_samples, sample_tp, sample_tn, sample_fp, sample_fn], dtype=torch.long, device='cuda' if torch.cuda.is_available() else 'cpu')
if dist.is_initialized():
    dist.all_reduce(totals, op=dist.ReduceOp.SUM)
total_correct_g, num_samples_g, sample_tp_g, sample_tn_g, sample_fp_g, sample_fn_g = totals.tolist()

accuracy = total_correct_g / num_samples_g if num_samples_g > 0 else 0.0

if is_main_process():
    print("\n" + "=" * 50)
    print("Baseline 样本级评估结果:")
    print(f" - 样本准确率: {accuracy:.4f}")
    print(f" - 正确样本数: {total_correct_g}")
    print(f" - 总样本数:   {num_samples_g}")
    print("-" * 30)
    print(f" - TP: {sample_tp_g}")
    print(f" - TN: {sample_tn_g}")
    print(f" - FP: {sample_fp_g}")
    print(f" - FN: {sample_fn_g}")
    print("=" * 50 + "\n")


# 清理分布式
cleanup_distributed()
