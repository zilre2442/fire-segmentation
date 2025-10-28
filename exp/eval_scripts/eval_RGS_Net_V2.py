import os
import time
import argparse
from typing import Optional

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, RandomSampler, DistributedSampler
import matplotlib.pyplot as plt
from tqdm import tqdm

from dataset import LandsatFireDataset
from models.RGS_Net_V2 import RGSNetV2


def _is_preinit_main() -> bool:
    return os.environ.get("LOCAL_RANK", "0") == "0"


DATA_ROOT = "data/full"
ALGORITHM = "voting"
SAVE_DIR = "output/RGS_Net_V2/voting_YYYYMMDDHHMM"  # 运行时建议通过 --save-dir 指定
TH_FIRE = 0.25
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
USE_FUSION = True
TAU = 1.0


def parse_args():
    parser = argparse.ArgumentParser(description="Distributed evaluation for RGS_Net_V2")
    parser.add_argument("--dist", action="store_true", help="启用分布式评估 (检测 LOCAL_RANK 也会自动启用)")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--bands", type=int, nargs=3, default=(7, 6, 5))
    parser.add_argument("--save-dir", type=str, default=SAVE_DIR)
    parser.add_argument("--tau", type=float, default=1.0)
    parser.add_argument("--use-fusion", action="store_true", default=USE_FUSION)
    parser.add_argument("--algo", type=str, default=ALGORITHM)
    parser.add_argument("--model-name", type=str, default="model_best")
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
        try:
            dist.barrier()
        except Exception:
            pass
        try:
            dist.destroy_process_group()
        except Exception:
            pass


def get_local_rank_default() -> int:
    return int(os.environ.get("LOCAL_RANK", "0"))


def is_main_process() -> bool:
    return (not dist.is_initialized()) or dist.get_rank() == 0


if _is_preinit_main():
    print(f"========== RGS_Net_V2 评估启动 ==========")
    print(f"当前时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"计算设备: {'GPU可用' if torch.cuda.is_available() else '仅限CPU'}")


args = parse_args()
ALGORITHM = args.algo
SAVE_DIR = args.save_dir
TAU = args.tau
USE_FUSION = args.use_fusion
model_name = args.model_name

test_data_csv = os.path.join(DATA_ROOT, f"{ALGORITHM}_test.csv")
if _is_preinit_main():
    print("\n[阶段 1/4] 准备测试数据")
    print(f"├─ 算法标签: {ALGORITHM}")
    print(f"├─ 测试集CSV: {test_data_csv}")

test_dataset = LandsatFireDataset(test_data_csv, bands=tuple(args.bands))
if _is_preinit_main():
    print(f"├─ 测试集样本数: {len(test_dataset)}")


rank = setup_distributed()
local_rank = get_local_rank_default() if is_dist_env() else 0
if DEVICE == 'cuda' and torch.cuda.is_available():
    torch.cuda.set_device(local_rank)

sampler = None
if dist.is_initialized():
    sampler = DistributedSampler(test_dataset, shuffle=False, drop_last=False)

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


# 加载模型
if is_main_process():
    print("\n[阶段 2/4] 加载预训练模型")
param_path = os.path.join(SAVE_DIR, f"{model_name}.pth")
if is_main_process():
    print(f"├─ 参数路径: {param_path}")

model = RGSNetV2(n_channels=3, n_filters=16, tau=TAU)
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
except Exception as e:
    print(f"!! 模型加载错误: {str(e)}")
    cleanup_distributed()
    raise SystemExit(1)


# 评估配置
if is_main_process():
    print("\n[阶段 3/4] 配置评估参数")
    visible = os.environ.get('CUDA_VISIBLE_DEVICES', '(未设置)')
    gpu_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
    print(f"├─ 推理设备: {DEVICE}")
    print(f"├─ 可见GPU: {visible} | 实际可用数量: {gpu_count}")
    print(f"├─ 火点阈值: {TH_FIRE}")
    print(f"└─ 输出目录: {SAVE_DIR}")


# 执行评估
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
detailed_results = []

with torch.inference_mode():
    test_iter = test_loader
    test_bar = tqdm(test_iter, desc=f"Test Size [{len(test_dataset)}]") if is_main_process() else test_iter
    for batch_idx, (images, masks) in enumerate(test_bar):
        images = images.to('cuda' if torch.cuda.is_available() else 'cpu', non_blocking=True)
        masks = masks.to('cuda' if torch.cuda.is_available() else 'cpu', non_blocking=True)

        outputs = model(images, fuse_outputs=USE_FUSION)
        probs = outputs.fused_probs if USE_FUSION else outputs.seg_probs
        preds = (probs > TH_FIRE)

        batch_size = images.size(0)
        num_samples += batch_size

        for i in range(batch_size):
            pred = preds[i].cpu().numpy()
            mask = masks[i].cpu().numpy()

            tp = np.logical_and(pred, mask).sum()
            fp = np.logical_and(pred, np.logical_not(mask)).sum()
            fn = np.logical_and(np.logical_not(pred), mask).sum()

            total_tp += tp
            total_fp += fp
            total_fn += fn

            precision = (tp / (tp + fp)) if (tp + fp) > 0 else 1
            recall = (tp / (tp + fn)) if (tp + fn) > 0 else 1
            f1 = (2 * (precision * recall) / (precision + recall)) if (precision + recall) > 0 else 0

            detailed_results.append({
                'sample_id': batch_idx * batch_size + i,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'tp': tp,
                'fp': fp,
                'fn': fn
            })


# 计算微平均并写出
result_filename = f"eval_{model_name}"
metrics = {}

totals = torch.tensor([total_tp, total_fp, total_fn, num_samples], dtype=torch.long, device='cuda' if torch.cuda.is_available() else 'cpu')
if dist.is_initialized():
    dist.all_reduce(totals, op=dist.ReduceOp.SUM)
total_tp_g, total_fp_g, total_fn_g, num_samples_g = totals.tolist()

precision_micro = total_tp_g / (total_tp_g + total_fp_g) if (total_tp_g + total_fp_g) > 0 else 0.0
recall_micro = total_tp_g / (total_tp_g + total_fn_g) if (total_tp_g + total_fn_g) > 0 else 0.0
f1_micro = 2 * (precision_micro * recall_micro) / (precision_micro + recall_micro) if (precision_micro + recall_micro) > 0 else 0.0

metrics.update({'precision_micro': precision_micro, 'recall_micro': recall_micro, 'f1_micro': f1_micro})

report_path = os.path.join(SAVE_DIR, f"{result_filename}.txt")
plot_path = os.path.join(SAVE_DIR, f"{result_filename}.png")
if is_main_process():
    os.makedirs(SAVE_DIR, exist_ok=True)
    with open(report_path, 'w') as f:
        f.write("="*50 + "\n")
        f.write("RGS_Net_V2 火点检测模型评估报告\n")
        f.write("="*50 + "\n\n")
        f.write(f"设备: {DEVICE}\n")
        f.write(f"阈值: {TH_FIRE}\n")
        f.write(f"测试样本数: {num_samples_g}\n\n")
        f.write("="*50 + "\n")
        f.write("评估指标 (微平均)\n")
        f.write("="*50 + "\n")
        f.write(f"精确率: {metrics['precision_micro']:.4f}\n")
        f.write(f"召回率: {metrics['recall_micro']:.4f}\n")
        f.write(f"F1分数: {metrics['f1_micro']:.4f}\n\n")
        if not dist.is_initialized():
            f.write("="*50 + "\n")
            f.write("详细指标分布\n")
            f.write("="*50 + "\n")
            f.write(f"- 精确率范围: {min(r['precision'] for r in detailed_results):.4f} ~ {max(r['precision'] for r in detailed_results):.4f}\n")
            f.write(f"- 召回率范围: {min(r['recall'] for r in detailed_results):.4f} ~ {max(r['recall'] for r in detailed_results):.4f}\n")
            f.write(f"- F1分数范围: {min(r['f1_score'] for r in detailed_results):.4f} ~ {max(r['f1_score'] for r in detailed_results):.4f}\n")

    # 绘制图表
    if not dist.is_initialized():
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        precisions = [r['precision'] for r in detailed_results]
        recalls = [r['recall'] for r in detailed_results]
        plt.scatter(recalls, precisions, alpha=0.5)
        plt.title('Precision-Recall')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.subplot(1, 2, 2)
        data_to_plot = [precisions, recalls, [r['f1_score'] for r in detailed_results]]
        plt.boxplot(data_to_plot, labels=['Precision', 'Recall', 'F1'])
        plt.title('Metrics Distribution')
        plt.tight_layout()
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.figure(figsize=(6, 4))
        names = ['Precision', 'Recall', 'F1']
        vals = [metrics['precision_micro'], metrics['recall_micro'], metrics['f1_micro']]
        plt.bar(names, vals, color=['#4C72B0', '#55A868', '#C44E52'])
        plt.ylim(0, 1)
        for i, v in enumerate(vals):
            plt.text(i, v + 0.02, f"{v:.3f}", ha='center')
        plt.title('Micro Metrics (Distributed)')
        plt.tight_layout()
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()

if is_main_process():
    print("\n" + "="*50)
    print("RGS_Net_V2 火点检测模型评估结果:")
    print(f" - 精确率: {metrics['precision_micro']:.4f}")
    print(f" - 召回率: {metrics['recall_micro']:.4f}")
    print(f" - F1分数: {metrics['f1_micro']:.4f}")
    print(f"\n评估结果已保存至: {SAVE_DIR}")
    print(f"- 文本报告: {report_path}")
    print(f"- 可视化图表: {plot_path}")
    print("="*50 + "\n")


# 样本可视化（仅主进程）
if is_main_process():
    temp_loader = DataLoader(
        test_loader.dataset,
        batch_size=5,
        sampler=RandomSampler(test_loader.dataset),
        num_workers=test_loader.num_workers,
        pin_memory=pin_mem,
    )

    images, true_masks = next(iter(temp_loader))
    images = images.to('cuda' if torch.cuda.is_available() else 'cpu', non_blocking=True)
    true_masks = true_masks.to('cuda' if torch.cuda.is_available() else 'cpu', non_blocking=True)

    with torch.inference_mode():
        pred_outputs = model(images, fuse_outputs=USE_FUSION)
        pred_probs = pred_outputs.fused_probs if USE_FUSION else pred_outputs.seg_probs
        pred_masks = (pred_probs > TH_FIRE).float()

    fig, axes = plt.subplots(5, images.shape[1] + 2, figsize=(12, 5 * 5))
    for i in range(5):
        true_mask = true_masks[i].cpu().numpy().squeeze()
        pred_mask = pred_masks[i].cpu().numpy().squeeze()
        image = images[i].cpu().numpy()

        for band in range(images.shape[1]):
            axes[i, band].imshow(image[band], cmap='gray')
            axes[i, band].set_title(f"C{band + 1}", fontsize=10)
            axes[i, band].axis('off')
        axes[i, -2].imshow(true_mask, cmap='gray')
        axes[i, -2].set_title(f"GT | {np.sum(true_mask)} pixels", fontsize=10)
        axes[i, -2].axis('off')
        axes[i, -1].imshow(pred_mask, cmap='gray')
        axes[i, -1].set_title(f"pred | {np.sum(pred_mask)} pixels", fontsize=10)
        axes[i, -1].axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, model_name + "_prediction.png"), dpi=300, bbox_inches='tight')
    print(f"\n评估完成! 结果已保存至 {SAVE_DIR}")


cleanup_distributed()

# CUDA_VISIBLE_DEVICES=3,4,5 torchrun --nproc_per_node=3 exp/eval_scripts/eval_RGS_Net_V2.py --dist --save-dir output/RGS_Net_V2/voting_202510230911 --tau 1
