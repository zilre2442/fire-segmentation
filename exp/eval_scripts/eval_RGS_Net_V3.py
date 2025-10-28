import os
import time
import argparse
from typing import Optional

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, RandomSampler, DistributedSampler

from dataset import LandsatFireDataset
from models.RGS_Net_V3 import RGSNetV3


def _is_preinit_main() -> bool:
    return os.environ.get("LOCAL_RANK", "0") == "0"


if _is_preinit_main():
    print("========== RGS-Net V3 火灾检测评估启动 ==========")
    print(f"当前时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"计算设备: {'GPU可用' if torch.cuda.is_available() else '仅限CPU'}")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DATA_ROOT = "data/full"
ALGORITHM = "voting"
SAVE_DIR = "output/RGS_Net_V3/voting_date"
BANDS = (7, 6, 5)

SAMPLE_DIR = os.path.join(SAVE_DIR, "pred_samples")

# 使用示例:
# 1) 分布式评估 (torchrun 自动设置分布式环境变量):
#
#    CUDA_VISIBLE_DEVICES=3,5 torchrun --nproc_per_node=2 exp/eval_scripts/eval_RGS_Net_V3.py --dist --batch-size 64 --save-dir output/RGS_Net_V3/voting_date
#
# 2) 单卡评估（手动设置环境变量）:
#
#    RANK=0 WORLD_SIZE=1 LOCAL_RANK=0 python3 exp/eval_scripts/eval_RGS_Net_V3.py --batch-size 64 --save-dir output/RGS_Net_V3/voting_date
#
# 3) 注意点:
#    - 脚本默认寻找 `SAVE_DIR/weights/model_best.pth`，可通过 --save-dir 指定不同目录。
#    - 评估脚本会保存可视化样例到保存目录下的 pred_samples 子目录。



def parse_args():
    parser = argparse.ArgumentParser(description="Distributed evaluation for RGSNetV3")
    parser.add_argument("--dist", action="store_true", help="启用分布式评估")
    parser.add_argument("--batch-size", type=int, default=64)
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
    print(f"└─ 测试集样本数: {len(test_dataset)}")

rank = setup_distributed()
local_rank = get_local_rank_default() if is_dist_env() else 0
if torch.cuda.is_available():
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
    print(f"测试批次数: {len(test_loader)}")

if is_main_process():
    print("\n[阶段 2/4] 加载预训练模型")
model_name = "model_best"
param_path = os.path.join(SAVE_DIR, "weights", f"{model_name}.pth")
if is_main_process():
    print(f"├─ 参数路径: {param_path}")

model = RGSNetV3(n_channels=len(BANDS), n_filters=64)
if is_main_process():
    print(f"└─ 网络架构: {model.__class__.__name__}")
try:
    map_loc = None if torch.cuda.is_available() else "cpu"
    state_dict = torch.load(param_path, map_location=map_loc)
    if isinstance(state_dict, dict) and "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]
    if all(isinstance(k, str) and k.startswith("module.") for k in state_dict.keys()):
        state_dict = {k[len("module."):]: v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
except Exception as exc:
    print(f"!! 模型加载错误: {exc}")
    raise SystemExit(1)

if is_main_process():
    print("\n[阶段 3/4] 配置评估参数")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "(未设置)")
    gpu_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
    print(f"├─ 推理设备: {DEVICE}")
    print(f"├─ 可见GPU: {visible} | 实际可用数量: {gpu_count}")
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

with torch.inference_mode():
    iterator = tqdm(test_loader, desc=f"Test Size [{len(test_dataset)}]") if is_main_process() else test_loader
    for images, masks in iterator:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        images = images.to(device, non_blocking=True)
        masks = (masks > 0).to(device, non_blocking=True)

        outputs = model(images, binarize=True)
        preds = outputs.binary_mask.bool()

        batch_size = images.size(0)
        num_samples += batch_size

        masks_b = masks.bool()
        total_tp += torch.logical_and(preds, masks_b).sum().item()
        total_fp += torch.logical_and(preds, ~masks_b).sum().item()
        total_fn += torch.logical_and(~preds, masks_b).sum().item()

totals = torch.tensor(
    [total_tp, total_fp, total_fn, num_samples],
    dtype=torch.long,
    device='cuda' if torch.cuda.is_available() else 'cpu',
)
if dist.is_initialized():
    dist.all_reduce(totals, op=dist.ReduceOp.SUM)
total_tp_g, total_fp_g, total_fn_g, num_samples_g = totals.tolist()

precision = total_tp_g / (total_tp_g + total_fp_g) if (total_tp_g + total_fp_g) > 0 else 0.0
recall = total_tp_g / (total_tp_g + total_fn_g) if (total_tp_g + total_fn_g) > 0 else 0.0
f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

result_filename = f"eval_{model_name}"
report_path = os.path.join(SAVE_DIR, f"{result_filename}.txt")
if is_main_process():
    os.makedirs(SAVE_DIR, exist_ok=True)
    with open(report_path, "w") as f:
        f.write("=" * 50 + "\n")
        f.write("RGS-Net V3 火点检测模型评估报告\n")
        f.write("=" * 50 + "\n\n")

        f.write(f"设备: {DEVICE}\n")
        f.write(f"测试样本数: {num_samples_g}\n\n")

        f.write("=" * 50 + "\n")
        f.write("评估指标 (微平均)\n")
        f.write("=" * 50 + "\n")
        f.write(f"精确率: {precision:.4f}\n")
        f.write(f"召回率: {recall:.4f}\n")
        f.write(f"F1分数: {f1:.4f}\n")

if is_main_process():
    print("\n" + "=" * 50)
    print("RGS-Net V3 火点检测模型评估结果:")
    print(f" - 精确率: {precision:.4f}")
    print(f" - 召回率: {recall:.4f}")
    print(f" - F1分数: {f1:.4f}")
    print(f"\n评估结果已保存至: {SAVE_DIR}")
    print(f"- 文本报告: {report_path}")
    print("=" * 50 + "\n")

if is_main_process():
    os.makedirs(SAMPLE_DIR, exist_ok=True)

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
    true_masks = (true_masks > 0).float().to(device, non_blocking=True)

    with torch.inference_mode():
        outputs = model(images, binarize=True)
        seg_mask = outputs.seg_mask.cpu().numpy()
        rec_mask = outputs.rec_mask.cpu().numpy()
        fused_mask = outputs.binary_mask.cpu().numpy()

    imgs_np = images.cpu().numpy()
    gts_np = true_masks.cpu().numpy()
    for idx in range(imgs_np.shape[0]):
        # 复合显示：将选择的3个波段组合为可视化图（便于肉眼观察）
        rgb = imgs_np[idx][:3]
        rgb_img = (rgb * 255).clip(0, 255).astype(np.uint8)
        rgb_img = np.transpose(rgb_img, (1, 2, 0))
        Image.fromarray(rgb_img, mode="RGB").save(os.path.join(SAMPLE_DIR, f"sample_{idx:02d}_rgb.png"))

        gt_img = (gts_np[idx][0] * 255).astype(np.uint8)
        Image.fromarray(gt_img, mode="L").save(os.path.join(SAMPLE_DIR, f"sample_{idx:02d}_gt.png"))

        seg_img = (seg_mask[idx][0] * 255).astype(np.uint8)
        Image.fromarray(seg_img, mode="L").save(os.path.join(SAMPLE_DIR, f"sample_{idx:02d}_seg.png"))

        rec_img = (rec_mask[idx][0] * 255).astype(np.uint8)
        Image.fromarray(rec_img, mode="L").save(os.path.join(SAMPLE_DIR, f"sample_{idx:02d}_rec.png"))

        fused_img = (fused_mask[idx][0] * 255).astype(np.uint8)
        Image.fromarray(fused_img, mode="L").save(os.path.join(SAMPLE_DIR, f"sample_{idx:02d}_fused.png"))

    # 可视化样例：每行 5 列（RGB复合、GT、Seg、Rec、Fused），不再逐波段展示
    rows = images.size(0)
    cols = 5
    fig, axes = plt.subplots(rows, cols, figsize=(16, 4 * rows))

    for i in range(rows):
        # 复合 RGB
        rgb = imgs_np[i][:3]
        rgb_img = (rgb * 255).clip(0, 255).astype(np.uint8)
        rgb_img = np.transpose(rgb_img, (1, 2, 0))

        sample_gt = gts_np[i][0]
        sample_seg = seg_mask[i][0]
        sample_rec = rec_mask[i][0]
        sample_fused = fused_mask[i][0]

        axes[i, 0].imshow(rgb_img)
        axes[i, 0].set_title("Input (RGB)", fontsize=10)
        axes[i, 0].axis("off")

        axes[i, 1].imshow(sample_gt, cmap="gray")
        axes[i, 1].set_title("GT", fontsize=10)
        axes[i, 1].axis("off")

        axes[i, 2].imshow(sample_seg, cmap="gray")
        axes[i, 2].set_title("Seg", fontsize=10)
        axes[i, 2].axis("off")

        axes[i, 3].imshow(sample_rec, cmap="gray")
        axes[i, 3].set_title("Rec", fontsize=10)
        axes[i, 3].axis("off")

        axes[i, 4].imshow(sample_fused, cmap="gray")
        axes[i, 4].set_title("Fused", fontsize=10)
        axes[i, 4].axis("off")

    plt.tight_layout()
    vis_path = os.path.join(SAVE_DIR, f"{model_name}_prediction.png")
    plt.savefig(vis_path, dpi=300, bbox_inches="tight")

    print(f"示例输出已保存至: {SAMPLE_DIR}")
    print(f"可视化样例已保存: {vis_path}")
    print("\n========== RGS-Net V3 评估流程结束 ==========")

cleanup_distributed()
