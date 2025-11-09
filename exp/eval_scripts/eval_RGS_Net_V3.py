import os
import time
import argparse
from typing import Optional, Dict, List, Tuple

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
DATA_ROOT = "data/splits_land8fire"
ALGORITHM = "Land8Fire"
SAVE_DIR = "output/RGS_Net_V3/Land8Fire_202511071056"
BANDS = (7, 6, 5)

SAMPLE_DIR = os.path.join(SAVE_DIR, "pred_samples")

# 使用示例:
# 1) 分布式评估 (torchrun 自动设置分布式环境变量):
#
#    CUDA_VISIBLE_DEVICES=1 torchrun --nproc_per_node=1 exp/eval_scripts/eval_RGS_Net_V3.py --dist --batch-size 64 --save-dir output/RGS_Net_V3/voting_date
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
    # 样例类别阈值: (very_few, few, many)，单位=像素数，类别定义:
    # none: =0, very_few: [1, thr1], few: (thr1, thr2], many: (thr2, thr3], very_many: >thr3
    parser.add_argument(
        "--cat-thresholds",
        type=int,
        nargs=3,
        default=(10, 100, 1000),
        metavar=("THR1", "THR2", "THR3"),
        help="火点像素数分类阈值: very_few<=THR1<=few<=THR2<=many<=THR3<very_many",
    )
    parser.add_argument("--samples-per-cat", type=int, default=5, help="每个类别选择的样例数量")
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
eval_model = model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model

num_samples = 0

total_tp = 0
total_fp = 0
total_fn = 0
total_px = 0

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
        total_px += masks_b.numel()

totals = torch.tensor(
    [total_tp, total_fp, total_fn, num_samples, total_px],
    dtype=torch.long,
    device='cuda' if torch.cuda.is_available() else 'cpu',
)
if dist.is_initialized():
    dist.all_reduce(totals, op=dist.ReduceOp.SUM)
total_tp_g, total_fp_g, total_fn_g, num_samples_g, total_px_g = totals.tolist()

precision = total_tp_g / (total_tp_g + total_fp_g) if (total_tp_g + total_fp_g) > 0 else 0.0
recall = total_tp_g / (total_tp_g + total_fn_g) if (total_tp_g + total_fn_g) > 0 else 0.0
f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

# IoU 指标（fire类、background类、mIoU）
den_fire = (total_tp_g + total_fp_g + total_fn_g)
iou_fire = (total_tp_g / den_fire) if den_fire > 0 else 0.0
tn = max(0, total_px_g - total_tp_g - total_fp_g - total_fn_g)
den_bg = (tn + total_fp_g + total_fn_g)
iou_bg = (tn / den_bg) if den_bg > 0 else 0.0
miou = (iou_fire + iou_bg) / 2.0

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
        f.write(f"F1分数: {f1:.4f}\n\n")
        f.write("IoU 指标\n")
        f.write("-" * 50 + "\n")
        f.write(f"Fire IoU: {iou_fire:.4f}  (TP={total_tp_g}, FP={total_fp_g}, FN={total_fn_g})\n")
        f.write(f"Back IoU: {iou_bg:.4f}   (TN={tn}, FP={total_fp_g}, FN={total_fn_g})\n")
        f.write(f"mIoU: {miou:.4f}\n")

if is_main_process():
    print("\n" + "=" * 50)
    print("RGS-Net V3 火点检测模型评估结果:")
    print(f" - 精确率: {precision:.4f}")
    print(f" - 召回率: {recall:.4f}")
    print(f" - F1分数: {f1:.4f}")
    print(f" - Fire IoU: {iou_fire:.4f}")
    print(f" - Back IoU: {iou_bg:.4f}")
    print(f" - mIoU: {miou:.4f}")
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
        Image.fromarray(rgb_img).save(os.path.join(SAMPLE_DIR, f"sample_{idx:02d}_rgb.png"))

        gt_img = (gts_np[idx][0] * 255).astype(np.uint8)
        Image.fromarray(gt_img).save(os.path.join(SAMPLE_DIR, f"sample_{idx:02d}_gt.png"))

        seg_img = (seg_mask[idx][0] * 255).astype(np.uint8)
        Image.fromarray(seg_img).save(os.path.join(SAMPLE_DIR, f"sample_{idx:02d}_seg.png"))

        rec_img = (rec_mask[idx][0] * 255).astype(np.uint8)
        Image.fromarray(rec_img).save(os.path.join(SAMPLE_DIR, f"sample_{idx:02d}_rec.png"))

        fused_img = (fused_mask[idx][0] * 255).astype(np.uint8)
        Image.fromarray(fused_img).save(os.path.join(SAMPLE_DIR, f"sample_{idx:02d}_fused.png"))

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

    # =============================
    # 基于GT火点像素数的类别选样与标注
    # =============================
    print("\n[附加输出] 基于火点数的类别样例选择与标注")
    thr1, thr2, thr3 = args.cat_thresholds
    samples_per_cat = max(1, args.samples_per_cat)
    CATEGORY_DIR = os.path.join(SAVE_DIR, "category_samples")
    os.makedirs(CATEGORY_DIR, exist_ok=True)

    def categorize(count: int) -> str:
        if count == 0:
            return "none"
        if count <= thr1:
            return "very_few"
        if count <= thr2:
            return "few"
        if count <= thr3:
            return "many"
        return "very_many"

    # 第二个专用 DataLoader（仅主进程，不使用分布式采样），遍历全量测试集
    full_loader = DataLoader(
        test_loader.dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=test_loader.num_workers,
        pin_memory=pin_mem,
    )

    # 为每个类别收集若干样例（保存图像、GT、预测和计数）
    selected: Dict[str, List[Tuple[np.ndarray, np.ndarray, np.ndarray, int, int]]] = {
        "none": [], "very_few": [], "few": [], "many": [], "very_many": []
    }

    with torch.inference_mode():
        for images_b, masks_b in tqdm(full_loader, desc="挑选样例"):
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            images_b = images_b.to(device, non_blocking=True)
            masks_b = (masks_b > 0).to(device, non_blocking=True)

            # 使用eval_model以避免在分布式环境下对其它rank进行同步
            outputs_b = eval_model(images_b, binarize=True)
            preds_b = outputs_b.binary_mask

            # 逐样本处理
            for i in range(images_b.size(0)):
                gt = masks_b[i]
                pr = preds_b[i]
                gt_cnt = int(gt.sum().item())
                pr_cnt = int(pr.sum().item())
                cat = categorize(gt_cnt)

                if len(selected[cat]) < samples_per_cat:
                    # 组装可视化所需的 numpy
                    img_np = images_b[i].detach().cpu().numpy()
                    gt_np = gt.detach().cpu().numpy()
                    pr_np = pr.detach().cpu().numpy()
                    selected[cat].append((img_np, gt_np, pr_np, gt_cnt, pr_cnt))

            # 若所有类别已满则提前结束
            if all(len(v) >= samples_per_cat for v in selected.values()):
                break

    # 输出每个类别的对比图：每行一个样本，5列=Input(RGB), GT(计数), Pred(计数)
    # 为了简洁，这里使用3列：Input/GT/Pred，并在标题中标注计数
    def plot_category(cat_name: str, items: List[Tuple[np.ndarray, np.ndarray, np.ndarray, int, int]]):
        if not items:
            return
        rows = len(items)
        cols = 3
        fig, axes = plt.subplots(rows, cols, figsize=(12, 4 * rows))
        if rows == 1:
            axes = np.expand_dims(axes, axis=0)
        for r, (img_np, gt_np, pr_np, gt_cnt, pr_cnt) in enumerate(items):
            # 复合 RGB：取前3个通道
            rgb = img_np[:3]
            rgb_img = (rgb * 255).clip(0, 255).astype(np.uint8)
            rgb_img = np.transpose(rgb_img, (1, 2, 0))

            axes[r, 0].imshow(rgb_img)
            axes[r, 0].set_title(f"Input (RGB)")
            axes[r, 0].axis("off")

            gt_show = gt_np[0] if gt_np.ndim == 3 else gt_np
            axes[r, 1].imshow(gt_show, cmap="gray")
            axes[r, 1].set_title(f"GT | fire px: {gt_cnt}")
            axes[r, 1].axis("off")

            pr_show = pr_np[0] if pr_np.ndim == 3 else pr_np
            axes[r, 2].imshow(pr_show, cmap="gray")
            axes[r, 2].set_title(f"Pred | fire px: {pr_cnt}")
            axes[r, 2].axis("off")

        plt.tight_layout()
        out_path = os.path.join(CATEGORY_DIR, f"cat_{cat_name}_thr_{thr1}-{thr2}-{thr3}_k{samples_per_cat}.png")
        plt.savefig(out_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"类别[{cat_name}]样例对比已保存: {out_path}")

    for cat_name in ["very_many", "many", "few", "very_few", "none"]:
        plot_category(cat_name, selected[cat_name])

cleanup_distributed()
