"""评估脚本: Dual-Sight Fire

- 支持分布式评估，自动聚合全局指标
- 默认从 `--save-dir/weights/model_best.pth` 加载参数；也支持 `--model-path`
- 支持选择输出类型进行阈值化评估: fused / seg（仅用于指标与主对比图）
- 指标：Precision / Recall / F1 / IoU(Fire/Back) / mIoU
- 保存样例：pred_samples (BANS/GT/Seg/Rec/Fused) 与整体可视化；附加按火点数量类别输出
- 保存样例：pred_samples (BANS/GT/Seg/Rec/Fused) 与整体可视化；可选附加按火点数量类别输出（启用 `--extra-output`）

示例：
CUDA_VISIBLE_DEVICES=6,7 torchrun --nproc_per_node=2 --master_port=65530 exp/eval_scripts/eval_DualSight_Fire.py --dist --data-root data/splits_activefire --algo voting --threshold 0.5 --save-dir output/DualSight_Fire/voting_202602032156 --output-type fused
CUDA_VISIBLE_DEVICES=7 python exp/eval_scripts/eval_DualSight_Fire.py --data-root data/splits_activefire --algo voting --threshold 0.5 --save-dir output/DualSight_Fire/voting_202512061415 --output-type seg
"""

from __future__ import annotations

import os
import sys
import argparse
import time
from typing import Tuple, List, Optional, Dict

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, RandomSampler, DistributedSampler
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from dataset import LandsatFireDataset
from models.DualSight_Fire import DualSight_Fire


def _is_preinit_main() -> bool:
    """在初始化分布式之前，也仅允许全局 rank 0 打印日志。"""
    return os.environ.get("RANK", "0") == "0"


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate Dual-Sight Fire (aligned with V3_1)")
    p.add_argument("--dist", action="store_true", help="启用分布式评估")
    p.add_argument("--data-root", type=str, default="data/splits_activefire")
    p.add_argument("--algo", type=str, default="voting")
    p.add_argument("--bands", type=int, nargs=3, default=(7, 6, 5))
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--model-path", type=str, default=None, help="模型参数路径（默认从save-dir推断）")
    p.add_argument("--save-dir", type=str, default="output/DualSight_Fire/voting_date", help="训练输出目录，用于推断权重路径和保存评估结果")
    p.add_argument("--output-type", type=str, choices=["fused", "seg"], default="fused")
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--fire-category", type=str, default=None, choices=["very_few", "few", "many", "very_many"], help="按火点数量选择子集")
    p.add_argument("--cat-thresholds", type=int, nargs=3, default=(10, 100, 1000), metavar=("THR1","THR2","THR3"))
    p.add_argument("--samples-per-cat", type=int, default=5, help="每个类别选择的样例数量")
    p.add_argument("--samples", type=int, default=5, help="主可视化中保存的样例数量")
    p.add_argument("--extra-output", action="store_true", help="启用附加输出：按GT火点像素数进行类别样例选择与标注")
    return p.parse_args()

def get_test_csv(root: str, algo: str, fire_category: str = None) -> str:
    if fire_category is None:
        return os.path.join(root, f"{algo}_test.csv")
    if algo == "Land8Fire":
        subdir = os.path.join(root, "by_fire_pixels")
    else:
        subdir = os.path.join(root, f"by_fire_pixels_{algo}")
    return os.path.join(subdir, "test", f"{algo}_test_{fire_category}.csv")


def is_dist_env() -> bool:
    return int(os.environ.get("WORLD_SIZE", "1")) > 1 or "LOCAL_RANK" in os.environ


def setup_distributed(backend: str = "nccl") -> Optional[int]:
    if not is_dist_env():
        return None
    if not dist.is_initialized():
        dist.init_process_group(backend=backend, init_method="env://")
    return dist.get_rank()


def cleanup_distributed() -> None:
    if dist.is_initialized():
        dist.destroy_process_group()


def get_local_rank_default() -> int:
    return int(os.environ.get("LOCAL_RANK", "0"))


def is_main_process() -> bool:
    return (not dist.is_initialized()) or dist.get_rank() == 0


def main():
    if _is_preinit_main():
        print("========== Dual-Sight Fire 火灾检测评估启动 ==========")
        print(f"当前时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"计算设备: {'GPU可用' if torch.cuda.is_available() else '仅限CPU'}")

    args = parse_args()
    ALGO = args.algo
    SAVE_DIR = args.save_dir
    BANDS = tuple(args.bands)

    # 测试数据集准备
    test_csv = get_test_csv(args.data_root, ALGO, args.fire_category)
    if _is_preinit_main():
        print("\n[阶段 1/4] 准备测试数据")
        print(f"├─ 算法标签: {ALGO}")
        print(f"├─ 火点类别: {args.fire_category if args.fire_category else '全量数据集'}")
        print(f"├─ 测试集CSV: {test_csv}")

    ds = LandsatFireDataset(test_csv, bands=BANDS)
    if _is_preinit_main():
        print(f"└─ 测试集样本数: {len(ds)}")

    setup_distributed()  # 确保分布式环境已初始化，仅 rank0 输出后续信息

    local_rank = get_local_rank_default() if is_dist_env() else 0
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)

    sampler = DistributedSampler(ds, shuffle=False, drop_last=False) if dist.is_initialized() else None
    pin_mem = torch.cuda.is_available()
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=(sampler is None),
        num_workers=args.num_workers,
        pin_memory=pin_mem,
        sampler=sampler,
    )
    if is_main_process():
        print(f"测试批次数: {len(loader)}")

    # 加载模型与权重
    if is_main_process():
        print("\n[阶段 2/4] 加载预训练模型")  # 权重路径推断：优先使用 --model-path，否则默认 save-dir/weights/model_best.pth
    model_name = "model_best"
    param_path = args.model_path or os.path.join(SAVE_DIR, "weights", f"{model_name}.pth")
    if is_main_process():
        print(f"├─ 参数路径: {param_path}")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = DualSight_Fire(n_channels=len(BANDS)).to(device)
    if is_main_process():
        print(f"└─ 网络架构: {model.__class__.__name__}")
        print(f"└─ 重建权重参数: {model.rec_weight}")

    try:
        map_loc = None if torch.cuda.is_available() else 'cpu'
        state_dict = torch.load(param_path, map_location=map_loc)
        if isinstance(state_dict, dict) and 'state_dict' in state_dict:
            state_dict = state_dict['state_dict']
        if all(isinstance(k, str) and k.startswith('module.') for k in state_dict.keys()):
            state_dict = {k[len('module.'):]: v for k, v in state_dict.items()}
        model.load_state_dict(state_dict)
    except Exception as exc:
        print(f"!! 模型加载错误: {exc}")
        cleanup_distributed()
        raise SystemExit(1)

    # 评估阶段
    if is_main_process():
        print("\n[阶段 3/4] 配置评估参数")
        visible = os.environ.get("CUDA_VISIBLE_DEVICES", "(未设置)")
        gpu_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
        print(f"├─ 推理设备: {device}")
        print(f"├─ 可见GPU: {visible} | 实际可用数量: {gpu_count}")
        print(f"└─ 输出目录: {SAVE_DIR}")
        print(f"└─ 重建权重参数: {model.rec_weight}")

    if is_main_process():
        print("\n[阶段 4/4] 开始模型评估")
    model.to(device)
    if dist.is_initialized():
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[local_rank] if torch.cuda.is_available() else None,
        )
    model.eval()
    eval_model = model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model

    total_tp = 0
    total_fp = 0
    total_fn = 0
    total_px = 0
    num_samples = 0

    with torch.inference_mode():
        iterator = loader if not is_main_process() else __import__('tqdm').tqdm(loader, desc=f"Test Size [{len(ds)}]")
        for images, masks in iterator:
            images = images.to(device, non_blocking=True)
            masks = (masks > 0).to(device, non_blocking=True)

            seg_logits, rec_logits, fused_logits = eval_model(images)
            if args.output_type == 'fused':
                chosen = fused_logits
            elif args.output_type == 'seg':
                chosen = seg_logits
            else:
                chosen = rec_logits

            preds = (torch.sigmoid(chosen) >= args.threshold).bool()
            gt = masks.bool()

            batch_size = images.size(0)
            num_samples += batch_size
            total_tp += torch.logical_and(preds, gt).sum().item()
            total_fp += torch.logical_and(preds, ~gt).sum().item()
            total_fn += torch.logical_and(~preds, gt).sum().item()
            total_px += gt.numel()

    totals = torch.tensor(
        [total_tp, total_fp, total_fn, num_samples, total_px],
        dtype=torch.long,
        device=device,
    )
    if dist.is_initialized():
        dist.all_reduce(totals, op=dist.ReduceOp.SUM)
    total_tp_g, total_fp_g, total_fn_g, num_samples_g, total_px_g = totals.tolist()

    precision = total_tp_g / (total_tp_g + total_fp_g) if (total_tp_g + total_fp_g) > 0 else 0.0
    recall = total_tp_g / (total_tp_g + total_fn_g) if (total_tp_g + total_fn_g) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    den_fire = (total_tp_g + total_fp_g + total_fn_g)
    iou_fire = (total_tp_g / den_fire) if den_fire > 0 else 0.0
    tn = max(0, total_px_g - total_tp_g - total_fp_g - total_fn_g)
    den_bg = (tn + total_fp_g + total_fn_g)
    iou_bg = (tn / den_bg) if den_bg > 0 else 0.0
    miou = (iou_fire + iou_bg) / 2.0

    # 写入评估报告
    os.makedirs(SAVE_DIR, exist_ok=True)
    result_filename = f"eval_{'model_best' if args.model_path is None else os.path.splitext(os.path.basename(args.model_path))[0]}"
    report_path = os.path.join(SAVE_DIR, f"{result_filename}.txt")
    if is_main_process():
        with open(report_path, "w") as f:
            f.write("=" * 50 + "\n")
            f.write("Dual-Sight Fire 火点检测模型评估报告\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"设备: {device}\n")
            f.write(f"测试样本数: {num_samples_g}\n")
            f.write(f"重建权重参数: {eval_model.rec_weight.item()}\n\n")
            f.write("=" * 50 + "\n")
            f.write("评估指标\n")
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
        print("Dual-Sight Fire 火点检测模型评估结果:")
        print(f" - 精确率: {precision:.4f}")
        print(f" - 召回率: {recall:.4f}")
        print(f" - F1分数: {f1:.4f}")
        print(f" - Fire IoU: {iou_fire:.4f}")
        print(f" - Back IoU: {iou_bg:.4f}")
        print(f" - mIoU: {miou:.4f}")
        print(f" - 重建权重参数: {eval_model.rec_weight.item()}")
        print(f"\n评估结果已保存至: {SAVE_DIR}")
        print(f"- 文本报告: {report_path}")
        print("=" * 50 + "\n")

    # pred_samples: 保存BANS/GT/Seg/Rec/Fused（与V3_1对齐）
    if is_main_process():
        SAMPLE_DIR = os.path.join(SAVE_DIR, "pred_samples")
        os.makedirs(SAMPLE_DIR, exist_ok=True)

        sample_loader = DataLoader(
            loader.dataset,
            batch_size=max(1, args.samples),
            sampler=RandomSampler(loader.dataset),
            num_workers=loader.num_workers,
            pin_memory=pin_mem,
        )

        images, true_masks = next(iter(sample_loader))
        images = images.to(device, non_blocking=True)
        true_masks = (true_masks > 0).float().to(device, non_blocking=True)

        with torch.inference_mode():
            seg_logits, rec_logits, fused_logits = eval_model(images)
            seg_mask = (torch.sigmoid(seg_logits) >= args.threshold).float().cpu().numpy()
            rec_mask = (torch.sigmoid(rec_logits) >= args.threshold).float().cpu().numpy()
            fused_mask = (torch.sigmoid(fused_logits) >= args.threshold).float().cpu().numpy()

        imgs_np = images.cpu().numpy()
        gts_np = true_masks.cpu().numpy()
        for idx in range(imgs_np.shape[0]):
            rgb = imgs_np[idx][:3]
            rgb_img = (rgb * 255).clip(0, 255).astype(np.uint8)
            rgb_img = np.transpose(rgb_img, (1, 2, 0))
            __import__('PIL').Image.fromarray(rgb_img).save(os.path.join(SAMPLE_DIR, f"sample_{idx:02d}_rgb.png"))

            gt_img = (gts_np[idx][0] * 255).astype(np.uint8)
            __import__('PIL').Image.fromarray(gt_img).save(os.path.join(SAMPLE_DIR, f"sample_{idx:02d}_gt.png"))

            seg_img = (seg_mask[idx][0] * 255).astype(np.uint8)
            __import__('PIL').Image.fromarray(seg_img).save(os.path.join(SAMPLE_DIR, f"sample_{idx:02d}_seg.png"))

            rec_img = (rec_mask[idx][0] * 255).astype(np.uint8)
            __import__('PIL').Image.fromarray(rec_img).save(os.path.join(SAMPLE_DIR, f"sample_{idx:02d}_rec.png"))

            fused_img = (fused_mask[idx][0] * 255).astype(np.uint8)
            __import__('PIL').Image.fromarray(fused_img).save(os.path.join(SAMPLE_DIR, f"sample_{idx:02d}_fused.png"))

        # 网格可视化：BANS/GT/Seg/Rec/Fused
        rows = images.size(0)
        cols = 5
        fig, axes = plt.subplots(rows, cols, figsize=(16, 4 * rows))
        if rows == 1:
            axes = np.expand_dims(axes, axis=0)
        for i in range(rows):
            rgb = imgs_np[i][:3]
            rgb_img = (rgb * 255).clip(0, 255).astype(np.uint8)
            rgb_img = np.transpose(rgb_img, (1, 2, 0))

            sample_gt = gts_np[i][0]
            sample_seg = seg_mask[i][0]
            sample_rec = rec_mask[i][0]
            sample_fused = fused_mask[i][0]

            axes[i, 0].imshow(rgb_img); axes[i, 0].set_title("Input (BANS)", fontsize=10); axes[i, 0].axis("off")
            axes[i, 1].imshow(sample_gt, cmap="gray"); axes[i, 1].set_title("GT", fontsize=10); axes[i, 1].axis("off")
            axes[i, 2].imshow(sample_seg, cmap="gray"); axes[i, 2].set_title("Seg", fontsize=10); axes[i, 2].axis("off")
            axes[i, 3].imshow(sample_rec, cmap="gray"); axes[i, 3].set_title("Rec", fontsize=10); axes[i, 3].axis("off")
            axes[i, 4].imshow(sample_fused, cmap="gray"); axes[i, 4].set_title("Fused", fontsize=10); axes[i, 4].axis("off")

        plt.tight_layout()
        vis_path = os.path.join(SAVE_DIR, f"{model_name}_prediction.png")
        plt.savefig(vis_path, dpi=300, bbox_inches="tight")
        print(f"示例输出已保存至: {SAMPLE_DIR}")
        print(f"可视化样例已保存: {vis_path}")

        # 附加输出：基于GT火点像素数的类别样例选择与标注（可选）
        if args.extra_output:
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

            full_loader = DataLoader(
                loader.dataset,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=loader.num_workers,
                pin_memory=pin_mem,
            )

            selected: Dict[str, List[Tuple[np.ndarray, np.ndarray, np.ndarray, int, int]]] = {
                "none": [], "very_few": [], "few": [], "many": [], "very_many": []
            }

            with torch.inference_mode():
                for images_b, masks_b in __import__('tqdm').tqdm(full_loader, desc="挑选样例"):
                    images_b = images_b.to(device, non_blocking=True)
                    masks_b = (masks_b > 0).to(device, non_blocking=True)

                    seg_b, rec_b, fused_b = eval_model(images_b)
                    preds_b = (torch.sigmoid(fused_b) >= args.threshold).float()

                    for i in range(images_b.size(0)):
                        gt = masks_b[i]
                        pr = preds_b[i]
                        gt_cnt = int(gt.sum().item())
                        pr_cnt = int(pr.sum().item())
                        cat = categorize(gt_cnt)

                        if len(selected[cat]) < samples_per_cat:
                            img_np = images_b[i].detach().cpu().numpy()
                            gt_np = gt.detach().cpu().numpy()
                            pr_np = pr.detach().cpu().numpy()
                            selected[cat].append((img_np, gt_np, pr_np, gt_cnt, pr_cnt))

                    if all(len(v) >= samples_per_cat for v in selected.values()):
                        break

            def plot_category(cat_name: str, items: List[Tuple[np.ndarray, np.ndarray, np.ndarray, int, int]]):
                if not items:
                    return
                rows = len(items)
                cols = 3
                fig, axes = plt.subplots(rows, cols, figsize=(12, 4 * rows))
                if rows == 1:
                    axes = np.expand_dims(axes, axis=0)
                for r, (img_np, gt_np, pr_np, gt_cnt, pr_cnt) in enumerate(items):
                    rgb = img_np[:3]
                    rgb_img = (rgb * 255).clip(0, 255).astype(np.uint8)
                    rgb_img = np.transpose(rgb_img, (1, 2, 0))

                    axes[r, 0].imshow(rgb_img)
                    axes[r, 0].set_title(f"Input (BANS)")
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

    cleanup_distributed()


if __name__ == "__main__":
    main()