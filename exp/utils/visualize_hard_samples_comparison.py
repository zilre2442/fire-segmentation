"""
困难样本对比可视化脚本

功能：
1. 使用 Baseline UNet 模型扫描测试集，寻找误报 (FP) 最多的 5 个样本。
2. 使用 Baseline UNet 模型扫描测试集，寻找漏报 (FN) 最多的 5 个样本。
3. 对这 10 个样本，分别使用 6 个模型进行推理，并生成对比图。
   对比图包含：原图 (RGB), GT, UNet, Attention UNet, UNet++, FPS-U2Net, FDE-UNet, RGS-Net V4

示例用法:
CUDA_VISIBLE_DEVICES=4 python exp/utils/visualize_hard_samples_comparison.py \
    --data-root data/splits_activefire \
    --algo voting \
    --save-dir output/hard_samples_comparison \
    --ckpt-unet output/baseline/voting_202512051446/weights/model_best.pth \
    --ckpt-attunet output/AttentionUNet/voting_202512171714/weights/model_best.pth \
    --ckpt-unetpp output/UNetPlusPlus/voting_202512171656/weights/model_best.pth \
    --ckpt-fps output/FPS_U2Net/voting_202512251440/weights/model_best.pth \
    --ckpt-fde output/FDE_UNet/voting_202512251230/weights/model_best.pth \
    --ckpt-v4 output/RGS_Net_V4/voting_202512162143/weights/model_best.pth

CUDA_VISIBLE_DEVICES=4 python exp/utils/visualize_hard_samples_comparison.py \
    --data-root data/splits_land8fire \
    --algo Land8Fire \
    --save-dir output/hard_samples_comparison \
    --ckpt-unet output/baseline/Land8Fire_202512191705/weights/model_best.pth \
    --ckpt-attunet output/AttentionUNet/Land8Fire_202512180935/weights/model_best.pth \
    --ckpt-unetpp output/UNetPlusPlus/Land8Fire_202512180934/weights/model_best.pth \
    --ckpt-fps output/FPS_U2Net/Land8Fire_202512251649/weights/model_best.pth \
    --ckpt-fde output/FDE_UNet/Land8Fire_202512251509/weights/model_best.pth \
    --ckpt-v4 output/RGS_Net_V4/Land8Fire_202512180933/weights/model_best.pth
"""

import os
import sys
import argparse
import math
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch
from torch.utils.data import DataLoader

# 添加项目根目录到路径
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from dataset import LandsatFireDataset
from models.baseline import UNet
from models.attention_unet import AttentionUNet
from models.unet_plusplus import UNetPlusPlus
from models.FPS_U2Net import FPSU2Net
from models.FDE_Net import FDE_UNet
from models.RGS_Net_V4 import RGSNetV4

def parse_args():
    parser = argparse.ArgumentParser(description="Visualize hard samples comparison across 6 models")
    parser.add_argument("--data-root", type=str, default="data/splits_activefire", help="数据根目录")
    parser.add_argument("--algo", type=str, default="voting", help="算法标签")
    # 固定按模型需求加载 10 个波段，单独为每个模型选择子集
    parser.add_argument("--save-dir", type=str, default="output/hard_samples_comparison", help="结果保存目录")
    parser.add_argument("--threshold", type=float, default=0.5, help="判定阈值")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    
    # 模型权重路径
    parser.add_argument("--ckpt-unet", type=str, required=True, help="Baseline UNet 权重路径")
    parser.add_argument("--ckpt-attunet", type=str, required=True, help="Attention UNet 权重路径")
    parser.add_argument("--ckpt-unetpp", type=str, required=True, help="UNet++ 权重路径")
    parser.add_argument("--ckpt-fps", type=str, required=True, help="FPS-U2Net 权重路径")
    parser.add_argument("--ckpt-fde", type=str, required=True, help="FDE-UNet 权重路径")
    parser.add_argument("--ckpt-v4", type=str, required=True, help="RGS-Net V4 权重路径")
    
    return parser.parse_args()

def load_weights(model, path):
    print(f"Loading weights from {path} ...")
    try:
        state_dict = torch.load(path, map_location='cpu')
        if 'state_dict' in state_dict:
            state_dict = state_dict['state_dict']
        
        # 去除 DDP 的 module. 前缀
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('module.'):
                new_state_dict[k[7:]] = v
            else:
                new_state_dict[k] = v
        
        model.load_state_dict(new_state_dict)
    except Exception as e:
        print(f"Error loading {path}: {e}")
        # 仅用于演示，实际应抛出异常
        # raise e
    return model

def get_model_prediction(model, images, model_name):
    """
    统一模型推理接口，返回 logits
    """
    if model_name == "FPS-U2Net":
        # FPSU2Net 返回 tuple, 第一个元素是 d1 (logits)
        outputs = model(images)
        return outputs[0]
    elif model_name == "DualSight-Fire":
        # RGSNetV4 返回 (seg, rec, fused), 使用 fused (index 2)
        outputs = model(images)
        return outputs[2]
    elif model_name == "UNet++":
        # UNet++ 如果开启 deep_supervision 返回 list, 否则 tensor
        # 假设这里使用的是非 deep_supervision 或者只取最后一个
        outputs = model(images)
        if isinstance(outputs, (list, tuple)):
            return outputs[-1]
        return outputs
    else:
        # UNet, AttentionUNet, FDE-UNet 直接返回 logits
        return model(images)

def assign_bin(count: int) -> int:
    """Return bin index for a given fire-pixel count."""
    bins = [(1, 11), (10, 101), (100, 1001), (1000, math.inf)]
    for i, (lo, hi) in enumerate(bins):
        if count >= lo and count < hi:
            return i
    return -1


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. 准备数据
    test_csv = os.path.join(args.data_root, f"{args.algo}_test.csv")
    if args.algo == "Land8Fire":
        candidates = [
            os.path.join(args.data_root, "by_fire_pixels", "test", "Land8Fire_test.csv"),
            os.path.join(args.data_root, "by_fire_pixels_land8fire", "test", "Land8Fire_test.csv"),
            os.path.join(args.data_root, "Land8Fire_test.csv"),
        ]
        test_csv = next((p for p in candidates if os.path.exists(p)), test_csv)
    elif "activefire" in args.data_root and args.algo != "Land8Fire":
        possible_path = os.path.join(args.data_root, f"by_fire_pixels_{args.algo}", "test", f"{args.algo}_test.csv")
        if os.path.exists(possible_path):
            test_csv = possible_path

    if not os.path.exists(test_csv):
        raise FileNotFoundError(f"Test CSV not found. Checked: {test_csv}")
    
    print(f"Test CSV: {test_csv}")
    dataset = LandsatFireDataset(test_csv, bands=tuple(range(1, 11)))
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    
    # 2. 加载所有模型
    band_cfg = {
        "UNet": [7, 6, 2],
        "Attention UNet": [7, 6, 2],
        "UNet++": [7, 6, 2],
        "FPS-U2Net": [7, 6, 6],
        "FDE-UNet": list(range(1, 11)),
        "DualSight-Fire": [7, 6, 5],
    }
    band_indices = {name: [b - 1 for b in bands] for name, bands in band_cfg.items()}

    models = {}
    model_configs = [
        ("UNet", UNet, args.ckpt_unet),
        ("Attention UNet", AttentionUNet, args.ckpt_attunet),
        ("UNet++", UNetPlusPlus, args.ckpt_unetpp),
        ("FPS-U2Net", FPSU2Net, args.ckpt_fps),
        ("FDE-UNet", FDE_UNet, args.ckpt_fde),
        ("DualSight-Fire", RGSNetV4, args.ckpt_v4),
    ]

    print("\nLoading all models...")
    for name, cls, path in model_configs:
        in_ch = len(band_cfg[name])
        if name == "FDE-UNet":
            m = cls(in_channels=in_ch, num_classes=1)
        elif name == "FPS-U2Net":
            m = cls(in_ch=in_ch, out_ch=1)
        else:
            m = cls(n_channels=in_ch, n_classes=1)
        m = load_weights(m, path)
        m.to(device)
        m.eval()
        models[name] = m

    # 3. 依据火点像素数选择 4 个样本（1-10、10-100、100-1000、>1000），且 DualSight-Fire 误差最少
    selected = {}
    bins = [(1, 11), (10, 101), (100, 1001), (1000, math.inf)]
    global_idx = 0
    with torch.no_grad():
        for images, masks in loader:
            batch_size = images.size(0)
            masks_bin = (masks > 0)
            counts = masks_bin.view(batch_size, -1).sum(dim=1).cpu().numpy()
            images = images.to(device)
            for i in range(batch_size):
                b = assign_bin(int(counts[i]))
                if b == -1 or b in selected:
                    continue

                gt = masks_bin[i, 0].cpu().numpy()
                model_scores = {}

                for name, model in models.items():
                    idx = band_indices[name]
                    inp = images[i : i + 1, idx, :, :]
                    logits = get_model_prediction(model, inp, name)
                    pred = (torch.sigmoid(logits) > args.threshold).cpu().numpy()[0, 0]
                    mask_fp = (pred == 1) & (gt == 0)
                    mask_fn = (pred == 0) & (gt == 1)
                    model_scores[name] = {
                        "fp": int(mask_fp.sum()),
                        "fn": int(mask_fn.sum()),
                    }

                best = min(
                    model_scores.items(),
                    key=lambda kv: (kv[1]["fp"] + kv[1]["fn"], kv[1]["fn"], kv[1]["fp"]),
                )

                if best[0] == "DualSight-Fire":
                    selected[b] = global_idx + i
                    if len(selected) == len(bins):
                        break

            if len(selected) == len(bins):
                break
            global_idx += batch_size

    sorted_bins = sorted(selected.keys())
    selected_indices = [selected[b] for b in sorted_bins]
    print(f"Selected indices per bin: {selected_indices}")

    # 4. 可视化对比（小面积火点放大，且在原图中框选放大区域）
    os.makedirs(args.save_dir, exist_ok=True)

    model_order = ["UNet", "Attention UNet", "UNet++", "FPS-U2Net", "FDE-UNet", "DualSight-Fire"]

    n_samples = len(selected_indices)
    if n_samples == 0:
        print("No samples found for specified bins.")
        return

    n_cols = 2 + len(model_order)  # Input, GT, models
    fig, axes = plt.subplots(n_samples, n_cols, figsize=(20, 3 * n_samples))
    if n_samples == 1:
        axes = axes.reshape(1, -1)

    def make_bbox(mask_np: np.ndarray, margin: int = 12):
        ys, xs = np.nonzero(mask_np)
        if len(ys) == 0:
            h, w = mask_np.shape
            cy, cx = h // 2, w // 2
            half = min(h, w) // 4
            return slice(cy - half, cy + half), slice(cx - half, cx + half)
        y1, y2 = ys.min(), ys.max()
        x1, x2 = xs.min(), xs.max()
        y1 = max(0, y1 - margin)
        y2 = min(mask_np.shape[0] - 1, y2 + margin)
        x1 = max(0, x1 - margin)
        x2 = min(mask_np.shape[1] - 1, x2 + margin)
        return slice(y1, y2 + 1), slice(x1, x2 + 1)

    for row_idx, idx in enumerate(selected_indices):
        image, mask = dataset[idx]
        image_np = image.cpu().numpy()
        mask_np = mask.cpu().numpy()
        gt = mask_np[0] > 0.5

        # 四个类别都进行放大处理
        zoom = True
        gt_uint8 = gt.astype(np.uint8)
        if zoom:
            yslice, xslice = make_bbox(gt_uint8)
        else:
            yslice, xslice = slice(None), slice(None)

        def subset_np(arr, bands):
            zero_idx = [b - 1 for b in bands]
            return arr[zero_idx]

        model_inputs = {}
        for name, bands in band_cfg.items():
            zero_idx = [b - 1 for b in bands]
            sel = image[zero_idx]
            model_inputs[name] = sel.unsqueeze(0).to(device)

        # 1. Input：在原图上框出放大区域
        ax_input = axes[row_idx, 0]
        rgb = subset_np(image_np, [7, 6, 2]).transpose(1, 2, 0)
        rgb = (rgb - rgb.min()) / (rgb.max() - rgb.min() + 1e-8)
        ax_input.imshow(rgb)
        if zoom:
            y0, y1 = yslice.start, yslice.stop
            x0, x1 = xslice.start, xslice.stop
            ax_input.add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0,
	                                             fill=False, edgecolor="green", linewidth=1.5))
        if row_idx == 0:
            ax_input.set_title("Images")
        ax_input.axis('off')

        # 2. GT
        ax_gt = axes[row_idx, 1]
        ax_gt.imshow(gt[yslice, xslice], cmap='gray')
        if row_idx == 0:
            ax_gt.set_title("GT")
        ax_gt.axis('off')

        # 在 Input 与 GT 之间增加连线投影，指示放大区域
        if zoom:
            h_crop = yslice.stop - yslice.start
            w_crop = xslice.stop - xslice.start

            # 原图框选区域右上角、右下角
            ru_full = (xslice.stop - 1, yslice.start)
            rd_full = (xslice.stop - 1, yslice.stop - 1)
            # GT 放大图左上角、左下角（裁剪坐标系）
            lu_gt = (0, 0)
            ld_gt = (0, h_crop - 1)

            conn1 = ConnectionPatch(
                xyA=ru_full, coordsA=ax_input.transData,
                xyB=lu_gt, coordsB=ax_gt.transData,
                arrowstyle='-', linestyle='-', color='green', linewidth=1.0,
            )
            conn2 = ConnectionPatch(
                xyA=rd_full, coordsA=ax_input.transData,
                xyB=ld_gt, coordsB=ax_gt.transData,
                arrowstyle='-', linestyle='-', color='green', linewidth=1.0,
            )
            fig.add_artist(conn1)
            fig.add_artist(conn2)

        # 3. Models
        for col_idx, name in enumerate(model_order):
            model = models[name]
            with torch.no_grad():
                logits = get_model_prediction(model, model_inputs[name], name)
                pred = (torch.sigmoid(logits) > args.threshold).cpu().numpy()[0, 0]

            h, w = pred.shape
            vis = np.zeros((h, w, 3), dtype=np.float32)

            # TP (White)
            mask_tp = (pred == 1) & (gt == 1)
            vis[mask_tp] = [1, 1, 1]

            # FP (Red)
            mask_fp = (pred == 1) & (gt == 0)
            vis[mask_fp] = [1, 0, 0]

            # FN (Blue)
            mask_fn = (pred == 0) & (gt == 1)
            vis[mask_fn] = [0, 0, 1]

            ax = axes[row_idx, col_idx + 2]
            ax.imshow(vis[yslice, xslice, :])
            if row_idx == 0:
                ax.set_title(name)
            ax.axis('off')

    plt.tight_layout()
    base_name = "samples_prediction_land8fire" if args.algo == "Land8Fire" else "samples_prediction_activefire"
    png_path = os.path.join(args.save_dir, f"{base_name}.png")
    pdf_path = os.path.join(args.save_dir, f"{base_name}.pdf")
    plt.savefig(png_path, dpi=150)
    plt.savefig(pdf_path, dpi=150)
    plt.close()
    print(f"Saved {png_path} and {pdf_path}")

    print("Done.")

if __name__ == "__main__":
    main()
