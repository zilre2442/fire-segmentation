"""
ActiveFire saliency-style fire probability maps for multiple models.

Selects samples by fire-pixel count bins: 1-10, 10-100, 100-1000, >1000 (up to two samples each).
Models: baseline+BCE (bands 7-6-2), baseline+SpatialFocalLoss (7-6-5), DualSight-Fire BCE (7-6-5), DualSight-Fire SFL (7-6-5).
Outputs probability heatmaps similar to provided examples; small-fire samples are zoomed for clearer comparison.

CUDA_VISIBLE_DEVICES=4 python exp/utils/visualize_saliency_activefire.py --data-root data/splits_activefire --algo voting --ckpt-baseline-bce output/baseline/voting_202512051446/weights/model_best.pth --ckpt-baseline-sfl output/baseline/voting_202512182150/weights/model_best.pth --ckpt-baseline-focal output/baseline/voting_202512191005/weights/model_best.pth --ckpt-v4-bce output/RGS_Net_V4/voting_202512052148/weights/model_best.pth --ckpt-v4-sfl output/RGS_Net_V4/voting_202512162143/weights/model_best.pth --save-dir output/hard_samples_comparison
"""
import os
import sys
import argparse
from typing import Dict, List, Tuple

import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

# project root
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from dataset import LandsatFireDataset
from models.baseline import UNet
from models.DualSight_Fire import RGSNetV4


def parse_args():
    parser = argparse.ArgumentParser(description="ActiveFire probability heatmaps across four models")
    parser.add_argument("--data-root", type=str, default="data/splits_activefire", help="数据根目录")
    parser.add_argument("--algo", type=str, default="voting", help="算法标签")
    parser.add_argument("--save-dir", type=str, default="output/hard_samples_comparison", help="结果保存目录")
    parser.add_argument("--threshold", type=float, default=0.5, help="用于样本选择的阈值(不影响可视化)")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=2, help="随机采样种子")
    parser.add_argument("--ckpt-baseline-bce", type=str, required=True, help="Baseline+BCE (7-6-2) 权重路径")
    parser.add_argument("--ckpt-baseline-sfl", type=str, required=True, help="Baseline+SFL (7-6-5) 权重路径")
    parser.add_argument("--ckpt-baseline-focal", type=str, required=True, help="Baseline+Focal (7-6-5) 权重路径")
    parser.add_argument("--ckpt-v4-bce", type=str, required=True, help="DualSight-Fire+BCE (7-6-5) 权重路径")
    parser.add_argument("--ckpt-v4-sfl", type=str, required=True, help="DualSight-Fire+SFL (7-6-5) 权重路径")
    return parser.parse_args()


def load_weights(model, path: str):
    state_dict = torch.load(path, map_location="cpu")
    if "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]
    cleaned = {k[7:] if k.startswith("module.") else k: v for k, v in state_dict.items()}
    model.load_state_dict(cleaned, strict=True)
    return model


def get_logits(model, x, name: str):
    if "DualSight-Fire" in name:
        out = model(x)
        return out[2]
    return model(x)


def select_indices(loader, seed: int) -> List[int]:
    large_indices: List[int] = []
    idx_offset = 0
    with torch.no_grad():
        for images, masks in loader:
            b = images.size(0)
            counts = (masks > 0).view(b, -1).sum(dim=1).cpu().numpy()
            for i in range(b):
                if int(counts[i]) > 1000:
                    large_indices.append(idx_offset + i)
            idx_offset += b

    if len(large_indices) == 0:
        return []

    rng = np.random.default_rng(seed)
    take = min(4, len(large_indices))
    chosen = rng.choice(large_indices, size=take, replace=False)
    return chosen.tolist()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    test_csv = os.path.join(args.data_root, f"{args.algo}_test.csv")
    possible_path = os.path.join(args.data_root, f"by_fire_pixels_{args.algo}", "test", f"{args.algo}_test.csv")
    if os.path.exists(possible_path):
        test_csv = possible_path
    if not os.path.exists(test_csv):
        raise FileNotFoundError(f"Test CSV not found: {test_csv}")

    dataset = LandsatFireDataset(test_csv, bands=tuple(range(1, 11)))
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    # Order intentionally set so U-Net+Focal appears between BCE and SFL
    band_cfg = {
        "U-Net+BCE": [7, 6, 2],
        "U-Net+Focal": [7, 6, 5],
        "U-Net+SFL": [7, 6, 5],
        "DualSight-Fire+BCE": [7, 6, 5],
        "DualSight-Fire+SFL": [7, 6, 5],
    }
    model_defs = {
        "U-Net+BCE": (UNet, args.ckpt_baseline_bce),
        "U-Net+Focal": (UNet, args.ckpt_baseline_focal),
        "U-Net+SFL": (UNet, args.ckpt_baseline_sfl),
        "DualSight-Fire+BCE": (RGSNetV4, args.ckpt_v4_bce),
        "DualSight-Fire+SFL": (RGSNetV4, args.ckpt_v4_sfl),
    }

    models = {}
    for name, (cls, path) in model_defs.items():
        in_ch = len(band_cfg[name])
        if "DualSight-Fire" in name:
            m = cls(n_channels=in_ch, n_classes=1)
        else:
            m = cls(n_channels=in_ch, n_classes=1)
        m = load_weights(m, path)
        m.to(device)
        m.eval()
        models[name] = m

    selected = select_indices(loader, args.seed)
    print(f"Selected indices (by bin order): {selected}")
    if len(selected) == 0:
        raise RuntimeError("No samples found for the requested bins.")

    os.makedirs(args.save_dir, exist_ok=True)

    # Collect >1000 samples with no zoom (full view)
    samples: List[Tuple[int, int]] = []  # (idx, fire_count)
    for idx in selected:
        _, mask = dataset[idx]
        fire_count = int((mask.numpy()[0] > 0).sum())
        samples.append((idx, fire_count))

    def make_bbox(mask_np: np.ndarray, margin: int = 12) -> Tuple[slice, slice]:
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

    def render(indices_info: List[Tuple[int, int]], suffix: str):
        if len(indices_info) == 0:
            return
        n_rows = len(indices_info)
        # Columns: RGB, then probability maps from each model
        model_names = ["Images"] + list(models.keys())
        n_cols = len(model_names)
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.5 * n_cols, 3.5 * max(1, n_rows)))
        if n_rows == 1:
            axes = np.expand_dims(axes, axis=0)
        im = None
        for row_idx, (ds_idx, fire_count) in enumerate(indices_info):
            image, mask = dataset[ds_idx]
            mask_np = (mask.numpy()[0] > 0).astype(np.uint8)
            yslice, xslice = slice(None), slice(None)

            # Precompute normalized RGB and its crop for overlay
            rgb_full = image[[6, 5, 1]].numpy().transpose(1, 2, 0)
            rgb_full = (rgb_full - rgb_full.min()) / (rgb_full.max() - rgb_full.min() + 1e-8)
            rgb_crop = rgb_full[yslice, xslice, :]

            for col_idx, name in enumerate(model_names):
                ax = axes[row_idx, col_idx]
                if name == "Images":
                    ax.imshow(rgb_full)
                    ax.axis("off")
                    if row_idx == 0:
                        ax.set_title(name, fontsize=10)
                    if col_idx == 0:
                        ax.set_ylabel(f">1000\n#fire={fire_count}", fontsize=9)
                    continue

                band_idx = [b - 1 for b in band_cfg[name]]
                inp = image[band_idx].unsqueeze(0).to(device)
                with torch.no_grad():
                    logits = get_logits(models[name], inp, name)
                    prob = torch.sigmoid(logits).cpu().numpy()[0, 0]
                # Show probability heatmap without RGB overlay
                im = ax.imshow(prob[yslice, xslice], cmap="jet", vmin=0.0, vmax=1.0)
                ax.axis("off")
                if row_idx == 0:
                    ax.set_title(name, fontsize=10)
                if col_idx == 0:
                    ax.set_ylabel(f">1000\n#fire={fire_count}", fontsize=9)

        cax = fig.add_axes([0.92, 0.2, 0.015, 0.6])
        fig.colorbar(im, cax=cax, label="Fire probability")
        plt.tight_layout(rect=[0, 0, 0.9, 1])

        base = f"saliency_activefire_{suffix}"
        png_path = os.path.join(args.save_dir, f"{base}.png")
        pdf_path = os.path.join(args.save_dir, f"{base}.pdf")
        plt.savefig(png_path, dpi=200)
        plt.savefig(pdf_path, dpi=200)
        plt.close()
        print(f"Saved {png_path} and {pdf_path}")

    render(samples, suffix="gt1000")


if __name__ == "__main__":
    main()
