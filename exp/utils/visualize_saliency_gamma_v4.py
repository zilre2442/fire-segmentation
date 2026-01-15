"""
ActiveFire saliency-style comparison for RGSNetV4 with different gamma settings.

Bins: 1-10, 10-100, 100-1000, >1000 (up to two samples each).
Models: four RGSNetV4 checkpoints with different gamma values (1.0, 1.5, 2.0, 2.5).
Outputs: one figure with RGB, GT, and four overlayed probability maps (jet on RGB).
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
from models.DualSight_Fire import RGSNetV4


def parse_args():
    parser = argparse.ArgumentParser(description="ActiveFire probability heatmaps for RGSNetV4 gamma sweep")
    parser.add_argument("--data-root", type=str, default="data/splits_activefire", help="数据根目录")
    parser.add_argument("--algo", type=str, default="voting", help="算法标签")
    parser.add_argument("--save-dir", type=str, default="output/hard_samples_comparison", help="结果保存目录")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=3, help="随机采样种子")
    parser.add_argument("--ckpt-gamma-1.0", dest="ckpt_g1", type=str, required=True, help="gamma=1.0 权重路径")
    parser.add_argument("--ckpt-gamma-1.5", dest="ckpt_g1_5", type=str, required=True, help="gamma=1.5 权重路径")
    parser.add_argument("--ckpt-gamma-2.0", dest="ckpt_g2", type=str, required=True, help="gamma=2.0 权重路径")
    parser.add_argument("--ckpt-gamma-2.5", dest="ckpt_g2_5", type=str, required=True, help="gamma=2.5 权重路径")
    return parser.parse_args()


def load_weights(model, path: str):
    state_dict = torch.load(path, map_location="cpu")
    if "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]
    cleaned = {k[7:] if k.startswith("module.") else k: v for k, v in state_dict.items()}
    model.load_state_dict(cleaned, strict=True)
    return model


def get_logits(model, x):
    out = model(x)
    return out[2]


def get_category(fire_count: int) -> str:
    if 1 <= fire_count < 11:
        return "1-10"
    if 10 <= fire_count < 101:
        return "10-100"
    if 100 <= fire_count < 1001:
        return "100-1000"
    return ">1000"


def select_indices(loader, seed: int) -> List[int]:
    """Select exactly up to 4 samples with >1000 fire pixels.

    We first collect all indices whose fire-pixel count is >1000, then
    randomly pick 4 (or fewer if not enough) using the given seed.
    """
    large_indices: List[int] = []
    idx_offset = 0
    with torch.no_grad():
        for images, masks in loader:
            b = images.size(0)
            counts = (masks > 0).view(b, -1).sum(dim=1).cpu().numpy()
            for i in range(b):
                c = int(counts[i])
                if c > 1000:
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

    band_cfg = [7, 6, 5]
    model_defs = {
        "gamma=1.0": args.ckpt_g1,
        "gamma=1.5": args.ckpt_g1_5,
        "gamma=2.0": args.ckpt_g2,
        "gamma=2.5": args.ckpt_g2_5,
    }

    models = {}
    for name, path in model_defs.items():
        m = RGSNetV4(n_channels=len(band_cfg), n_classes=1)
        m = load_weights(m, path)
        m.to(device)
        m.eval()
        models[name] = m

    selected = select_indices(loader, args.seed)
    print(f"Selected indices (by bin order): {selected}")
    if len(selected) == 0:
        raise RuntimeError("No samples found for the requested bins.")

    os.makedirs(args.save_dir, exist_ok=True)

    # Pack rows (all are >1000 bin); keep full view (no zoom)
    rows: List[Tuple[int, str, int, bool]] = []
    for idx in selected:
        _, mask = dataset[idx]
        fire_count = int((mask.numpy()[0] > 0).sum())
        cat = get_category(fire_count)
        zoom_flag = False
        rows.append((idx, cat, fire_count, zoom_flag))

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

    def render(indices_info: List[Tuple[int, str, int, bool]]):
        n_rows = len(indices_info)
        model_names = ["Images", "GT"] + list(models.keys())
        n_cols = len(model_names)
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.5 * n_cols, 3.5 * max(1, n_rows)))
        if n_rows == 1:
            axes = np.expand_dims(axes, axis=0)
        im = None
        for row_idx, (ds_idx, cat, fire_count, zoom_row) in enumerate(indices_info):
            image, mask = dataset[ds_idx]
            mask_np = (mask.numpy()[0] > 0).astype(np.uint8)
            if zoom_row:
                yslice, xslice = make_bbox(mask_np)
            else:
                yslice, xslice = slice(None), slice(None)

            rgb_full = image[[6, 5, 1]].numpy().transpose(1, 2, 0)
            rgb_full = (rgb_full - rgb_full.min()) / (rgb_full.max() - rgb_full.min() + 1e-8)
            rgb_crop = rgb_full[yslice, xslice, :]

            for col_idx, name in enumerate(model_names):
                ax = axes[row_idx, col_idx]
                if name == "Images":
                    if zoom_row:
                        ax.imshow(rgb_full)
                        y0, y1 = yslice.start, yslice.stop
                        x0, x1 = xslice.start, xslice.stop
                        ax.add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0,
                                                   fill=False, edgecolor="yellow", linewidth=1.5))
                    else:
                        ax.imshow(rgb_full)
                    ax.axis("off")
                    if row_idx == 0:
                        ax.set_title(name, fontsize=10)
                    if col_idx == 0:
                        ax.set_ylabel(f"{cat}\n#fire={fire_count}", fontsize=9)
                    continue

                if name == "GT":
                    gt = (mask.numpy()[0] > 0).astype(np.float32)
                    ax.imshow(gt[yslice, xslice], cmap="gray", vmin=0.0, vmax=1.0)
                    ax.axis("off")
                    if row_idx == 0:
                        ax.set_title(name, fontsize=10)
                    if col_idx == 0:
                        ax.set_ylabel(f"{cat}\n#fire={fire_count}", fontsize=9)
                    continue

                band_idx = [b - 1 for b in band_cfg]
                inp = image[band_idx].unsqueeze(0).to(device)
                with torch.no_grad():
                    logits = get_logits(models[name], inp)
                    prob = torch.sigmoid(logits).cpu().numpy()[0, 0]

                ax.imshow(rgb_crop)
                im = ax.imshow(prob[yslice, xslice], cmap="jet", vmin=0.0, vmax=1.0, alpha=0.75)
                ax.axis("off")
                if row_idx == 0:
                    ax.set_title(name, fontsize=10)
                if col_idx == 0:
                    ax.set_ylabel(f"{cat}\n#fire={fire_count}", fontsize=9)

        cax = fig.add_axes([0.92, 0.2, 0.015, 0.6])
        fig.colorbar(im, cax=cax, label="Fire probability")
        plt.tight_layout(rect=[0, 0, 0.9, 1])

        base = "saliency_activefire_v4_gamma"
        png_path = os.path.join(args.save_dir, f"{base}.png")
        pdf_path = os.path.join(args.save_dir, f"{base}.pdf")
        plt.savefig(png_path, dpi=200)
        plt.savefig(pdf_path, dpi=200)
        plt.close()
        print(f"Saved {png_path} and {pdf_path}")

    render(rows)


if __name__ == "__main__":
    main()