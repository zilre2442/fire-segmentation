import os
import sys
import argparse
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from dataset import LandsatFireDataset
from models.activefire_unet_baseline import ActiveFireUNetBaseline

# DATA_ROOT = "data/splits_activefire"
# ALGORITHM = "voting"
DATA_ROOT = "data/splits_land8fire"
ALGORITHM = "Land8Fire"
BANDS = (7,6,2)
THRESH = 0.5
# CUDA_VISIBLE_DEVICES=1 python exp/eval_scripts/eval_activefire_baseline_unet.py --weights output/ActiveFireBaseline/Land8Fire_202511261519/weights/model_best.pth

def parse_args():
    p = argparse.ArgumentParser(description="Evaluate ActiveFire UNet Baseline")
    p.add_argument("--data-root", type=str, default=DATA_ROOT)
    p.add_argument("--algorithm", type=str, default=ALGORITHM,
                   choices=["voting", "Kumar-Roy", "Murphy", "Schroeder", "intersection", "Land8Fire"])
    p.add_argument("--fire-category", type=str, default=None,
                   choices=["very_few", "few", "many", "very_many"])
    p.add_argument("--bands", type=int, nargs=3, default=BANDS)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--weights", type=str, required=True, help="Path to model_best.pth")
    p.add_argument("--base-filters", type=int, default=64)
    p.add_argument("--threshold", type=float, default=THRESH)
    return p.parse_args()


def get_test_csv(data_root: str, algorithm: str, fire_category: str = None) -> str:
    if fire_category is None:
        return os.path.join(data_root, f"{algorithm}_test.csv")
    if algorithm == "Land8Fire":
        subdir = os.path.join(data_root, "by_fire_pixels")
    else:
        subdir = os.path.join(data_root, f"by_fire_pixels_{algorithm}")
    return os.path.join(subdir, "test", f"{algorithm}_test_{fire_category}.csv")


def compute_metrics(tp, fp, fn, tn):
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    iou_fire = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0
    iou_back = tn / (tn + fp + fn) if (tn + fp + fn) > 0 else 0.0
    miou = (iou_fire + iou_back) / 2.0
    dice = (2 * tp) / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0
    return dict(precision=precision, recall=recall, f1=f1, iou_fire=iou_fire, iou_back=iou_back, miou=miou, dice=dice)


def main():
    args = parse_args()
    test_csv = get_test_csv(args.data_root, args.algorithm, args.fire_category)
    ds = LandsatFireDataset(test_csv, bands=tuple(args.bands))
    loader = torch.utils.data.DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ActiveFireUNetBaseline(n_channels=len(args.bands), base_filters=args.base_filters)
    state = torch.load(args.weights, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()

    tp=fp=fn=tn=0
    with torch.no_grad():
        for images, masks in tqdm(loader, desc="Evaluating"):
            images = images.to(device, non_blocking=True)
            gt = (masks > 0).float()
            if gt.dim() == 3: gt = gt.unsqueeze(1)
            gt = gt.to(device, non_blocking=True)
            logits = model(images)
            if logits.shape[2:] != gt.shape[2:]:
                logits = nn.functional.interpolate(logits, size=gt.shape[2:], mode="bilinear", align_corners=False)
            probs = torch.sigmoid(logits)
            preds = (probs >= args.threshold).float()
            tp += torch.logical_and(preds == 1, gt == 1).sum().item()
            fp += torch.logical_and(preds == 1, gt == 0).sum().item()
            fn += torch.logical_and(preds == 0, gt == 1).sum().item()
            tn += torch.logical_and(preds == 0, gt == 0).sum().item()

    metrics = compute_metrics(tp, fp, fn, tn)
    report_path = os.path.join(os.path.dirname(args.weights), "baseline_eval.txt")
    with open(report_path, "w") as f:
        f.write("ActiveFire UNet Baseline Evaluation\n")
        for k,v in metrics.items():
            f.write(f"{k}: {v:.6f}\n")
        f.write(f"TP:{tp} FP:{fp} FN:{fn} TN:{tn}\n")
    print("Evaluation complete. Metrics:")
    for k,v in metrics.items():
        print(f"{k}: {v:.6f}")
    print("Report saved:", report_path)

if __name__ == "__main__":
    main()
