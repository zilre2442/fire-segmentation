# DualSight-Fire

DualSight-Fire: A Cognitive Dual-Decoder Network for Robust Fire Segmentation in Multispectral Remote Sensing Image

This repository contains multiple fire segmentation models (Baseline / AttentionUNet / UNet++ / FDE-UNet / FPS-U2Net / Dual-Sight Fire and the StageFusion variant), along with training/evaluation scripts and dataset splitting utilities.

## 1. Repository Structure

(As implemented in the current codebase)

```
.
├─ dataset.py                         # Dataset: LandsatFireDataset (reads image/mask paths from CSV)
├─ loss.py                            # Losses (SpatialFocalLoss / SpatialFocalTverskyLoss, etc.)
├─ utils.py                           # Common utilities
├─ environment.yml                    # Conda environment (Python=3.10 + rasterio/gdal, etc.)
├─ models/
│  ├─ baseline.py                     # Baseline UNet
│  ├─ attention_unet.py               # Attention UNet
│  ├─ unet_plusplus.py                # UNet++
│  ├─ FDE_Net.py                      # FDE-UNet
│  ├─ FPS_U2Net.py                    # FPS-U2Net
│  ├─ DualSight_Fire.py               # Dual-Sight Fire / StageFusion
│  └─ RGS_Net_V4_no_transformer.py    # Historical/ablation file (kept)
├─ exp/
│  ├─ train_scripts/                  # Training scripts (single GPU / torchrun DDP)
│  ├─ eval_scripts/                   # Evaluation scripts
│  └─ utils/                          # Visualization/prediction utilities
├─ data/
│  ├─ splits_activefire/              # ActiveFire split CSVs (voting, etc.)
│  ├─ splits_land8fire/               # Land8Fire split CSVs
│  ├─ splits_manual/                  # Manual annotation splits
│  └─ ...                             # Data processing and splitting scripts
├─ dataset/
│  ├─ activefire/                     # Layout: images/ masks/
│  ├─ Land8Fire/                      # Layout: images/ masks/
│  └─ mannual_annotations/            # Manual annotation data
└─ output/                            # Training outputs (weights, logs, curves, eval reports)
```

## 2. Environment Setup

Conda is recommended (more reliable for `rasterio/gdal`).

```bash
conda env create -f environment.yml
conda activate fire
```

Install PyTorch (choose the command matching your CUDA version; example below uses CUDA 12.1):

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

Optional (only if you need FLOPs/model analysis):

```bash
pip install fvcore
```

## 3. Data Preparation and CSV Format

By default, training/evaluation scripts read `{algo}_train.csv / {algo}_val.csv / {algo}_test.csv` from the directory specified by `--data-root`.

Example: ActiveFire voting split:

```
data/splits_activefire/
├─ voting_train.csv
├─ voting_val.csv
└─ voting_test.csv
```

CSV format:
- The first row is a header (will be skipped)
- Each row contains at least two columns:
	1) Image path (e.g., multi-band GeoTIFF readable by `rasterio`)
	2) Mask path (readable by `rasterio`; by default, band 1 is used as the [H, W] mask)

Band notes:
- In `dataset.py`, `bands` are 1-indexed (for 10 Landsat bands, use 1 to 10)
- Training scripts default to `--bands 7 6 5`

## 4. Training

All training scripts support:
- Single GPU: run with `python ...`
- Multi-GPU DDP: run with `torchrun ...` (scripts automatically read `WORLD_SIZE/LOCAL_RANK`)

### 4.1 Dual-Sight Fire

Single GPU:

```bash
CUDA_VISIBLE_DEVICES=0 python exp/train_scripts/train_DualSight_Fire.py \
	--data-root data/splits_activefire --algo voting --bands 7 6 5
```

Multi-GPU:

```bash
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 --master_port=65531 \
	exp/train_scripts/train_DualSight_Fire.py --data-root data/splits_activefire --algo voting
```

Default output directory:
- `output/DualSight_Fire/{algo}_YYYYmmddHHMM/`

### 4.2 Dual-Sight Fire StageFusion

Single GPU:

```bash
CUDA_VISIBLE_DEVICES=0 python exp/train_scripts/train_DualSight_Fire_stagefusion.py \
	--data-root data/splits_activefire --algo voting --bands 7 6 5
```

Multi-GPU:

```bash
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 --master_port=65530 \
	exp/train_scripts/train_DualSight_Fire_stagefusion.py --data-root data/splits_activefire --algo voting
```

Default output directory:
- `output/DualSight_Fire_stagefusion/{algo}_YYYYmmddHHMM/`

### 4.3 Other Models

Also available under `exp/train_scripts/`:
- `train_baseline.py`
- `train_AttentionUNet.py`
- `train_UNetPlusPlus.py`
- `train_FDE_UNet.py`
- `train_FPS_U2Net.py`

Their argument style is consistent with Dual-Sight Fire (e.g., `--data-root/--algo/--bands/--epochs/--batch-size/...`).

## 5. Evaluation

Example: Dual-Sight Fire StageFusion:

```bash
CUDA_VISIBLE_DEVICES=0 python exp/eval_scripts/eval_DualSight_Fire_stagefusion.py \
	--data-root data/splits_activefire --algo voting --bands 7 6 5 \
	--output-type fused --threshold 0.5 \
	--save-dir output/DualSight_Fire_stagefusion/voting_202512191353
```

By default, weights are loaded from `--save-dir/weights/model_best.pth`. You can also specify `--model-path`.

Other evaluation scripts are under `exp/eval_scripts/`:
- `eval_DualSight_Fire.py`
- `eval_baseline.py` / `eval_AttentionUNet.py` / `eval_UNetPlusPlus.py` / `eval_FDE_UNet.py` / `eval_FPS_U2Net.py`

## 6. Output Contents

Training output directories typically include:
- `weights/model_best.pth`
- `train_log.txt`
- `hyperparameters.txt`
- `loss_curve.png`

## 7. License and Citation

This project is intended for academic research and teaching. If you use it in a paper or project, please cite this repository and acknowledge the authors.
