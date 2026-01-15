# DualSight-Fire

A Dual-Decoder network for small fire segmentation in remote sensing imagery.

本仓库包含多种分割模型（Baseline/AttentionUNet/UNet++/FDE-UNet/FPS-U2Net/Dual-Sight Fire 及 StageFusion 变体），以及配套的训练/评估脚本与数据切分工具。

## 1. 仓库结构

（以当前代码为准）

```
.
├─ dataset.py                         # 数据集：LandsatFireDataset（从 CSV 读取 img/mask 路径）
├─ loss.py                            # 损失函数（SpatialFocalLoss/SpatialFocalTverskyLoss 等）
├─ utils.py                           # 通用工具
├─ environment.yml                    # Conda 环境（Python=3.10 + rasterio/gdal 等）
├─ models/
│  ├─ baseline.py                     # Baseline UNet
│  ├─ attention_unet.py               # Attention UNet
│  ├─ unet_plusplus.py                # UNet++
│  ├─ FDE_Net.py                      # FDE-UNet
│  ├─ FPS_U2Net.py                    # FPS-U2Net
│  ├─ DualSight_Fire.py               # Dual-Sight Fire / StageFusion
│  └─ RGS_Net_V4_no_transformer.py    # 历史/消融文件（保留）
├─ exp/
│  ├─ train_scripts/                  # 训练脚本（单卡/torchrun DDP）
│  ├─ eval_scripts/                   # 评估脚本
│  └─ utils/                          # 可视化/预测等工具脚本
├─ data/
│  ├─ splits_activefire/              # ActiveFire 数据集划分 CSV（voting 等）
│  ├─ splits_land8fire/               # Land8Fire 数据集划分 CSV
│  ├─ splits_manual/                  # 手工标注相关划分
│  └─ ...                             # 数据处理与拆分脚本
├─ dataset/
│  ├─ activefire/                     # 数据组织：images/ masks/
│  ├─ Land8Fire/                      # 数据组织：images/ masks/
│  └─ mannual_annotations/            # 手工标注数据
└─ output/                            # 训练输出（权重、日志、曲线、评估报告）
```

## 2. 环境准备

推荐使用 conda（因为 `rasterio/gdal` 更稳定）。

```bash
conda env create -f environment.yml
conda activate fire
```

安装 PyTorch（按你的 CUDA 版本选择；以下示例为 CUDA 12.1）：

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

可选（仅当你需要 FLOPs/模型分析时）：

```bash
pip install fvcore
```

## 3. 数据准备与 CSV 格式

训练/评估脚本默认从 `--data-root` 指向的目录读取 `{algo}_train.csv / {algo}_val.csv / {algo}_test.csv`。

例如 ActiveFire voting 划分：

```
data/splits_activefire/
├─ voting_train.csv
├─ voting_val.csv
└─ voting_test.csv
```

CSV 格式：
- 第一行为表头（会被跳过）
- 每行至少两列：
	1) 影像路径（多波段 GeoTIFF 等，`rasterio` 可读）
	2) 掩膜路径（`rasterio` 可读；默认读取第 1 波段作为 [H,W] 掩膜）

波段说明：
- `dataset.py` 中 `bands` 以 1 开始计数（Landsat 10 个波段时，从 1 到 10）
- 训练脚本默认 `--bands 7 6 5`

## 4. 训练

所有训练脚本都支持：
- 单卡：直接 `python ...`
- 多卡 DDP：使用 `torchrun ...`（脚本会自动读取 `WORLD_SIZE/LOCAL_RANK`）

### 4.1 Dual-Sight Fire

单卡：

```bash
CUDA_VISIBLE_DEVICES=0 python exp/train_scripts/train_DualSight_Fire.py \
	--data-root data/splits_activefire --algo voting --bands 7 6 5
```

多卡：

```bash
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 --master_port=65531 \
	exp/train_scripts/train_DualSight_Fire.py --data-root data/splits_activefire --algo voting
```

输出目录默认：
- `output/DualSight_Fire/{algo}_YYYYmmddHHMM/`

### 4.2 Dual-Sight Fire StageFusion

单卡：

```bash
CUDA_VISIBLE_DEVICES=0 python exp/train_scripts/train_DualSight_Fire_stagefusion.py \
	--data-root data/splits_activefire --algo voting --bands 7 6 5
```

多卡：

```bash
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 --master_port=65530 \
	exp/train_scripts/train_DualSight_Fire_stagefusion.py --data-root data/splits_activefire --algo voting
```

输出目录默认：
- `output/DualSight_Fire_stagefusion/{algo}_YYYYmmddHHMM/`

### 4.3 其它模型

同样在 `exp/train_scripts/` 下提供：
- `train_baseline.py`
- `train_AttentionUNet.py`
- `train_UNetPlusPlus.py`
- `train_FDE_UNet.py`
- `train_FPS_U2Net.py`

它们的参数风格与 Dual-Sight Fire 基本一致（`--data-root/--algo/--bands/--epochs/--batch-size/...`）。

## 5. 评估

以 Dual-Sight Fire StageFusion 为例：

```bash
CUDA_VISIBLE_DEVICES=0 python exp/eval_scripts/eval_DualSight_Fire_stagefusion.py \
	--data-root data/splits_activefire --algo voting --bands 7 6 5 \
	--output-type fused --threshold 0.5 \
	--save-dir output/DualSight_Fire_stagefusion/voting_202512191353
```

默认会从 `--save-dir/weights/model_best.pth` 读取权重；也可用 `--model-path` 指定。

对应的其它评估脚本位于 `exp/eval_scripts/`：
- `eval_DualSight_Fire.py`
- `eval_baseline.py` / `eval_AttentionUNet.py` / `eval_UNetPlusPlus.py` / `eval_FDE_UNet.py` / `eval_FPS_U2Net.py`

## 6. 输出内容说明

训练输出目录通常包含：
- `weights/model_best.pth`：验证集最优权重
- `train_log.txt`：训练/验证 loss 记录
- `hyperparameters.txt`：超参数与模型统计
- `loss_curve.png`（如脚本生成）：loss 曲线

评估输出目录通常包含：
- `eval_*.txt`：Precision/Recall/F1/IoU/mIoU 等指标
- 可视化样例图（不同脚本生成的内容略有差异）

## 7. 许可证与引用

本项目仅用于学术研究与教学目的。若在论文或项目中使用，请引用本仓库并致谢作者。
