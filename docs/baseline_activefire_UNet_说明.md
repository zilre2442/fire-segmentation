# ActiveFire UNet 基线复现说明

## 来源与目标
本文件复现 `activefire-main/src/train/voting/unet_64f_2conv_762` 目录中的 Keras UNet (64 基础通道，两层卷积/块，带 Dropout & Skip Connection)。目的：在当前 fire-segmentation 项目中建立统一的 PyTorch 训练/评估基线，方便与 RGS 系列模型对比。

## 与原实现的差异
- 框架：Keras -> PyTorch。
- 输出：原模型最后使用 `sigmoid`；复现模型 `ActiveFireUNetBaseline` 输出 **logits**，训练用 `BCEWithLogitsLoss`（更数值稳定）。评估脚本再 `sigmoid + 阈值`。
- 数据：统一使用本项目的 `LandsatFireDataset`（按 `--algorithm` 和 `--fire-category` 选择 CSV）。
- Loss 可配置：`bce`, `focal_tversky`, `spatial_focal_tversky`，默认 `bce` 保持基线简单。
- 训练：支持分布式 (torchrun + DDP)，与现有 RGS 脚本风格一致，但不含多尺度/重建分支逻辑。

## 文件结构
- 模型：`models/activefire_unet_baseline.py`
- 训练：`exp/train_scripts/train_activefire_baseline_unet.py`
- 评估：`exp/eval_scripts/eval_activefire_baseline_unet.py`
- 文档：`docs/baseline_activefire_UNet_说明.md`

## 模型结构
Encoder (下采样 4 层) + Bottleneck + Decoder (上采样 4 层)：
```
Input -> [64]x2 -> [128]x2 -> [256]x2 -> [512]x2 -> [1024]x2 -> Up512 -> Up256 -> Up128 -> Up64 -> 1x1 Conv
```
每个块包含：Conv(3x3)-BN-ReLU -> Conv(3x3)-BN-ReLU。
下采样：MaxPool2d(2) + Dropout。
上采样：ConvTranspose2d + Concatenate Skip + Dropout + 双卷积块。

## 训练示例
单机多卡 (4 GPU)：
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 exp/train_scripts/train_activefire_baseline_unet.py \
  --algorithm voting --epochs 80 --batch-size 16 --lr 1e-3 --base-filters 64 --loss bce
```
指定火点类别：
```bash
torchrun --nproc_per_node=2 exp/train_scripts/train_activefire_baseline_unet.py \
  --algorithm voting --fire-category few --epochs 60
```
单卡调试：
```bash
RANK=0 WORLD_SIZE=1 LOCAL_RANK=0 python3 exp/train_scripts/train_activefire_baseline_unet.py --epochs 2 --batch-size 4
```

## 评估示例
```bash
eval_weights=output/ActiveFireBaseline/voting_202511241234/weights/model_best.pth
python3 exp/eval_scripts/eval_activefire_baseline_unet.py --algorithm voting --weights $eval_weights --batch-size 32
```
输出指标：Precision / Recall / F1 / Fire IoU / Back IoU / mIoU / Dice。

## 阈值与后处理
- 默认阈值 0.5（可通过 `--threshold` 调整）。
- 若需与原 activefire 设定 (`TH_FIRE=0.25`) 对齐，可传 `--threshold 0.25`。

## 与 RGS 系列对比建议
| 项目 | 特征融合 | 多尺度 | 重建分支 | 特殊损失 | 训练复杂度 |
|------|----------|--------|----------|----------|------------|
| ActiveFire Baseline | 单路径 UNet | 无 | 无 | 可选 Focal/Tversky | 低 |
| RGS V3.x | Seg + Rec 双分支 | 有 | 有 | SpatialFocalTversky + MaskedL1 | 高 |

## 后续可扩展点
1. 添加早停与调度策略 (当前仅 `ReduceLROnPlateau`)。  
2. 引入 OHEM 或区域加权提升小火点检测。  
3. 迁移更多 activefire 模型变体（small / smaller）。  
4. 加入推理脚本生成与原项目相同的文本数组输出格式。  

## 假设与验证
- 数据尺寸已与 UNet 要求兼容；如出现尺寸不匹配，脚本自动插值到掩膜大小。  
- `LandsatFireDataset` 已返回归一化后的张量；若需与原项目 MAX_PIXEL_VALUE=65535 归一化策略完全同步，可在 dataset 中加入模式标志。  

## 常见问题
1. **显存不足**：调整 `--base-filters 32` 或减小 `--batch-size`。  
2. **Loss 不下降**：尝试 `--loss focal_tversky` 或降低初始学习率。  
3. **评估指标偏低**：检查波段选择是否与训练一致，必要时尝试阈值扫描。  

## 快速阈值扫描脚本示例
```bash
for th in 0.25 0.35 0.5 0.6; do
  python3 exp/eval_scripts/eval_activefire_baseline_unet.py \
    --algorithm voting --weights $eval_weights --threshold $th --batch-size 32
done
```

---
若需扩展更多 activefire 模型或添加多尺度特征输出，请提出需求。