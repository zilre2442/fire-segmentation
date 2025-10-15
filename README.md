# RGS-Net：基于重建误差引导的火点分割（PyTorch）

本项目实现了一个共享编码器的双分支 U-Net：
- 分割分支输出火点概率图；
- 重建分支复原输入影像，利用重建误差作为“异常线索”门控分割概率，从而提升鲁棒性与精度。

## 特色
- 共享编码器 + 双解码器（分割 / 重建）
- 误差引导融合：`fused_probs = sigmoid(|x-\hat{x}|/tau) × seg_probs`
- 重建损失仅在背景上计算，避免强行还原火点像素
- torchrun 启动的分布式训练（DDP）
- 评估仅保留微平均 Precision/Recall/F1（更干净、可复现）
- 超参日志包含“分割损失 + 重建损失”的完整配置

## 目录结构
```
├─ data/
│  └─ full/
│     ├─ <ALGO>_train.csv
│     ├─ <ALGO>_val.csv
│     └─ <ALGO>_test.csv             # 每行：(image_path, mask_path)
├─ models/
│  ├─ RGS_Net.py                      # 模型定义（类名：RGSNet）
│  └─ baseline.py                     # 基线 UNet（参考）
├─ dataset.py                         # LandsatFireDataset（raster 读取）
├─ loss.py                            # FocalTverskyLoss、MaskedL1Loss 等
├─ utils.py                           # analyze_model_performance、adaptive_crop
├─ train_RGS_Net.py                   # 分布式训练入口
├─ eval_RGS_Net.py                    # 评估（仅微平均指标）
└─ output/                            # 训练输出
```

## 数据格式
`dataset.py` 期望 CSV 文件包含表头，且每行两列：
```
image_path,mask_path
/path/to/image.tif,/path/to/mask.tif
...
```
- 通过 `bands` 参数选择波段（示例常用 (7,6,5)）；
- Landsat 16-bit 影像会按 65535 归一化到 [0,1]。

## 环境安装
项目依赖：PyTorch、Rasterio、NumPy、Matplotlib、TQDM。

1）安装 PyTorch（根据你的 CUDA/OS 选择官方命令）：
- https://pytorch.org/get-started/locally/

2）安装其余依赖：
```bash
pip install rasterio numpy matplotlib tqdm
```

可选：若使用性能分析（CUDA profiler），请确保 CUDA 环境可用。

## 模型概览
文件：`models/RGS_Net.py`
- 类：`RGSNet`
- 前向输出（结构体）：
  - `seg_logits`：分割分支 logits
  - `seg_probs`：分割概率
  - `reconstruction`：重建结果 \(\hat{x}\)
  - `recon_error`：重建误差 \(|x-\hat{x}|\)
  - `fused_probs`：融合后的概率

融合过程（片段）：
```
recon_scalar = recon_error.mean(dim=1, keepdim=True)
weighting = torch.sigmoid(recon_scalar / tau)
fused = seg_probs * weighting
```

损失函数（见 `loss.py`）：
- 分割：`FocalTverskyLoss`（训练脚本默认）
- 重建：`MaskedL1Loss`（仅对背景像素计算）

## 训练（DDP）
脚本：`train_RGS_Net.py`

关键参数：
- `DATA_ROOT = data/full`
- `ALGORITHM`（CSV 前缀，可选：Kumar-Roy / Murphy / Schroeder / intersection / voting）
- `BANDS = (7, 6, 5)`
- `BATCH_SIZE = 128`，`EPOCHS = 200`，`LEARNING_RATE = 3e-4`
- 重建损失权重：`RECON_LOSS_WEIGHT = 1.0`
- 融合温度：`TAU = 1.0`

使用 4 张 GPU 训练示例：
```bash
torchrun --nproc_per_node=4 train_RGS_Net.py
```
输出（仅 rank 0 写入）：
- `output/RGS_Net/<ALGO>_<时间戳>/`
  - `model_best.pth`、`model_final.pth`、按间隔保存的 checkpoints
  - `hyperparameters.txt`（记录模型、优化器，以及“分割/重建损失”的配置）
  - `logs/`（各进程日志）

备注：
- 训练使用 `DistributedSampler`，必须通过 torchrun 启动以注入 RANK/WORLD_SIZE/LOCAL_RANK。
- 脚本包含早停与 ReduceLROnPlateau 学习率调度。
- 可通过全局参数启用 `adaptive_crop()`（缩放式课程学习）。

## 评估
脚本：`eval_RGS_Net.py`
- 设置 `SAVE_DIR` 指向训练输出目录（如 `output/RGS_Net/voting_YYYYMMDDHHMM`）。
- 是否使用融合：`USE_FUSION=True/False`；温度参数 `TAU` 可调。

运行：
```bash
python eval_RGS_Net.py
```
输出：
- `eval_<model_name>.txt`（微平均指标）
- 同时会保存一份小样本预测可视化 `<model_name>_prediction.png`

注意：评估脚本默认从 `SAVE_DIR/weights/model_best.pth` 加载；若你的权重保存在 `SAVE_DIR/model_best.pth`，请自行：
- 将权重复制到 `SAVE_DIR/weights/` 下；或
- 修改评估脚本中的 `param_path` 指向实际文件。

## 快速自检
使用 `utils.py` 的工具快速查看模型推理与显存：
```python
from models.RGS_Net import RGSNet
from utils import analyze_model_performance

analyze_model_performance(
    model=RGSNet(n_channels=3, n_filters=32),
    input_shape=(1, 3, 256, 256),
    device='cpu'
)
```

## 常见问题
- Rasterio/GDAL 安装：建议使用 conda-forge 渠道或参考你平台的 GDAL 安装指南。
- CSV 路径找不到：检查 CSV 内是否为绝对路径；`dataset.py` 会校验文件存在。
- CUDA OOM：降低 `BATCH_SIZE`，减少并行进程，或启用裁剪。
- DDP 卡住：务必用 `torchrun` 启动，并确保所有进程看到相同的数据与 CSV。
