# Fire-Segmentation: RGS-Net V3 火点分割（PyTorch）

面向 Landsat 遥感影像的火点分割模型与训练/评估脚本，主力模型为 RGS-Net V3：共享编码器、分割与重建双分支，利用空间重建误差引导分割，兼顾鲁棒性与精度。

## 亮点
- 共享编码器 + 双解码器（分割/重建），在复杂背景下更稳
- 空间重建误差引导（Spatial guidance）提升难样本判别力
- 支持分布式训练（torchrun + DDP），开箱即用
- 提供批量训练与批量评估脚本，便于超参搜索与结果对比
- 清晰的输出目录结构与日志/可视化产物

## 仓库结构
```
├─ dataset.py                        # 数据集定义：LandsatFireDataset
├─ loss.py                           # 损失：FocalTverskyLoss、SpatialFocalLoss 等
├─ utils.py                          # 常用工具：adaptive_crop 等
├─ models/
│  ├─ baseline.py                    # 基线模型（UNet 类）
│  ├─ RGS_Net_V1.py                  # RGS-Net V1
│  ├─ RGS_Net_V2.py                  # RGS-Net V2
│  └─ RGS_Net_V3.py                  # RGS-Net V3（推荐）
├─ exp/
│  ├─ train_scripts/
│  │  ├─ train_RGS_Net_V3.py         # 单次训练脚本（DDP）
│  │  └─ multi_run_train_RGS_Net_V3.py # 批量训练脚本（多组超参串行跑）
│  └─ eval_scripts/
│     ├─ eval_RGS_Net_V3.py          # 单次评估脚本
│     └─ multi_eval_RGS_Net_V3.py    # 批量评估脚本（汇总报告）
├─ data/
│  ├─ split_data_activefire.py       # 数据拆分工具（如需）
│  ├─ unzip_data_activefire.py       # 数据解压工具（如需）
│  └─ splits_activefire/             # 训练/验证/测试 CSV
├─ dataset/activefire/               # 原始数据组织（如已提供）
└─ output/                           # 训练输出与评估报告
```

## 数据准备
训练与评估脚本默认从 `data/splits_activefire/` 读取 CSV：

```
data/splits_activefire/
├─ voting_train.csv
├─ voting_val.csv
└─ voting_test.csv
```

CSV 每行通常包含图像与掩膜路径。可通过 `dataset.py` 的 `LandsatFireDataset` 的 `bands` 参数选择用于训练的波段（默认 `(7,6,5)`）。

## 环境准备
建议 Python 3.8+，安装依赖：

```bash
pip install -r requirements.txt
```

PyTorch 请依据你的 CUDA/OS 在官网选择命令安装：https://pytorch.org/get-started/locally/

## 快速开始

### 单次训练（RGS-Net V3）
脚本：`exp/train_scripts/train_RGS_Net_V3.py`

常用参数（在脚本顶部常量中设置）：
- 数据与算法：`DATA_ROOT = "data/splits_activefire"`，`ALGORITHM = "voting"`
- 波段：`BANDS = (7, 6, 5)`
- 训练：`BATCH_SIZE = 64`，`EPOCHS = 200`，`LEARNING_RATE = 3e-4`
- 日志/保存：`SAVE_DIR = output/RGS_Net_V3/<algo>_<时间戳>`（脚本会自动创建）

运行示例（2 张 GPU）：
```bash
CUDA_VISIBLE_DEVICES=1,2 torchrun --nproc_per_node=2 exp/train_scripts/train_RGS_Net_V3.py
```

训练输出（位于 `output/RGS_Net_V3/` 下，示例）：
- `weights/`：`model_best.pth`、周期性保存的权重
- `logs/`：各 rank 日志
- `loss_curves.png` 等可选可视化

### 批量训练（多组超参串行）
脚本：`exp/train_scripts/multi_run_train_RGS_Net_V3.py`

1) 在脚本内的 `EXPERIMENTS` 列表中定义多组实验（每组包含名称、描述和超参）。
2) 运行：
```bash
CUDA_VISIBLE_DEVICES=1,2 python3 exp/train_scripts/multi_run_train_RGS_Net_V3.py
```

产物：
- `output/RGS_Net_V3/multi_run/` 目录下，每个实验生成一个独立子目录（含 `weights/`、`logs/`、`experiment_config.json`）
- 汇总文件：`experiments_summary.json`，以及 `multi_run_log.txt`

### 单次评估
脚本：`exp/eval_scripts/eval_RGS_Net_V3.py`

将 `--save-dir` 指向训练输出目录（包含 `weights/model_best.pth`）：
```bash
python3 exp/eval_scripts/eval_RGS_Net_V3.py \
  --batch-size 64 \
  --num-workers 4 \
  --algo voting \
  --save-dir output/RGS_Net_V3/voting_YYYYMMDDHHMM
```

产物：
- 指标：`eval_model_best.txt`（微平均 Precision/Recall/F1）
- 可视化：`model_best_prediction.png`、随机样例图保存在 `pred_samples/`

提示：如需分布式评估，可用 torchrun（确保设置 `MASTER_ADDR/MASTER_PORT` 或直接使用单进程）。

### 批量评估（对批量训练产物批量评估并汇总）
脚本：`exp/eval_scripts/multi_eval_RGS_Net_V3.py`

默认读取 `output/RGS_Net_V3/multi_run/experiments_summary.json` 并逐个评估：
```bash
CUDA_VISIBLE_DEVICES=1 python3 exp/eval_scripts/multi_eval_RGS_Net_V3.py
```

或自定义汇总文件：
```bash
CUDA_VISIBLE_DEVICES=1 python3 exp/eval_scripts/multi_eval_RGS_Net_V3.py \
  --summary-file output/RGS_Net_V3/multi_run/experiments_summary.json
```

产物：
- 文本对比报告：`evaluation_comparison.txt`（按 F1 排序）
- JSON 报告：`evaluation_results.json`

## 配置与可调参数小抄
损失（参见 `loss.py` 与训练脚本注入参数）：
- 分割分支：SpatialFocalLoss（`SEG_*` 参数）
  - 常用：`SEG_ALPHA`、`SEG_GAMMA`、`SEG_WEIGHT_STRATEGY`（small/large）
- 重建分支：SpatialFocalLoss（`REC_*` 参数）
  - 常用：`REC_ALPHA`、`REC_GAMMA`、`REC_WEIGHT_STRATEGY`

调优建议（目标不同优先级不同）：
- 提高精确率（减少误报）：提高 `*_GAMMA`、增大背景权重（如 `background_weight`）、后处理删除小连通域、提升推理阈值
- 提高召回率（减少漏报）：适当增大 `SEG_ALPHA`，降低推理阈值，放宽小连通域过滤

## 常见问题与排错
- NCCL/分布式初始化错误（如缺少 MASTER_ADDR）：
  - 单进程运行评估：`python3 eval_RGS_Net_V3.py ...`（不使用 torchrun）
  - 或设置：`MASTER_ADDR=127.0.0.1 MASTER_PORT=29500 RANK=0 WORLD_SIZE=1 LOCAL_RANK=0`
- 找不到日志文件：脚本会自动创建 `output/...` 目录；若自定义 `SAVE_DIR`，请确保有写权限
- 权重路径：默认从 `SAVE_DIR/weights/model_best.pth` 读取
- CUDA OOM：降低 `BATCH_SIZE`、减少 GPU 进程数，或使用更小裁剪/分辨率
- CSV 路径：确保为有效可读路径；必要时使用绝对路径

## 许可证与引用
本项目仅用于学术研究与教学目的。若在论文或项目中使用，请引用本仓库并致谢作者。

—— Happy Segmenting 🔥
