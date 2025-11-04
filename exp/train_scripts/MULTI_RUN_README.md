# RGS-Net V3 多次训练脚本使用指南

## 功能说明

`multi_run_train_RGS_Net_V3.py` 是一个批量训练脚本，可以自动执行多个不同超参数配置的训练任务。

**主要功能：**
- ✅ 依次执行多个训练任务，每个任务使用不同的超参数
- ✅ 自动记录每次训练的配置和对应的输出目录
- ✅ 生成详细的日志和汇总报告
- ✅ 支持训练中断后继续执行
- ✅ 每次训练都有独立的输出目录和配置文件

## 使用方法

### 1. 配置实验参数

编辑 `multi_run_train_RGS_Net_V3.py` 中的 `EXPERIMENTS` 列表，添加你想要运行的实验配置：

```python
EXPERIMENTS = [
    {
        "name": "exp1_baseline",
        "description": "基线配置",
        "params": {
            "BATCH_SIZE": 64,
            "LEARNING_RATE": 3e-4,
            "EPOCHS": 200,
            "EARLY_STOPPING_PATIENCE": 10,
            # 分割分支 SpatialFocalLoss 参数
            "SEG_WEIGHT_STRATEGY": "small",
            "SEG_ALPHA": 0.75,
            "SEG_GAMMA": 2.0,
            "SEG_WEIGHT_MIN": 1.0,
            "SEG_WEIGHT_MAX": 3.0,
            "SEG_WEIGHT_GAMMA": 1.5,
            # 重建分支 SpatialFocalLoss 参数
            "REC_WEIGHT_STRATEGY": "large",
            "REC_ALPHA": 0.75,
            "REC_GAMMA": 1.5,
            "REC_WEIGHT_MIN": 1.0,
            "REC_WEIGHT_MAX": 4.0,
            "REC_WEIGHT_GAMMA": 1.5,
        }
    },
    # 添加更多实验配置...
]
```

### 2. 运行批量训练

使用以下命令启动批量训练：

```bash
CUDA_VISIBLE_DEVICES=1,2 python3 exp/train_scripts/multi_run_train_RGS_Net_V3.py
```

### 3. 查看结果

所有结果保存在 `output/RGS_Net_V3/multi_run/` 目录下：

```
output/RGS_Net_V3/multi_run/
├── multi_run_log.txt              # 批量训练总日志
├── experiments_summary.json       # 所有实验的汇总信息（JSON格式）
├── exp1_baseline_202510311000/    # 第一个实验的输出目录
│   ├── experiment_config.json     # 该实验的配置
│   ├── hyperparameters.txt        # 详细的超参数记录
│   ├── weights/                   # 模型权重
│   ├── logs/                      # 训练日志
│   └── loss_curves.png            # 损失曲线图
├── exp2_higher_lr_202510311030/   # 第二个实验的输出目录
│   └── ...
└── temp_scripts/                  # 临时生成的训练脚本
    ├── train_exp1_baseline.py
    └── train_exp2_higher_lr.py
```

## 可配置的超参数

### 基本训练参数
- `BATCH_SIZE`: 批大小（默认 64）
- `LEARNING_RATE`: 学习率（默认 3e-4）
- `EPOCHS`: 训练轮数（默认 200）
- `EARLY_STOPPING_PATIENCE`: 早停耐心值（默认 10）

### 分割分支 SpatialFocalLoss 参数
- `SEG_WEIGHT_STRATEGY`: 权重策略 ("small" 或 "large")
- `SEG_ALPHA`: Focal Loss alpha 参数
- `SEG_GAMMA`: Focal Loss gamma 参数
- `SEG_WEIGHT_MIN`: 最小权重
- `SEG_WEIGHT_MAX`: 最大权重
- `SEG_WEIGHT_GAMMA`: 权重 gamma

### 重建分支 SpatialFocalLoss 参数
- `REC_WEIGHT_STRATEGY`: 权重策略
- `REC_ALPHA`: Focal Loss alpha 参数
- `REC_GAMMA`: Focal Loss gamma 参数
- `REC_WEIGHT_MIN`: 最小权重
- `REC_WEIGHT_MAX`: 最大权重
- `REC_WEIGHT_GAMMA`: 权重 gamma

## 实验汇总文件格式

`experiments_summary.json` 包含所有实验的详细信息：

```json
{
    "total_experiments": 3,
    "successful_experiments": 2,
    "failed_experiments": 1,
    "gpu_devices": "1,2",
    "num_gpus": 2,
    "experiments": [
        {
            "name": "exp1_baseline",
            "description": "基线配置",
            "output_dir": "output/RGS_Net_V3/multi_run/exp1_baseline_202510311000",
            "config_file": "output/RGS_Net_V3/multi_run/exp1_baseline_202510311000/experiment_config.json",
            "success": true,
            "duration_seconds": 3600.5,
            "start_time": "2025-10-31T10:00:00",
            "end_time": "2025-10-31T11:00:00",
            "params": { ... }
        }
    ],
    "generated_at": "2025-10-31T15:30:00"
}
```

## 常见使用场景

### 1. 学习率搜索
```python
EXPERIMENTS = [
    {"name": "lr_1e4", "params": {"LEARNING_RATE": 1e-4, ...}},
    {"name": "lr_3e4", "params": {"LEARNING_RATE": 3e-4, ...}},
    {"name": "lr_5e4", "params": {"LEARNING_RATE": 5e-4, ...}},
]
```

### 2. Focal Loss 参数调优
```python
EXPERIMENTS = [
    {"name": "gamma_1.5", "params": {"SEG_GAMMA": 1.5, ...}},
    {"name": "gamma_2.0", "params": {"SEG_GAMMA": 2.0, ...}},
    {"name": "gamma_3.0", "params": {"SEG_GAMMA": 3.0, ...}},
]
```

### 3. 批大小实验
```python
EXPERIMENTS = [
    {"name": "bs32", "params": {"BATCH_SIZE": 32, ...}},
    {"name": "bs64", "params": {"BATCH_SIZE": 64, ...}},
    {"name": "bs128", "params": {"BATCH_SIZE": 128, ...}},
]
```

## 注意事项

1. **GPU 资源**: 确保 GPU 有足够的显存运行配置的批大小
2. **磁盘空间**: 每次训练会保存模型权重，确保有足够的磁盘空间
3. **时间安排**: 多次训练可能需要很长时间，建议在后台运行
4. **日志监控**: 可以通过 `tail -f output/RGS_Net_V3/multi_run/multi_run_log.txt` 实时查看进度

## 后台运行

使用 `nohup` 或 `tmux` 在后台运行：

```bash
# 使用 nohup
nohup CUDA_VISIBLE_DEVICES=1,2 python3 exp/train_scripts/multi_run_train_RGS_Net_V3.py > multi_run.log 2>&1 &

# 使用 tmux
tmux new -s multi_train
CUDA_VISIBLE_DEVICES=1,2 python3 exp/train_scripts/multi_run_train_RGS_Net_V3.py
# 按 Ctrl+B 然后 D 分离会话
```

## 中断与恢复

如果训练中断，脚本会在 `experiments_summary.json` 中记录已完成的实验。你可以：
1. 查看汇总文件了解已完成的实验
2. 修改 `EXPERIMENTS` 列表，移除已完成的实验
3. 重新运行脚本继续未完成的训练
