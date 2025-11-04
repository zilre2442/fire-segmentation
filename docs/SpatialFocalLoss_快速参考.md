# SpatialFocalLoss 快速参考指南

## 重要更新（v2.0）

**已移除下采样功能**，始终使用原始分辨率计算连通域权重，确保对 1～5 像素的小火点能够准确计算。

## 损失函数公式（简要）

### Spatial Focal Loss
$$
\text{SFL} = \frac{1}{N} \sum_{x,y} w(x, y) \cdot \left[-\alpha_t (1 - p_t)^\gamma \log(p_t)\right]
$$

### 空间权重策略
- **Small**: $w \propto (1/A_i)^{\gamma_w}$ （小连通域高权重）
- **Large**: $w \propto (A_i)^{\gamma_w}$ （大连通域高权重）
- **Uniform**: $w = 1.0$ （前景统一权重）

其中 $A_i$ 是连通域 $i$ 的面积（像素数）。

详细数学推导请参考 [完整文档](./SpatialFocalLoss优化说明.md#损失函数数学公式)。

## 性能对比

| 策略 | 速度 | 精度 | 适用场景 |
|------|------|------|----------|
| **small** | 慢 (~2s/batch) | 高 | 小目标检测（火点） |
| **large** | 慢 (~2s/batch) | 高 | 大区域重建（背景） |
| **uniform** | 极快 (~1ms/batch) | 中 | 快速实验 |

## 基本使用

### 1. 标准配置（推荐）
```python
from loss import SpatialFocalLoss

# 分割分支：强调小火点
seg_criterion = SpatialFocalLoss(
    weight_strategy="small",
    alpha=0.8,
    gamma=1.5,
    weight_min=1.0,
    weight_max=4.0,
    weight_gamma=1.5,
)

# 重建分支：强调大背景
recon_criterion = SpatialFocalLoss(
    weight_strategy="large",
    alpha=0.25,
    gamma=1.2,
    weight_min=1.0,
    weight_max=3.0,
    weight_gamma=1.0,
)
```

### 2. 快速实验（速度优先）
```python
# 两个分支都用 uniform
seg_criterion = SpatialFocalLoss(weight_strategy="uniform")
recon_criterion = SpatialFocalLoss(weight_strategy="uniform")
```

## 关键参数说明

### weight_strategy
- `"small"`: 小连通域高权重（适合火点）
- `"large"`: 大连通域高权重（适合背景）
- `"uniform"`: 不加权（最快）

### weight_min / weight_max
- 控制权重范围
- 建议：分割 [1.0, 4.0]，重建 [1.0, 3.0]

### weight_gamma
- 控制权重随面积变化的速率
- 值越大，权重差异越明显
- 建议：分割 1.5，重建 1.0

### alpha / gamma (Focal Loss 参数)
- alpha: 正样本权重
- gamma: 聚焦困难样本
- 建议：分割 (0.8, 1.5)，重建 (0.25, 1.2)

## 性能权衡

### 当前配置（去除下采样）
- ✅ **优点**：小目标（1～5像素）连通域准确计算，无丢失
- ❌ **缺点**：训练速度慢（~4s/batch，约为 FocalTverskyLoss 的 800 倍）

### 建议
1. **精度优先**：使用 small/large 策略（当前配置）
2. **速度优先**：使用 uniform 策略
3. **平衡方案**：初期 uniform，后期切换 small/large

## 常见问题

### Q: 为什么去除下采样？
**A:** 数据集中火点像素只有 1～5 个，下采样会导致小连通域完全丢失。

### Q: 训练太慢怎么办？
**A:** 可以：
1. 使用 `uniform` 策略（速度提升 2000 倍）
2. 使用其他损失函数（如 `FocalTverskyLoss`）
3. 混合策略：初期 uniform，后期 small/large

### Q: 如何选择策略？
**A:**
- **小目标检测**：small
- **大区域重建**：large
- **快速实验**：uniform

## 完整示例

```python
import torch
from loss import SpatialFocalLoss

# 创建损失函数
criterion = SpatialFocalLoss(
    weight_strategy="small",  # 小目标优先
    alpha=0.8,
    gamma=1.5,
    weight_min=1.0,
    weight_max=4.0,
    weight_gamma=1.5,
)

# 使用
pred = model(inputs)  # [B, 1, H, W]
target = labels       # [B, 1, H, W]
loss = criterion(pred, target)
loss.backward()
```

## 性能测试

运行以下命令测试不同策略的性能：
```bash
python test_spatial_strategies.py
```

## 更多信息

详细文档请参考：[SpatialFocalLoss优化说明.md](./SpatialFocalLoss优化说明.md)

---
最后更新: 2025-10-30  
版本: 2.0
