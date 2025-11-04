# SpatialFocalLoss 优化说明文档

## 优化概述

针对 RGS-Net V3 的双分支架构，对 `SpatialFocalLoss` 进行了优化，支持三种连通域加权策略，分别适用于不同的任务目标。

## 损失函数数学公式

### 基础 Focal Loss

标准 Focal Loss 定义为：

$$
\text{FL}(p_t) = -\alpha_t (1 - p_t)^\gamma \log(p_t)
$$

其中：
- $p_t = \begin{cases} p & \text{if } y = 1 \\ 1-p & \text{otherwise} \end{cases}$
- $p$: 模型预测的正类概率（经过 sigmoid）
- $y$: 真实标签（0 或 1）
- $\alpha_t$: 类别平衡权重
- $\gamma$: 聚焦参数，控制对困难样本的关注度

### Spatial Focal Loss（空间加权版本）

我们在 Focal Loss 的基础上引入空间权重 $w(x, y)$，得到：

$$
\text{SFL} = \frac{1}{N} \sum_{x,y} w(x, y) \cdot \text{FL}(p_t(x, y))
$$

展开为：

$$
\text{SFL} = \frac{1}{N} \sum_{x,y} w(x, y) \cdot \left[-\alpha_t (1 - p_t(x,y))^\gamma \log(p_t(x,y))\right]
$$

其中：
- $w(x, y)$: 位置 $(x, y)$ 的空间权重
- $N = \sum_{x,y} w(x, y)$: 归一化因子

### 空间权重计算

空间权重 $w(x, y)$ 基于连通域面积计算，分为三种策略：

#### 1. Small 策略（小目标优先）

$$
w(x, y) = \begin{cases}
w_{\min} + (w_{\max} - w_{\min}) \cdot \left(\frac{1/A_i}{\max_j(1/A_j)}\right)^{\gamma_w} & \text{if } (x,y) \in C_i \\
w_{\text{bg}} & \text{otherwise}
\end{cases}
$$

其中：
- $C_i$: 第 $i$ 个连通域
- $A_i$: 连通域 $C_i$ 的面积（像素数）
- $w_{\min}, w_{\max}$: 权重范围
- $\gamma_w$: 权重衰减幂次
- $w_{\text{bg}}$: 背景权重

**特点**：小面积连通域获得更高权重，$A_i \downarrow \Rightarrow w \uparrow$

#### 2. Large 策略（大目标优先）

$$
w(x, y) = \begin{cases}
w_{\min} + (w_{\max} - w_{\min}) \cdot \left(\frac{A_i}{\max_j A_j}\right)^{\gamma_w} & \text{if } (x,y) \in C_i \\
w_{\text{bg}} & \text{otherwise}
\end{cases}
$$

**特点**：大面积连通域获得更高权重，$A_i \uparrow \Rightarrow w \uparrow$

#### 3. Uniform 策略（无加权）

$$
w(x, y) = \begin{cases}
1.0 & \text{if foreground} \\
w_{\text{bg}} & \text{otherwise}
\end{cases}
$$

**特点**：所有前景像素权重相同，无连通域计算开销

### 完整计算流程

1. **预测概率**：
   $$p = \sigma(\text{logits})$$

2. **计算连通域**：
   使用 OpenCV 的 `connectedComponents` 算法标记连通域：
   $$\text{labels}, \text{num\_labels} = \text{connectedComponents}(\text{mask})$$

3. **统计面积**：
   $$A_i = |\{(x, y) : \text{labels}(x, y) = i\}|, \quad i = 1, 2, \ldots, \text{num\_labels}$$

4. **计算权重图**：
   根据选定策略（small/large/uniform）计算每个像素的权重 $w(x, y)$

5. **计算损失**：
   $$\text{SFL} = \frac{\sum_{x,y} w(x, y) \cdot \text{FL}(p_t(x, y))}{\sum_{x,y} w(x, y)}$$

### 参数配置示例

#### 分割分支（Small 策略）
- $\alpha = 0.8$（正样本权重）
- $\gamma = 1.5$（Focal Loss 聚焦参数）
- $w_{\min} = 1.0, w_{\max} = 4.0$（权重范围）
- $\gamma_w = 1.5$（权重衰减幂次）

#### 重建分支（Large 策略）
- $\alpha = 0.25$（降低正样本权重）
- $\gamma = 1.2$（较平滑的聚焦）
- $w_{\min} = 1.0, w_{\max} = 3.0$（权重范围）
- $\gamma_w = 1.0$（线性权重映射）

## 核心改进

### 1. 新增参数

#### `weight_strategy` (str, 默认: "small")
控制连通域加权策略：
- **"small"**: 小连通域获得更高权重（适合稀疏小目标，如火点检测）
- **"large"**: 大连通域获得更高权重（适合大面积区域，如背景重建）
- **"uniform"**: 不做连通域加权，前景统一权重 1.0（最快，适合对连通域大小不敏感的场景）

### 2. 策略实现细节

#### Small 策略（小目标优先）
```python
# 使用归一化反面积作为权重
inv_area = 1.0 / area
weights = weight_min + (weight_max - weight_min) * (inv_area ** weight_gamma)
```
- 小连通域 → 高权重
- 孤立火点被强调
- 适合分割分支

#### Large 策略（大目标优先）
```python
# 使用归一化面积作为权重
area_norm = area / max_area
weights = weight_min + (weight_max - weight_min) * (area_norm ** weight_gamma)
```
- 大连通域 → 高权重
- 大面积背景被强调
- 适合重建分支

#### Uniform 策略（无加权）
```python
# 前景统一权重，跳过连通域计算
weights = 1.0 for foreground
weights = background_weight for background
```
- 最快速度（~1ms）
- 不依赖连通域大小
- 适合编码器或快速实验

### 3. 性能优化

移除了跨 batch 的权重缓存机制（存在维度不匹配风险），改为：
- 每个 batch 重新计算权重
- 使用 OpenCV 的高效连通域算法（GPU → CPU → GPU）
- uniform 策略跳过连通域计算

**重要更新**：已移除下采样功能（`weight_downsample_factor`），始终使用原始分辨率计算连通域权重。这确保了对火点像素只有 1～5 个的小样本能够准确计算连通域面积，避免下采样导致的连通域丢失问题。

## RGS-Net V3 模型适配

### 架构分析
```
输入图像
   ↓
共享编码器 (无独立损失)
   ├→ 分割解码器 → seg_logits → seg_loss (small策略)
   └→ 重建解码器 → rec_logits → rec_loss (large策略)
```

### 训练脚本配置

```python
# 分割分支：强调小火点
seg_criterion = SpatialFocalLoss(
    weight_strategy="small",        # 小连通域更高权重
    alpha=0.8,                      # 标准 Focal Loss 正样本权重
    gamma=1.5,                      # 聚焦困难样本
    weight_min=1.0,
    weight_max=4.0,                 # 最小火点权重可达 4.0
    weight_gamma=1.5,               # 控制权重随面积衰减的幂次
)

# 重建分支：强调大背景
recon_criterion = SpatialFocalLoss(
    weight_strategy="large",        # 大连通域更高权重
    alpha=0.25,                     # 降低正样本权重，避免背景主导
    gamma=1.2,                      # 较低 gamma，更平滑的损失
    weight_min=1.0,
    weight_max=3.0,                 # 大背景区域权重可达 3.0
    weight_gamma=1.0,               # 线性权重映射
)
```

## 性能基准测试

### 测试环境
- 批大小: 32
- 图像尺寸: 256×256
- 设备: CUDA GPU
- 配置: 使用原始分辨率（无下采样）

### 结果
| 策略 | 耗时 (ms) | 相对 Uniform | 特点 |
|------|-----------|--------------|------|
| Small | ~1973 | ~2064x | 精确小目标权重 |
| Large | ~2023 | ~2116x | 精确大目标权重 |
| Uniform | ~0.96 | 1.0x | 无连通域计算 |

### 分析
1. **Uniform 最快**：跳过连通域计算，仅需 ~1ms
2. **Small/Large 策略较慢**：使用原始分辨率计算连通域，OpenCV CPU 计算耗时约 2s/batch
3. **精度保证**：去除下采样后，对 1～5 像素的小火点能够准确计算连通域面积

对于完整训练：
- 每个 batch 总损失计算时间: seg_loss(~2s) + rec_loss(~2s) ≈ **4s/batch**
- 相比 FocalTverskyLoss (~5ms)，增加约 **800倍**
- **性能瓶颈**：OpenCV 的连通域算法在 CPU 上执行，GPU ↔ CPU 数据传输和 CPU 计算是主要开销

### 性能权衡建议
鉴于当前性能开销较大，建议根据实际场景选择：

1. **精度优先**（当前配置）：
   - 使用 small/large 策略
   - 适合小目标密集场景
   - 训练时间较长（~4s/batch）

2. **速度优先**：
   - 使用 uniform 策略
   - 适合快速实验
   - 训练时间极短（~1ms/batch）

3. **平衡方案**（未来优化）：
   - 实现 CUDA kernel 加速连通域计算
   - 或使用混合策略（训练初期 uniform，后期切换 small/large）

## 使用建议

### 1. 标准配置（推荐）
```python
# 训练脚本中已配置
seg_criterion = SpatialFocalLoss(weight_strategy="small")
recon_criterion = SpatialFocalLoss(weight_strategy="large")
```

### 2. 追求速度（快速实验）
```python
# 两个分支都用 uniform（跳过连通域计算）
seg_criterion = SpatialFocalLoss(weight_strategy="uniform")
recon_criterion = SpatialFocalLoss(weight_strategy="uniform")
```

### 3. 精度优先（小目标场景）
```python
# 默认配置已经是精度优先（原始分辨率）
seg_criterion = SpatialFocalLoss(
    weight_strategy="small",
    weight_min=1.0,
    weight_max=5.0,  # 增加权重范围，更强调小目标
    weight_gamma=2.0  # 更陡峭的权重曲线
)
```

## 参数调优指南

### weight_min / weight_max
- 控制权重的取值范围
- 分割分支：`[1.0, 4.0]` - 给小火点 4倍权重
- 重建分支：`[1.0, 3.0]` - 给大背景 3倍权重

### weight_gamma
- 控制权重随面积变化的速率
- 值越大，小/大目标的权重差异越明显
- 分割分支：1.5（强调小目标）
- 重建分支：1.0（线性映射）

### alpha / gamma
- Focal Loss 的标准参数
- 分割分支：alpha=0.8, gamma=1.5（标准配置）
- 重建分支：alpha=0.25, gamma=1.2（避免背景主导）

## 小目标场景的特殊考虑

### 问题：火点像素只有 1～5 个
数据集中存在大量样本（256×256）中火点像素只有 1～5 个，这对连通域计算提出了特殊要求：

### 解决方案：去除下采样
- **原问题**：下采样会导致极小连通域（1～5像素）在降采样后完全丢失
- **当前方案**：始终使用原始分辨率计算连通域权重
- **效果**：确保即使是单个像素的火点也能被准确识别和加权

### 性能权衡
- **计算时间**：约 4s/batch（OpenCV CPU 计算 + GPU↔CPU 数据传输）
- **精度提升**：小目标连通域面积计算准确，无丢失风险
- **适用场景**：火点检测等极小目标分割任务
- **限制**：训练速度较慢，不适合需要快速迭代的场景

### 替代方案
如果性能开销不可接受，可以考虑：
1. **使用 uniform 策略**：牺牲连通域加权，换取极快的训练速度
2. **使用标准 Focal Loss**：如 `FocalTverskyLoss`，无空间加权但速度快
3. **混合训练**：初期用快速损失函数，后期切换到精确的 SpatialFocalLoss

## 关于编码器

您提到"uniform策略应用于编码器"。需要说明的是：

**RGS-Net V3 的编码器是共享的**，没有独立的损失函数。编码器通过两个解码器的损失反向传播来训练：
```
总损失 = seg_loss + rec_loss
       = SpatialFocalLoss(small) + SpatialFocalLoss(large)
```

编码器同时学习：
- 分割任务需要的小目标特征（来自 seg_loss）
- 重建任务需要的大背景特征（来自 rec_loss）

如果您希望编码器有独立的正则化损失，可以考虑：
1. 添加编码器特征的重建损失
2. 添加对比学习损失
3. 添加特征一致性约束

但这需要修改模型架构，不在当前优化范围内。

## 验证与测试

### 快速功能测试
```bash
python test_spatial_strategies.py
```

### 完整训练测试
```bash
# 单卡训练（测试 2-3 个 epoch）
RANK=0 WORLD_SIZE=1 LOCAL_RANK=0 python exp/train_scripts/train_RGS_Net_V3.py
```

### 评估模型
```bash
# 评估已训练模型
RANK=0 WORLD_SIZE=1 LOCAL_RANK=0 python exp/eval_scripts/eval_RGS_Net_V3.py \
  --batch-size 64 \
  --save-dir output/RGS_Net_V3/voting_YYYYMMDDHHMM
```

## 总结

### 改进点
1. ✅ 支持三种策略：small / large / uniform
2. ✅ 移除跨 batch 缓存，避免维度问题
3. ✅ 使用 OpenCV 高效连通域算法加速
4. ✅ 针对 V3 模型双分支优化配置
5. ✅ 去除下采样，确保小目标（1～5像素）准确计算

### 效果预期
- **分割分支**: 对小火点更敏感，召回率提升，无连通域丢失
- **重建分支**: 对大背景更关注，重建质量提升
- **训练速度**: 约 4s/batch，相比 FocalTverskyLoss 慢约 800倍
- **适用场景**: 小目标检测精度要求高，可接受较长训练时间的场景

### 性能优化建议
如果训练速度是关键瓶颈，建议：
1. **使用 uniform 策略**：跳过连通域计算，速度提升 2000 倍
2. **等待 CUDA 实现**：未来可通过 CUDA kernel 加速连通域计算
3. **混合策略**：训练初期用 uniform 快速收敛，后期切换 small/large 精调

### 后续优化方向
1. 考虑 CUDA kernel 加速连通域计算（需要自定义算子）
2. 探索自适应权重策略（根据连通域分布自动选择）
3. 动态调整权重范围（训练过程中自适应）

---

文档版本: 2.0  
更新日期: 2025-10-30  
作者: GitHub Copilot  
更新说明: 移除下采样功能，确保小目标（1～5像素火点）的连通域准确计算
