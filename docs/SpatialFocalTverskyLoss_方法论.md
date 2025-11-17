## Spatial FocalTversky Loss 方法论

本文给出 Spatial FocalTversky Loss 在二类火点分割中的完整表述，包括 Tversky 指数、focal 机制、空间加权及其组合形式，并说明关键超参数的含义与建议取值范围。

### 1. 记号与输出

- $y \in \{0,1\}^{H\times W}$：二值 GT 掩膜（1=fire，0=background）。
- $p \in [0,1]^{H\times W}$：模型输出的火点概率图（由 logits 经 sigmoid 或 softmax 得到）。
- 空间加权图 $w \in [w_{min}, w_{max}]^{H\times W}$：依据目标区域形状/面积/连通性等构造的像素权重（实现细节见 3 节）。

### 2. Tversky 指数与损失

定义带权 Tversky 指数：

$$
\mathrm{TI}(p,y) = \frac{\sum_{i} w_i\, p_i y_i}{\sum_{i} w_i\, p_i y_i + \alpha\sum_{i} w_i\, p_i (1-y_i) + \beta\sum_{i} w_i\, (1-p_i) y_i}.
$$

其中 $(\alpha,\beta)$ 控制 FP 与 FN 的相对惩罚，常用如 $\alpha=0.6,\;\beta=0.4$（更侧重召回）。对应的 Tversky 损失：

$$
\mathcal{L}_{\mathrm{tversky}} = 1 - \mathrm{TI}(p,y).
$$

为增强对难例的聚焦，可使用“Focal Tversky”形式：

$$
\mathcal{L}_{\mathrm{focal\_tversky}} = \big(1 - \mathrm{TI}(p,y)\big)^{\gamma}, \quad \gamma>1.
$$

在实现中，往往将两者线性组合（见第 4 节）。

### 3. 空间加权（Spatial Weighting）

为提升对小目标、边界和稀疏火点的敏感度，我们使用空间加权：

1) 基本范围：$w_i \in [w_{min}, w_{max}]$，默认 $w_{min}=1.0$、$w_{max}=3.0$。

2) 依据连通域面积 $A$ 的幂律：对 $y=1$ 的像素，

$$
w_i = \min\Big(w_{max},\; w_{min} + c\cdot A^{-\eta}\Big), \quad \eta = \text{area\_gamma} > 0.
$$

面积越小权重越高（强调小火点）。$c$ 为归一化常数，使 $w$ 落入区间。对背景像素可乘以系数 $\lambda_{bg}$（`background_weight`），便于平衡背景贡献。

3) 边界/细节：也可选择对边界像素（由形态学梯度或距离变换得到）提高 $w$，本实现提供参数以控制该趋势。

以上构成的 $w$ 用于所有像素级项的加权求和。

### 4. 与 Focal 的组合形式

实际实现采用 Tversky 与 Focal 的加权和：

$$
\mathcal{L}_{seg}(p,y) = \lambda_{\mathrm{t}}\, \big(1-\mathrm{TI}(p,y)\big)\; +\; \lambda_{\mathrm{f}}\, \mathcal{L}_{\mathrm{focal}}(p,y),
$$

其中（以二分类为例）

$$
\mathcal{L}_{\mathrm{focal}}(p,y) = -\frac{1}{N}\sum_{i} w_i\,\Big[\; \alpha_f y_i (1-p_i)^{\gamma_f}\log p_i 
\;+\; (1-\alpha_f) (1-y_i) p_i^{\gamma_f}\log (1-p_i) \;\Big].
$$

超参数说明：
- $\lambda_{\mathrm{t}},\lambda_{\mathrm{f}}$：两部分的权重（如 0.7/0.3）。
- $\alpha_f$：正负样本平衡（如 0.85），$\gamma_f$：focal 指数（如 1.6）。
- $\alpha,\beta$：Tversky 的 FP/FN 权衡（如 0.6/0.4）。
- $w_{min}, w_{max}, \eta(=\text{area\_gamma}), \lambda_{bg}$：空间加权相关。

### 5. 训练细节与实践建议

- 数值稳定：对 $p$ 做裁剪 $p\leftarrow \operatorname{clip}(p, \epsilon, 1-\epsilon)$，避免 $\log 0$。
- 类别极度不均衡：提高 $\beta$（加强对 FN 的惩罚）或提高 $\alpha_f$、$\gamma_f$。
- 小目标占比高：增大 $w_{max}$ 或 area\_gamma，以加强对小连通域的权重。
- 背景伪影明显：适当提高 $\lambda_{bg}$ 抑制背景的虚假响应。

### 6. 与 DualSight-Fire 的配合

- 主尺度：$\mathcal{L}_{seg}$ 同时用于 $\hat{y}_{seg}$ 与 $\hat{y}_{fused}$，后者保证融合参数（如 $w_{rec}$）获得梯度。
- 多尺度：对 $\{\hat{y}_{seg}^{(s)}\}$ 逐尺度计算后取平均，并乘以阶段权重（见 DualSight-Fire 文档）。

### 7. 总结

Spatial FocalTversky Loss 通过“空间加权 + Tversky 的不均衡建模 + Focal 的难例聚焦”三者结合，使得在“极度稀疏、面积跨度大”的火点检测场景中，模型对小火点与边界更敏感，对难例更关注，同时维持对 FP/FN 的可调平衡。

—— 以上方法在代码中对应：`loss.py` 中的 `SpatialFocalTverskyLoss`，训练脚本示例见 `exp/train_scripts/train_RGS_Net_V3_1.py`。
