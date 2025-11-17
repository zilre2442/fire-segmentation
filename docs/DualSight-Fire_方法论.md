## DualSight-Fire（RGS_Net v3.1）方法论

本文系统阐述 DualSight-Fire（即 RGS_Net v3.1）的设计思路、网络结构、损失与监督策略以及训练/推理要点。其核心思想是“分割-重建双视角 + 可学习融合 + 多尺度稳健监督”。

### 1. 设计动机（Why）

- 单一分割分支在极度稀疏且不均衡的火点检测中易受噪声/伪影影响。
- 引入“重建视角”可在输入空间重构背景，从而提供一种对异常（火点）更敏感的负证据；再以可学习权重与分割结果融合，提高鲁棒性。
- 多尺度监督可增强不同尺度下的可分性，但需避免分布式训练（DDP）中“未参与反传”的参数同步问题。

### 2. 网络结构（What）

- 共享编码器（Shared Encoder）提取金字塔特征。
- 双解码器：
  - 分割解码器（Seg Decoder）输出 $\hat{y}_{seg}$ 以及多尺度 logits 列表 $\{\hat{y}_{seg}^{(s)}\}$。
  - 重建解码器（Rec Decoder）输出 $\hat{X}_{rec}$（通道数与输入一致），以及多尺度列表 $\{\hat{X}_{rec}^{(s)}\}$。
- 背景变换模块（Background Transform）：对中间特征施加规范化 + 1×1 卷积投影，形成轻量可学习的背景表征，替代人工 flip/normalize。
- 融合层（Fusion）：以可学习权重将分割与重建信息在 logit 空间融合。

#### 2.1 融合机制

重建分支输出与输入通道一致（如 $C$ 个波段）。为与单通道分割 logits 对齐，先对重建输出在通道维做平均：

$$
\hat{y}_{rec}^{(1)} = \operatorname{mean}_{c}(\hat{X}_{rec}[c]) \in \mathbb{R}^{1\times H \times W}.
$$

再定义可学习参数 $w_{rec} \in \mathbb{R}$，最终融合 logits：

$$
\hat{y}_{fused} = \hat{y}_{seg} - w_{rec} \cdot \hat{y}_{rec}^{(1)}.
$$

注：记号“−”体现“重建得好→更像背景→降低火点置信”的直觉；$w_{rec}$ 由数据自适应学习。

#### 2.2 多尺度输出

- 训练：模型提供 $\{\hat{y}_{seg}^{(s)}\}, \{\hat{X}_{rec}^{(s)}\}$ 用于多尺度损失；为避免 DDP 未用参数，训练期不额外计算多尺度 fused 输出。
- 推理：可在 eval 模式下额外导出 $\{\hat{y}_{fused}^{(s)}\}$ 作为分析与可视化。

### 3. 监督与损失（How）

总体损失（主尺度）由三部分组成：

$$
\mathcal{L}_{main} = \mathcal{L}_{seg}(\hat{y}_{seg}, y)\; +\; \mathcal{L}_{rec}(\hat{X}_{rec}, X, y)\; +\; \mathcal{L}_{seg}(\hat{y}_{fused}, y).
$$

- $\mathcal{L}_{seg}$：Spatial FocalTversky Loss（下文另文档详述）。
- $\mathcal{L}_{rec}$：Masked L1（仅在 $y=1$ 或有效 mask 区域计入）。
- $\mathcal{L}_{seg}(\hat{y}_{fused}, y)$：确保 $w_{rec}$ 等融合参数获得梯度并参与优化。

多尺度损失按尺度平均，再乘以阶段权重 $\lambda_{ms}(t)$：

$$
\mathcal{L}_{ms} = \lambda_{ms}(t) \cdot \Big[\; \frac{1}{S}\sum_{s} \mathcal{L}_{seg}(\hat{y}_{seg}^{(s)}, y)\; +\; \frac{1}{S}\sum_{s} \mathcal{L}_{rec}(\hat{X}_{rec}^{(s)}, X, y)\; \Big].
$$

最终：$\mathcal{L} = \mathcal{L}_{main} + \mathcal{L}_{ms}$。

### 4. 渐进式多尺度监督（DDP 友好）

为实现“早期弱多尺度、后期强多尺度”的课程式训练，同时避免 DDP 的“未完成约简”错误，采用“所有 epoch 始终计算全尺度损失，但使用阶段权重衰增”的策略：

$$
\lambda_{ms}(t) = \begin{cases}
0.1, & t < T_1\\
0.5, & T_1 \le t < T_2\\
1.0, & t \ge T_2
\end{cases}
$$

这样每个 epoch 全部尺度分支都有梯度，避免“未用参数”，同时保留渐进式效果。

### 5. 训练与推理要点（Notes）

- 建议损失组合：$\mathcal{L}_{seg}$ 采用 Spatial FocalTversky；$\mathcal{L}_{rec}$ 采用 Masked L1；主尺度 fused 也计算分割损失。
- DDP：若调试需要可临时启用 `find_unused_parameters=True` 定位问题；稳定训练依赖于“始终全尺度监督 + 阶段权重”。
- 早停：可在进入阶段三后才启用早停计数，避免早期波动干扰（代码已实现）。
- 超参参考：$w_{rec}$ 从 1.0 初始化，无需显式约束；多尺度权重分段值可按数据与收敛速度微调。

### 6. 误差模式与边界情况（Edge Cases）

- 极低火点样本（1–3 px）：$\mathcal{L}_{seg}$ 的 focal 成分有助于抑制易例，聚焦难例。
- 大面积火点：Tversky 的 $(\alpha,\beta)$ 让 FN/FP 惩罚可调，减少类别不均衡影响。
- 背景伪影：重建分支学到“正常背景”更容易，$\hat{y}_{rec}^{(1)}$ 高时抑制 $\hat{y}_{seg}$，降低误检。

### 7. 与 v3 的关键差异

- 引入可学习融合权重 $w_{rec}$，替代固定系数。
- 背景变换由“手工处理”变为“Norm+Conv1x1”的可学习模块。
- 训练期稳健的多尺度监督；推理可导出多尺度 fused 供分析。
- 融合分支参与主尺度损失，确保融合参数持续更新。

—— 以上方法在代码中主要对应：`models/RGS_Net_V3_1.py` 与 `exp/train_scripts/train_RGS_Net_V3_1.py`。
