## RGS-Net V3_1 改进说明

本文档总结并说明 `RGS-Net V3_1` 在本代码库中的主要改进、实现位置、使用说明和实验建议，便于复现、调参与后续扩展。

### 一、核心改进一览
- 可学习的重建权重参数 `rec_weight`：
  - 位置：`models/RGS_Net_V3_1.py` 中 `RGSNetV3_1`。
  - 功能：融合逻辑由原来的 `fused_logits = seg_logits - rec_logits` 更新为 `fused_logits = seg_logits - self.rec_weight * rec_logits`，`rec_weight` 为可学习的 `nn.Parameter`（初始化为 1.0），训练/评估过程会记录该参数。

- 学习型背景变换（替代特征翻转）：
  - 位置：`models/RGS_Net_V3_1.py` 中 `BackgroundTransform` 与 `ReconstructionDecoderUNetV3_1`。
  - 实现：在重建解码器中对每个尺度的特征先做归一化（BatchNorm），再用 1x1 卷积变换：`bg_feat = Conv1x1(Norm(feat))`。这样替换了原来的“特征翻转”操作，使背景变换成为可学习层。

- 解码器输出多尺度特征并返回多尺度 logits：
  - 位置：`SegmentationDecoderUNetV3_1` 与 `ReconstructionDecoderUNetV3_1`。
  - 说明：两个解码器均在每个尺度上添加了 `OutConv` 输出层，forward 返回 `(final_output, multi_scale_outputs)`，便于多尺度监督或可视化。

- 多尺度融合输出（新增）：
  - 位置：`RGSNetV3_1.forward` 已生成 `multi_scale_fused_logits`，即对每尺度的 seg/logits 与 rec/logits 应用相同融合规则：`s_logit - rec_weight * r_logit`，并作为 `ForwardOutputV3_1` 的字段返回。

- 损失函数与多尺度监督：
  - 分割分支使用 `SpatialFocalTverskyLoss`（见 `loss.py`）。
  - 重建分支使用 `MaskedL1Loss`（见 `loss.py`）。
  - 模型实现并支持多尺度监督（训练脚本在主/多尺度上均计算 seg 与 rec 的损失并累加）。

### 二、训练相关（进阶与渐进式多尺度训练）
- 训练脚本：`exp/train_scripts/train_RGS_Net_V3_1.py`。
  - 已支持多尺度损失的计算并将其加入总损失（seg + rec + multi_scale_seg + multi_scale_rec）。
  - 为避免早期多尺度监督引入震荡，已实现“渐进式多尺度训练（Scheme A）”：
    - 新增配置 `MULTI_SCALE_START_EPOCH`（默认 20），在 `epoch < MULTI_SCALE_START_EPOCH` 时跳过多尺度损失计算；在 `epoch >= MULTI_SCALE_START_EPOCH` 时启用。
    - 此实现已写入训练脚本并记录在超参数文件。

建议（可选改进）：
- 若希望更平滑地引入多尺度监督，推荐使用权重热身（Scheme B）：在 `MULTI_SCALE_START_EPOCH` 后用线性/余弦曲线将多尺度权重从 0 增加到 1。该方案有利于避免引入时的突变。
- 若希望更细粒度控制，可按尺度分阶段启用（coarse → fine），或为每尺度分别设置独立权重。

### 三、使用指南（快速示例）
- 前向输出示例：
```py
outputs = model(images, binarize=False)
# 主尺度：
fused = outputs.fused_logits            # 融合后的 logits
fused_prob = torch.sigmoid(fused)

# 多尺度融合：
for i, s in enumerate(outputs.multi_scale_fused_logits):
    prob = torch.sigmoid(s)
    up = F.interpolate(prob, size=images.shape[2:], mode='bilinear', align_corners=False)
    # 可视化或保存 up
```

- 在训练中如果要将多尺度 fused logits 也作为监督目标（可选）：
  - 在训练循环中类似 `multi_scale_seg_logits`/`multi_scale_rec_logits` 的处理，将 `multi_scale_fused_logits` 按需 resize 后传入分割损失（或专门对 fused 做损失）。

### 四、代码位置索引
- 模型：`models/RGS_Net_V3_1.py`（encoder/decoder/background transform/forward/compute_losses）
- 训练：`exp/train_scripts/train_RGS_Net_V3_1.py`（损失实例化、训练循环、多尺度损失、渐进式配置）
- 损失实现：`loss.py`（`SpatialFocalTverskyLoss`, `MaskedL1Loss`）

### 五、实验建议（消融/超参）
- 基线：始终不开启多尺度监督（作为对照）。
- Scheme A（开/关）：尝试 `MULTI_SCALE_START_EPOCH` = 10、20、50，比较最终 val IoU 与收敛稳定性。
- Scheme B（热身）：`start=5, warmup=25`（线性增长权重）。可与不同的 seg/recon 权重配合探索。
- 监控项：主尺度 seg/rec loss、多尺度 seg/rec loss（每尺度）、rec_weight 的变化（是否收敛到 >1/ <1），以及 val IoU/F1/precision/recall。

---

如需我：
- 将 Scheme B（权重热身）直接实现到训练脚本，并同时记录每 epoch 的 multi-scale 权重与 per-scale loss；或
- 添加 eval/visualization 脚本以保存多尺度融合预测（方便做定性比较）。

请选择你希望我接着做的项（例如“实现 Scheme B 热身”或“添加可视化脚本”）。
