## RGS_Net v3.1 相较于 v3 的改进（中文说明）

本文档记录在实现 `RGS_Net_V3_1` 时对 `RGS_Net_V3` 做出的主要变更、动机、实现细节与训练/推理注意事项。目的是帮助维护者和复现实验的同学快速理解设计决策与工程要点。

### 概要

RGS_Net v3.1 在 v3 基础上做了 5 类主要改进，侧重于：更灵活的分割-重建融合、对背景信息的更好变换、在模型和训练中引入多尺度融合输出、明确的损失设计，以及一种 DDP 友好的渐进式多尺度监督策略。

### 主要改进点

1. 可学习的重建权重融合（learnable rec_weight）
   - 目的：让模型自己决定重建分支对分割 logits 的抑制权重，而不是使用固定因子。
   - 实现：在模型里加入 `rec_weight = nn.Parameter(torch.tensor(1.0))`。融合形式为：
     fused_logits = seg_logits - rec_weight * rec_logits_for_fusion
   - 注意：当重建输出有多通道时，先对重建 logits 在通道维度做平均（mean），以匹配分割 logits 的单通道概率格式。

2. 背景变换模块（BackgroundTransform）改进
   - 目的：用轻量可学习变换替代简单的 flip/normalize 操作，从而给重建分支/融合提供更强的表征。
   - 实现：在编码器输出/特征上先做归一化（LayerNorm/InstanceNorm 等），随后用 1x1 Conv（Conv1x1）进行通道变换（投影），替代之前的手工变换。

3. 多尺度融合输出（multi-scale fused outputs）
   - 目的：使模型在多个尺度上都产生分割 / 重建的融合输出，便于在训练中进行多尺度监督以及在推理中获得更稳健的结果。
   - 实现细节：模型在 forward 中返回多尺度的 `seg_logits`, `rec_logits` 以及 `multi_scale_seg_logits`、`multi_scale_rec_logits`。为了避免训练期间 DDP 的“参数未收到梯度”问题，`multi_scale_fused_logits`（多尺度的 fused 输出）只在推理时（`model.eval()`）额外计算并返回；训练过程中仍会计算多尺度的分割/重建损失（见第 5 点）。

4. 损失函数明确化
   - 分割损失：使用 `SpatialFocalTverskyLoss`（结合了 focal 的难例关注和 Tversky 的不平衡处理）。
   - 重建损失：使用 `MaskedL1Loss`（仅对有效区域 / mask 生效）。
   - 融合损失（fused_loss）：对主尺度的 fused_logits 也计算分割损失，以确保 `rec_weight` 等融合参数有梯度更新。

5. 渐进式多尺度监督（DDP 安全的实现）
   - 问题背景：最初设计是按阶段逐步启用更多尺度的监督（比如前若干 epoch 只监督粗尺度），但在 DDP 下会导致某些尺度对应的参数在某些 epoch 没有参与反向传播，从而触发 `Expected to have finished reduction` 的错误（参数未收到梯度）。
   - 解决策略：始终对所有尺度计算损失，但根据训练阶段应用缩放权重以实现“渐进式效果”。
     - 举例：三阶段权重为 [0.1, 0.5, 1.0]，在 stage0（早期）给细尺度较小权重，stage2（晚期）给全部尺度权重 1。这样既保留了渐进式的训练信号，又保证所有参数在每个 epoch 都会收到梯度，避免 DDP 同步问题。

### 与 v3 的工程/代码差异（文件位置与要点）

- 模型实现：`models/RGS_Net_V3_1.py`
  - 新增 `rec_weight` 可学习参数。
  - 将 reconstruction decoder 的输出通道改为与输入通道一致（n_channels），并在融合时取平均到单通道用于融合。
  - 增加返回 `multi_scale_seg_logits`、`multi_scale_rec_logits`（训练用）以及 `multi_scale_fused_logits`（仅在推理时返回）。

- 训练脚本：`exp/train_scripts/train_RGS_Net_V3_1.py`
  - 导入并使用 `SpatialFocalTverskyLoss`（seg）和 `MaskedL1Loss`（rec）。
  - 计算 `fused_loss`（主尺度），并把它加入总损失，确保 `rec_weight` 有梯度。
  - 实现“始终计算所有尺度损失 + 阶段性缩放权重”的渐进式多尺度监督逻辑（通过 epoch 判断阶段并设定缩放系数，如 0.1 / 0.5 / 1.0）。
  - 为调试 DDP 问题在调试时短暂使用 `DistributedDataParallel(..., find_unused_parameters=True)`，但最终保留的是上面的训练策略以避免长期依赖该标志。

### 训练与推理注意事项

- 训练复现
  - 主训练脚本：`exp/train_scripts/train_RGS_Net_V3_1.py`。
  - 分布式启动示例（单机单卡或多卡按需）：

    ```bash
    # 单机单卡示例
    torchrun --nproc_per_node=1 exp/train_scripts/train_RGS_Net_V3_1.py --config ...
    ```

  - 在配置中注意 multi-scale 阶段的 epoch 划分（例如：stage1_end, stage2_end）以及对应的尺度权重。

- 推理与多尺度 fused 输出
  - 若需要多尺度融合输出（`multi_scale_fused_logits`），在推理时将模型置为 eval 模式：

    ```python
    model.eval()
    out = model(input)
    # out.multi_scale_fused_logits 在 eval 时会返回列表
    ```

  - 训练中为了稳定与 DDP 兼容，`multi_scale_fused_logits` 在 `model.train()` 下不会被额外计算以避免引入不必要的参数路径（训练仍会计算 seg/rec 的多尺度损失，从而保证参数参与梯度）。

### DDP / 同步问题回顾

在调试过程中曾遇到 `RuntimeError: Expected to have finished reduction in the prior iteration before starting a new one...`，定位到原因是某些参数在部分 epoch 没有参与反向传播（unused parameters），导致下一轮 reduction sync 失败。为解决该问题做了两方面工作：

1. 临时使用 `find_unused_parameters=True` 进行排查，定位哪些参数未收到梯度；
2. 在训练逻辑上调整为“始终计算所有尺度损失 + 阶段性缩放权重”，确保各尺度的 out_conv 等参数在每个 epoch 都有梯度流。该方案在功能上仍实现渐进式训练效果，而且在 DDP 下安全可靠。

### 性能与调优记录（简要）

- DataLoader 调优：尝试过不同 `NUM_WORKERS` 与 `prefetch_factor` 配置以平衡主机 I/O 与 GPU 计算。最终在当前数据与模型配置下，训练表现为计算主导（data_ms ≈ 0 ms, comp_ms ≈ 700 ms，约 1.03 s/it），说明数据读取不是瓶颈。
- AMP：曾短期试验自动混合精度（autocast + GradScaler），但在本项目配置下对稳定性影响不一，实验后暂时回退为普通精度训练（如需可再尝试更细粒度的 AMP 调优）。

### 小结与后续建议

- 已实现的核心目标：
  - 可学习的融合权重（`rec_weight`）、背景变换模块、多尺度输出与训练时的稳健多尺度监督、明确的损失组合（seg/rec/fused）。

- 后续可选工作（建议清单）：
  1. 对 `rec_weight` 做范围或正则约束（例如 clamp 或 L2）以观察融合行为是否更稳定。 
  2. 对多尺度权重做超参搜索（例如更多阶段或不同权重谱）以优化收敛与最终分割质量。
  3. 若训练在多卡分布式下使用，建议做小规模多卡一致性测试（保证在不同卡数下 loss/metric 曲线一致性）。
  4. 考虑对数据读取做简单缓存或采用更高效的存储格式（如预先切片的 numpy/torch tensor），在硬件不同的机器上可进一步提升吞吐。

### 参考与变更记录

- 主要修改文件（实现要点）: 
  - `models/RGS_Net_V3_1.py` — 模型结构、rec_weight、multi-scale 返回项 
  - `exp/train_scripts/train_RGS_Net_V3_1.py` — 多尺度监督训练逻辑、fused_loss、训练参数调优
  - `dataset.py` / 数据相关脚本 — (若有) 数据读取或预处理的小调整

最后，如果你希望我把这份文档翻译为英文版本、添加到 README 的“Models” 小节，或者为每个改动生成一个简短的单元测试/集成检查脚本，我可以继续实现这些扩展。

---
（文档自动生成于仓库变更后，用于快速回顾 RGS_Net_v3.1 的设计与工程实现）
