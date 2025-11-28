"""RGS-Net V4 (TransformerBackgroundBlock ablation).

该变体用于消融实验：保留RGS-Net V4的整体编码-双解码结构，但将
Reconstruction decoder 中的 TransformerBackgroundBlock 替换为恒等映射，
从而评估窗口注意力背景建模的贡献。
"""

from __future__ import annotations

import torch.nn as nn

from models.RGS_Net_V4 import RGSNetV4


class RGSNetV4NoTransformer(RGSNetV4):
    """RGS-Net V4 的消融版本：移除 TransformerBackgroundBlock。"""

    def __init__(
        self,
        n_channels: int = 3,
        n_classes: int = 1,
        base_filters: int = 64,
        dropout: float = 0.1,
        batchnorm: bool = True,
    ) -> None:
        super().__init__(
            n_channels=n_channels,
            n_classes=n_classes,
            base_filters=base_filters,
            dropout=dropout,
            batchnorm=batchnorm,
        )
        # 用恒等映射替换背景变换模块，彻底消融 TransformerBackgroundBlock。
        self.bg_transforms = nn.ModuleList([nn.Identity() for _ in range(5)])


__all__ = ["RGSNetV4NoTransformer"]
