"""RGS-Net V3_1

基于 V3 的改进版本，做出以下核心调整：
- 融合逻辑添加可学习权重参数：fused_logits = seg_logits - self.rec_weight * rec_logits
- 特征翻转前做归一化处理，特征翻转替换为 `bg_feat = Conv1x1(Norm(feat))` 学习型背景变换
- 解码器添加多尺度的融合特征输出
- 损失函数：分割分支使用SpatialFocalTverskyLoss，重建分支使用MaskedL1Loss
- 训练过程中使用多尺度损失监督，渐进式多尺度训练
"""

from __future__ import annotations

import os
import sys

# 确保项目根目录在 sys.path 中，以便导入顶层模块（如 utils）
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

Tensor = torch.Tensor


class DoubleConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, batchnorm: bool = True) -> None:
        super().__init__()
        layers: List[nn.Module] = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels) if batchnorm else nn.Identity(),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels) if batchnorm else nn.Identity(),
            nn.ReLU(inplace=True),
        ]
        self.block = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.block(x)


class Down(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, batchnorm: bool) -> None:
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv = DoubleConv(in_channels, out_channels, batchnorm=batchnorm)

    def forward(self, x: Tensor) -> Tensor:
        return self.conv(self.pool(x))


class Up(nn.Module):
    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
        batchnorm: bool,
        use_bilinear: bool,
    ) -> None:
        super().__init__()
        if use_bilinear:
            self.up: nn.Module = nn.Sequential(
                nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
                nn.Conv2d(in_channels, out_channels, kernel_size=1),
            )
        else:
            self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = DoubleConv(out_channels + skip_channels, out_channels, batchnorm=batchnorm)

    def forward(self, x: Tensor, skip: Tensor) -> Tensor:
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            diff_y = skip.size(-2) - x.size(-2)
            diff_x = skip.size(-1) - x.size(-1)
            x = F.pad(x, [diff_x // 2, diff_x - diff_x // 2, diff_y // 2, diff_y - diff_y // 2])
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x: Tensor) -> Tensor:
        return self.conv(x)


class SharedEncoderUNet(nn.Module):
    def __init__(
        self,
        in_channels: int,
        base_filters: int,
        depth: int,
        batchnorm: bool,
        dropout: float,
    ) -> None:
        super().__init__()
        if depth < 2:
            raise ValueError("depth 至少为 2")

        filters = [base_filters * (2 ** i) for i in range(depth)]
        self.stem = DoubleConv(in_channels, filters[0], batchnorm=batchnorm)
        self.down_blocks = nn.ModuleList()
        for idx in range(1, depth):
            self.down_blocks.append(Down(filters[idx - 1], filters[idx], batchnorm=batchnorm))

        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: Tensor) -> List[Tensor]:
        feats: List[Tensor] = []
        out = self.stem(x)
        feats.append(out)
        for down in self.down_blocks:
            out = self.dropout(down(out))
            feats.append(out)
        return feats


class SegmentationDecoderUNetV3_1(nn.Module):
    """V3_1版本的分割解码器，支持多尺度特征输出"""
    def __init__(
        self,
        feature_channels: List[int],
        out_channels: int,
        batchnorm: bool,
        dropout: float,
        use_bilinear: bool,
    ) -> None:
        super().__init__()
        self.ups: List[nn.Module] = nn.ModuleList()
        self.out_convs: List[OutConv] = nn.ModuleList()
        
        # 创建多尺度输出层
        for i in range(len(feature_channels)):
            # 对应不同尺度的输出
            self.out_convs.append(OutConv(feature_channels[i], out_channels))
            
        ups: List[nn.Module] = []
        for idx in range(len(feature_channels) - 1, 0, -1):
            in_ch = feature_channels[idx]
            skip_ch = feature_channels[idx - 1]
            out_ch = feature_channels[idx - 1]
            ups.append(Up(in_ch, skip_ch, out_ch, batchnorm=batchnorm, use_bilinear=use_bilinear))
        self.up_blocks = nn.ModuleList(ups)
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        self.final_out_conv = OutConv(feature_channels[0], out_channels)

    def forward(self, features: List[Tensor]) -> Tuple[Tensor, List[Tensor]]:
        # 多尺度特征输出
        multi_scale_outputs = []
        for i, feat in enumerate(features):
            multi_scale_outputs.append(self.out_convs[i](feat))
        
        # 标准上采样路径
        x = features[-1]
        for idx, block in enumerate(self.up_blocks):
            skip = features[-2 - idx]
            x = block(x, skip)
        final_output = self.final_out_conv(self.dropout(x))
        
        return final_output, multi_scale_outputs


class BackgroundTransform(nn.Module):
    """学习型背景变换模块"""
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.norm = nn.BatchNorm2d(in_channels)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        
    def forward(self, x: Tensor) -> Tensor:
        # 先归一化再通过1x1卷积
        x = self.norm(x)
        return self.conv(x)


class ReconstructionDecoderUNetV3_1(nn.Module):
    """V3_1版本的重建解码器"""
    def __init__(
        self,
        feature_channels: List[int],
        out_channels: int,
        batchnorm: bool,
        dropout: float,
        use_bilinear: bool,
    ) -> None:
        super().__init__()
        # 创建背景变换模块列表
        self.bg_transforms = nn.ModuleList([
            BackgroundTransform(ch, ch) for ch in feature_channels
        ])
        
        # 创建多尺度输出层
        self.out_convs = nn.ModuleList([
            OutConv(ch, out_channels) for ch in feature_channels
        ])
        
        ups: List[nn.Module] = []
        for idx in range(len(feature_channels) - 1, 0, -1):
            in_ch = feature_channels[idx]
            skip_ch = feature_channels[idx - 1]
            out_ch = feature_channels[idx - 1]
            ups.append(Up(in_ch, skip_ch, out_ch, batchnorm=batchnorm, use_bilinear=use_bilinear))
        self.up_blocks = nn.ModuleList(ups)
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        self.out_conv = OutConv(feature_channels[0], out_channels)

    def forward(self, features: List[Tensor]) -> Tuple[Tensor, List[Tensor]]:
        # 应用背景变换
        transformed_feats = []
        for feat, transform in zip(features, self.bg_transforms):
            transformed_feats.append(transform(feat))
        
        # 多尺度特征输出
        multi_scale_outputs = []
        for i, feat in enumerate(transformed_feats):
            multi_scale_outputs.append(self.out_convs[i](feat))
        
        # 标准上采样路径
        x = transformed_feats[-1]
        for idx, block in enumerate(self.up_blocks):
            skip = transformed_feats[-2 - idx]
            x = block(x, skip)
        final_output = self.out_conv(self.dropout(x))
        
        return final_output, multi_scale_outputs


@dataclass
class ForwardOutputV3_1:
    seg_logits: Tensor
    seg_mask: Tensor
    rec_logits: Tensor
    rec_mask: Tensor
    fused_logits: Tensor
    fused_mask: Tensor
    binary_mask: Tensor
    multi_scale_seg_logits: List[Tensor]  # 多尺度分割输出
    multi_scale_rec_logits: List[Tensor] = None  # 多尺度重建输出
    multi_scale_fused_logits: List[Tensor] = None  # 多尺度融合输出（seg - rec_weight * rec）
    
    def __post_init__(self):
        if self.multi_scale_rec_logits is None:
            self.multi_scale_rec_logits = []
        if self.multi_scale_fused_logits is None:
            self.multi_scale_fused_logits = []


class RGSNetV3_1(nn.Module):
    def __init__(
        self,
        n_channels: int = 3,
        n_classes: int = 1,
        n_filters: int = 64,
        depth: int = 5,
        batchnorm: bool = True,
        dropout: float = 0.1,
        use_bilinear: bool = False,
        binarize_threshold: float = 0.5,
    ) -> None:
        super().__init__()
        if n_classes != 1:
            raise ValueError("当前实现仅支持单通道火点掩膜")

        self.n_channels = n_channels
        self.binarize_threshold = binarize_threshold
        
        # 可学习的重建权重参数
        self.rec_weight = nn.Parameter(torch.tensor(1.0))

        self.encoder = SharedEncoderUNet(
            in_channels=n_channels,
            base_filters=n_filters,
            depth=depth,
            batchnorm=batchnorm,
            dropout=dropout,
        )

        feature_channels = [n_filters * (2 ** i) for i in range(depth)]
        self.seg_decoder = SegmentationDecoderUNetV3_1(
            feature_channels=feature_channels,
            out_channels=n_classes,
            batchnorm=batchnorm,
            dropout=dropout,
            use_bilinear=use_bilinear,
        )
        self.rec_decoder = ReconstructionDecoderUNetV3_1(
            feature_channels=feature_channels,
            out_channels=n_channels,
            batchnorm=batchnorm,
            dropout=dropout,
            use_bilinear=use_bilinear,
        )

    def forward(self, x: Tensor, *, binarize: bool = True) -> ForwardOutputV3_1:
        features = self.encoder(x)
        seg_logits, multi_scale_seg_logits = self.seg_decoder(features)
        rec_logits, multi_scale_rec_logits = self.rec_decoder(features)

        seg_mask = torch.sigmoid(seg_logits)
        rec_mask = torch.sigmoid(rec_logits)

        # 使用可学习权重的融合逻辑
        # rec_logits 是多通道(n_channels)，需要降维到1通道以便与 seg_logits(1通道) 融合
        rec_logits_for_fusion = rec_logits.mean(dim=1, keepdim=True)
        fused_logits = seg_logits - self.rec_weight * rec_logits_for_fusion
        fused_mask = torch.sigmoid(fused_logits)

        # 多尺度融合输出：推理时计算，训练时为空（避免DDP未使用参数问题）
        multi_scale_fused_logits: List[Tensor] = []
        if not self.training and multi_scale_seg_logits and multi_scale_rec_logits:
            # 推理模式：计算多尺度融合特征
            for s_logit, r_logit in zip(multi_scale_seg_logits, multi_scale_rec_logits):
                r_logit_for_fusion = r_logit.mean(dim=1, keepdim=True)
                multi_scale_fused_logits.append(
                    s_logit - self.rec_weight * r_logit_for_fusion
                )

        if binarize:
            binary_mask = (fused_mask >= self.binarize_threshold).float()
        else:
            binary_mask = fused_mask

        return ForwardOutputV3_1(
            seg_logits=seg_logits,
            seg_mask=seg_mask,
            rec_logits=rec_logits,
            rec_mask=rec_mask,
            fused_logits=fused_logits,
            fused_mask=fused_mask,
            binary_mask=binary_mask,
            multi_scale_seg_logits=multi_scale_seg_logits,
            multi_scale_rec_logits=multi_scale_rec_logits,
            multi_scale_fused_logits=multi_scale_fused_logits,
        )

    def compute_losses(
        self,
        *,
        seg_logits: Tensor,
        rec_logits: Tensor,
        multi_scale_seg_logits: List[Tensor],
        multi_scale_rec_logits: List[Tensor],
        targets_seg: Tensor,
        seg_loss_fn: Callable[[Tensor, Tensor], Tensor],
        recon_loss_fn: Optional[Callable[[Tensor, Tensor, Tensor], Tensor]] = None,
        targets_recon: Optional[Tensor] = None,
        inputs: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        targets_seg = self._ensure_channel_dim(targets_seg)
        targets_seg = targets_seg.float()

        if targets_recon is None:
            targets_recon = 1.0 - targets_seg
        else:
            targets_recon = self._ensure_channel_dim(targets_recon).float()
            
        if inputs is None:
            raise ValueError("inputs must be provided for reconstruction loss")

        # 主要分割损失
        seg_loss = seg_loss_fn(seg_logits, targets_seg)
        
        # 重建损失使用MaskedL1Loss
        if recon_loss_fn is not None:
            rec_loss = recon_loss_fn(rec_logits, inputs, targets_seg)
        else:
            raise ValueError("Reconstruction loss function must be provided")

        # 多尺度监督损失 - 分割分支
        multi_scale_seg_loss = 0.0
        for scale_logits in multi_scale_seg_logits:
            # 调整尺度以匹配目标大小
            if scale_logits.shape[2:] != targets_seg.shape[2:]:
                scale_logits = F.interpolate(scale_logits, size=targets_seg.shape[2:], mode='bilinear', align_corners=False)
            multi_scale_seg_loss += seg_loss_fn(scale_logits, targets_seg)
        
        # 平均多尺度分割损失
        if len(multi_scale_seg_logits) > 0:
            multi_scale_seg_loss = multi_scale_seg_loss / len(multi_scale_seg_logits)
            
        # 多尺度监督损失 - 重建分支
        multi_scale_rec_loss = 0.0
        for scale_logits in multi_scale_rec_logits:
            # 调整尺度以匹配输入大小
            if scale_logits.shape[2:] != inputs.shape[2:]:
                scale_logits = F.interpolate(scale_logits, size=inputs.shape[2:], mode='bilinear', align_corners=False)
            multi_scale_rec_loss += recon_loss_fn(scale_logits, inputs, targets_seg)
        
        # 平均多尺度重建损失
        if len(multi_scale_rec_logits) > 0:
            multi_scale_rec_loss = multi_scale_rec_loss / len(multi_scale_rec_logits)

        total_loss = seg_loss + rec_loss + multi_scale_seg_loss + multi_scale_rec_loss
        return {
            "loss": total_loss, 
            "seg_loss": seg_loss, 
            "rec_loss": rec_loss,
            "multi_scale_seg_loss": multi_scale_seg_loss,
            "multi_scale_rec_loss": multi_scale_rec_loss
        }

    @staticmethod
    def _ensure_channel_dim(mask: Tensor) -> Tensor:
        if mask.dim() == 3:
            return mask.unsqueeze(1)
        return mask


__all__ = ["RGSNetV3_1", "ForwardOutputV3_1"]