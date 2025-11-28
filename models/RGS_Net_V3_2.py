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
        # 修改：输出层对应解码器上采样后的特征通道数
        for idx in range(len(feature_channels) - 1, 0, -1):
            self.out_convs.append(OutConv(feature_channels[idx - 1], out_channels))
            
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
        # 多尺度特征输出（前N-1个上采样块 + 最终的seg_logits）
        multi_scale_outputs = []
        
        # 标准上采样路径
        x = features[-1]
        for idx, block in enumerate(self.up_blocks):
            skip = features[-2 - idx]
            x = block(x, skip)
            # 对前N-1个上采样块进行多尺度输出
            if idx < len(self.up_blocks) - 1:
                multi_scale_outputs.append(self.out_convs[idx](x))
            
        final_output = self.final_out_conv(self.dropout(x))
        # 将最终输出也加入多尺度列表（作为最后一个尺度）
        multi_scale_outputs.append(final_output)
        
        return final_output, multi_scale_outputs


class TransformerBackgroundBlock(nn.Module):
    """
    基于 Window Attention 的背景特征变换模块
    
    流程:
    Input (B, C, H, W) 
      -> Pad (if needed)
      -> Window Partition -> (B*nW, N, C)  where N = ws*ws
      -> LayerNorm 
      -> Multi-Head Self-Attention (Local Context within Window)
      -> Residual Add
      -> LayerNorm
      -> Feed Forward Network (Channel Mixing)
      -> Residual Add
      -> Window Reverse -> Output (B, C, H, W)
    """
    def __init__(self, in_channels: int, out_channels: int = None, num_heads: int = 8, dropout: float = 0.1, window_size: int = 8):
        super().__init__()
        self.window_size = window_size
        
        # 如果未指定 out_channels 或与 in_channels 不同，Transformer 通常保持维度不变
        # 这里为了兼容接口保留参数，但实际输出通道数等于输入通道数
        if out_channels is not None and out_channels != in_channels:
            # 如果确实需要改变通道数，可以在最后加一个 Linear 层，但标准 Block 通常不改变
            pass
            
        # 确保通道数能被头数整除
        if in_channels % num_heads != 0:
            # 尝试降低头数
            for h in [4, 2, 1]:
                if in_channels % h == 0:
                    num_heads = h
                    break
        
        self.norm1 = nn.LayerNorm(in_channels)
        self.attn = nn.MultiheadAttention(embed_dim=in_channels, num_heads=num_heads, dropout=dropout, batch_first=True)
        
        self.norm2 = nn.LayerNorm(in_channels)
        self.ffn = nn.Sequential(
            nn.Linear(in_channels, in_channels * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(in_channels * 4, in_channels),
            nn.Dropout(dropout),
        )

    def window_partition(self, x: Tensor) -> Tensor:
        """
        Args:
            x: (B, C, H, W)
        Returns:
            windows: (num_windows*B, window_size*window_size, C)
        """
        B, C, H, W = x.shape
        x = x.view(B, C, H // self.window_size, self.window_size, W // self.window_size, self.window_size)
        windows = x.permute(0, 2, 4, 3, 5, 1).contiguous().view(-1, self.window_size * self.window_size, C)
        return windows

    def window_reverse(self, windows: Tensor, H: int, W: int) -> Tensor:
        """
        Args:
            windows: (num_windows*B, window_size*window_size, C)
            H, W: Image height and width
        Returns:
            x: (B, C, H, W)
        """
        B = int(windows.shape[0] / (H * W / self.window_size / self.window_size))
        x = windows.view(B, H // self.window_size, W // self.window_size, self.window_size, self.window_size, -1)
        x = x.permute(0, 5, 1, 3, 2, 4).contiguous().view(B, -1, H, W)
        return x

    def forward(self, x: Tensor) -> Tensor:
        b, c, h, w = x.shape
        
        # 0. Padding if needed
        pad_l = pad_t = 0
        pad_r = (self.window_size - w % self.window_size) % self.window_size
        pad_b = (self.window_size - h % self.window_size) % self.window_size
        if pad_r > 0 or pad_b > 0:
            x = F.pad(x, (pad_l, pad_r, pad_t, pad_b))
        
        _, _, H, W = x.shape
        
        # 1. Window Partition: (B, C, H, W) -> (B*nW, ws*ws, C)
        x_windows = self.window_partition(x)
        
        # 2. Self-Attention Block (Pre-Norm)
        x_norm = self.norm1(x_windows)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x_windows = x_windows + attn_out
        
        # 3. Feed Forward Network Block (Pre-Norm)
        x_norm = self.norm2(x_windows)
        ffn_out = self.ffn(x_norm)
        x_windows = x_windows + ffn_out
        
        # 4. Window Reverse
        out = self.window_reverse(x_windows, H, W)
        
        # 5. Remove padding
        if pad_r > 0 or pad_b > 0:
            out = out[:, :, :h, :w]
            
        return out


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
        # 使用 TransformerBackgroundBlock 替代原有的 BackgroundTransform
        self.bg_transforms = nn.ModuleList([
            TransformerBackgroundBlock(ch, ch) for ch in feature_channels
        ])
        
        # 创建多尺度输出层
        # 修改：输出层对应解码器上采样后的特征通道数
        self.out_convs = nn.ModuleList()
        for idx in range(len(feature_channels) - 1, 0, -1):
            self.out_convs.append(OutConv(feature_channels[idx - 1], out_channels))
        
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
        
        # 多尺度特征输出（前N-1个上采样块 + 最终的rec_logits）
        multi_scale_outputs = []
        
        # 标准上采样路径
        x = transformed_feats[-1]
        for idx, block in enumerate(self.up_blocks):
            skip = transformed_feats[-2 - idx]
            x = block(x, skip)
            # 对前N-1个上采样块进行多尺度输出
            if idx < len(self.up_blocks) - 1:
                multi_scale_outputs.append(self.out_convs[idx](x))
            
        final_output = self.out_conv(self.dropout(x))
        # 将最终输出也加入多尺度列表（作为最后一个尺度）
        multi_scale_outputs.append(final_output)
        
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
            out_channels=1,
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
        # 3.2 中重建分支输出已调整为 1 通道，与分割分支对齐
        # 为兼容旧权重，如遇到 >1 通道则回退为通道均值
        rec_logits_for_fusion = (
            rec_logits if rec_logits.shape[1] == 1 else rec_logits.mean(dim=1, keepdim=True)
        )
        fused_logits = seg_logits - self.rec_weight * rec_logits_for_fusion
        fused_mask = torch.sigmoid(fused_logits)

        # 多尺度融合输出：推理时计算，训练时为空（避免DDP未使用参数问题）
        multi_scale_fused_logits: List[Tensor] = []
        if not self.training and multi_scale_seg_logits and multi_scale_rec_logits:
            # 推理模式：计算多尺度融合特征
            for s_logit, r_logit in zip(multi_scale_seg_logits, multi_scale_rec_logits):
                r_logit_for_fusion = (
                    r_logit if r_logit.shape[1] == 1 else r_logit.mean(dim=1, keepdim=True)
                )
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

    @staticmethod
    def _ensure_channel_dim(mask: Tensor) -> Tensor:
        if mask.dim() == 3:
            return mask.unsqueeze(1)
        return mask


__all__ = ["RGSNetV3_1", "ForwardOutputV3_1"]