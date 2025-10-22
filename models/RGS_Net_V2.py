"""RGS-Net V2

改进点（基于 Unet_2 架构）：
- 使用共享编码器与双解码器（分割/重建）。
- 在重建分支的每个上采样阶段，利用该阶段的编码器特征与重建分支对应阶段的特征计算重建误差，
  将误差映射到[0,1]的权重图，作为对分割分支“同级跳跃特征”的引导（加权融合）。

直觉：背景区域更容易被重建（误差小），火点区域更难被重建（误差大）。
在每个解码阶段用该误差生成的权重图对分割跳连进行空间门控，可抑制背景伪阳性、突出疑似火点区域，
从而提升分割鲁棒性。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

Tensor = torch.Tensor


# ---------------------------------------------------------------------------
# 基础模块（与 Unet_2 内部模块等价的最小实现）
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# 主结构
# ---------------------------------------------------------------------------


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


@dataclass
class ForwardOutputV2:
    seg_logits: Tensor
    seg_probs: Tensor
    reconstruction: Tensor
    recon_error: Tensor
    fused_probs: Tensor
    stage_weights: List[Tensor]  # 每级的权重图（B,1,H,W）


class RGSNetV2(nn.Module):
    """共享编码器 + 重建引导的分级融合分割。

    在每个上采样阶段：
    - 重建分支先进行上采样并与同级编码器特征拼接-卷积，得到阶段特征 R_k。
    - 计算与编码器特征 E_k 的差异误差 |R_k - E_k|，经通道平均与平滑卷积、Sigmoid 得到权重 W_k ∈ (0,1)。
    - 分割分支在同级上采样后，使用 W_k 对编码器跳跃特征 E_k 做空间门控（E_k ⊙ W_k），再与分割上采样特征拼接-卷积。
    """

    def __init__(
        self,
        n_channels: int = 3,
        n_classes: int = 1,
        n_filters: int = 32,
        depth: int = 5,
        batchnorm: bool = True,
        dropout: float = 0.1,
        use_bilinear: bool = False,
        tau: float = 1.0,  # 权重温度（控制误差映射到(0,1)的敏感度）
    ) -> None:
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.tau = tau

        self.encoder = SharedEncoderUNet(
            in_channels=n_channels,
            base_filters=n_filters,
            depth=depth,
            batchnorm=batchnorm,
            dropout=dropout,
        )

        feat_chs = [n_filters * (2 ** i) for i in range(depth)]

        # 分支解码模块（逐级执行，便于交互）
        self.rec_up_blocks = nn.ModuleList()
        self.seg_up_blocks = nn.ModuleList()
        self.attn_smooth = nn.ModuleList()  # 将误差(1通道)平滑后再 Sigmoid

        for idx in range(depth - 1, 0, -1):
            in_ch = feat_chs[idx]
            skip_ch = feat_chs[idx - 1]
            out_ch = feat_chs[idx - 1]

            self.rec_up_blocks.append(Up(in_ch, skip_ch, out_ch, batchnorm=batchnorm, use_bilinear=use_bilinear))
            self.seg_up_blocks.append(Up(in_ch, skip_ch, out_ch, batchnorm=batchnorm, use_bilinear=use_bilinear))
            self.attn_smooth.append(nn.Conv2d(1, 1, kernel_size=3, padding=1))

        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        self.seg_out = OutConv(feat_chs[0], n_classes)
        self.rec_out = OutConv(feat_chs[0], n_channels)

    # ----------------------------------------------
    # 前向：联合双解码，逐级计算权重并门控分割跳连
    # ----------------------------------------------
    def forward(self, x: Tensor, *, fuse_outputs: bool = True, tau: Optional[float] = None) -> ForwardOutputV2:
        feats = self.encoder(x)  # len = depth, 从高分辨率到低分辨率

        x_seg = feats[-1]
        x_rec = feats[-1]
        stage_weights: List[Tensor] = []

        blocks = zip(self.rec_up_blocks, self.seg_up_blocks, self.attn_smooth)
        # 对应的 skip 特征顺序：feats[-2], feats[-3], ... , feats[0]
        for stage_idx, (rec_block, seg_block, smooth) in enumerate(blocks):
            enc_skip = feats[-2 - stage_idx]

            # 重建分支上采样到该级
            x_rec = rec_block(x_rec, enc_skip)  # 形状与 enc_skip 通道数一致

            # 误差 -> 权重（单通道），值域(0,1)
            err = torch.abs(x_rec - enc_skip).mean(dim=1, keepdim=True)
            t = self.tau if tau is None else tau
            attn = torch.sigmoid(smooth(err / max(t, 1e-6)))  # (B,1,H,W)
            stage_weights.append(attn)

            # 分割分支：对跳连做空间门控后再拼接
            gated_skip = enc_skip * attn
            x_seg = seg_block(x_seg, gated_skip)

        seg_logits = self.seg_out(self.dropout(x_seg))
        rec_img = self.rec_out(self.dropout(x_rec))

        if self.n_classes == 1:
            seg_probs = torch.sigmoid(seg_logits)
        else:
            seg_probs = F.softmax(seg_logits, dim=1)

        recon_error = torch.abs(x - rec_img)
        fused = self._final_fuse(seg_probs, recon_error, tau=tau)

        return ForwardOutputV2(
            seg_logits=seg_logits,
            seg_probs=seg_probs,
            reconstruction=rec_img,
            recon_error=recon_error,
            fused_probs=fused,
            stage_weights=stage_weights,
        )

    def _final_fuse(self, seg_probs: Tensor, recon_error: Tensor, tau: Optional[float]) -> Tensor:
        t = self.tau if tau is None else tau
        if t <= 0:
            raise ValueError("温度参数 tau 必须为正数")
        recon_scalar = recon_error.mean(dim=1, keepdim=True)
        w = torch.sigmoid(recon_scalar / t)
        return seg_probs * w

    # ----------------------------------------------
    # 训练期损失（与 Unet_2 类似）
    # ----------------------------------------------
    def compute_losses(
        self,
        *,
        seg_logits: Tensor,
        reconstruction: Tensor,
        targets_seg: Tensor,
        inputs: Tensor,
        seg_loss_fn: Callable[[Tensor, Tensor], Tensor],
        recon_loss_fn: Optional[Callable[[Tensor, Tensor], Tensor]] = None,
        fire_mask: Optional[Tensor] = None,
        reduction: str = "mean",
    ) -> Dict[str, Tensor]:
        seg_loss = seg_loss_fn(seg_logits, targets_seg)

        if fire_mask is None:
            fire_mask = self._infer_fire_mask(targets_seg)
        fire_mask = fire_mask.float()
        if fire_mask.dim() == 3:
            fire_mask = fire_mask.unsqueeze(1)
        background_mask = 1.0 - fire_mask
        if background_mask.shape[1] != inputs.shape[1]:
            background_mask = background_mask.expand(-1, inputs.shape[1], -1, -1)

        if recon_loss_fn is None:
            diff = torch.abs((reconstruction - inputs) * background_mask)
            if reduction == "sum":
                rec_loss = diff.sum()
            else:
                denom = background_mask.sum().clamp_min(1.0)
                rec_loss = diff.sum() / denom
        else:
            rec_loss = recon_loss_fn(reconstruction * background_mask, inputs * background_mask)

        total = seg_loss + rec_loss
        return {"loss": total, "seg_loss": seg_loss, "rec_loss": rec_loss}

    @staticmethod
    def _infer_fire_mask(targets_seg: Tensor) -> Tensor:
        if targets_seg.dim() == 4 and targets_seg.shape[1] == 1:
            return (targets_seg > 0.5).float()
        return torch.argmax(targets_seg, dim=1, keepdim=True).float()


__all__ = ["RGSNetV2", "ForwardOutputV2"]

if __name__ == "__main__":
    # 简单自测
    model = RGSNetV2(n_channels=3, n_filters=32)
    x = torch.randn(2, 3, 256, 256)
    out = model(x)
    print(out.seg_logits.shape, out.seg_probs.shape, out.reconstruction.shape)
