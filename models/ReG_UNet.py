"""ReG-UNet: Reconstruction-Gated U-Net (共享编码器的重建门控分割网络).

本模块提供一个共享编码器、双解码器（分割/重建）的 U-Net 变体：

* 分割分支输出火点概率图 :math:`R_{seg}`。
* 重建分支复原输入影像 :math:`\hat{x}`，计算重建误差 :math:`R_{rec} = |x - \hat{x}|`。

通过对重建误差进行 Sigmoid 门控并与分割概率相乘，实现“重建门控 (Reconstruction-Gated)”的抑噪与鲁棒融合。

核心特性（ReG-UNet）
-----------
* U-Net 风格共享编码器，提供多尺度跳跃特征。
* 分割/重建双解码器共享跳跃连接、但拥有各自的上采样路径与参数。
* 前向默认输出：logits、分割概率、重建结果、重建误差与融合后的概率。
* `compute_losses` 支持外部注入分割/重建损失（可直接使用 ``loss.py`` 中的实现），
  重建损失按火点掩码屏蔽前景、仅在背景上计算。

使用示例
~~~~~~~~
>>> model = ReGUNet(n_channels=3, n_classes=1)
>>> images = torch.randn(2, 3, 256, 256)
>>> outputs = model(images)
>>> outputs.seg_probs.shape
torch.Size([2, 1, 256, 256])
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

Tensor = torch.Tensor


# ---------------------------------------------------------------------------
# 基础卷积与模块
# ---------------------------------------------------------------------------


class DoubleConv(nn.Module):
    """经典 U-Net 中的双卷积单元."""

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

    def forward(self, x: Tensor) -> Tensor:  # noqa: D401
        return self.block(x)


class Down(nn.Module):
    """下采样模块: MaxPool + DoubleConv."""

    def __init__(self, in_channels: int, out_channels: int, batchnorm: bool) -> None:
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv = DoubleConv(in_channels, out_channels, batchnorm=batchnorm)

    def forward(self, x: Tensor) -> Tensor:  # noqa: D401
        return self.conv(self.pool(x))


class Up(nn.Module):
    """上采样模块: 转置卷积/双线性上采样 + 拼接 skip + DoubleConv."""

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

    def forward(self, x: Tensor, skip: Tensor) -> Tensor:  # noqa: D401
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:  # 对齐空间尺寸
            diff_y = skip.size(-2) - x.size(-2)
            diff_x = skip.size(-1) - x.size(-1)
            x = F.pad(x, [diff_x // 2, diff_x - diff_x // 2, diff_y // 2, diff_y - diff_y // 2])
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    """末端 1x1 卷积."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x: Tensor) -> Tensor:  # noqa: D401
        return self.conv(x)


# ---------------------------------------------------------------------------
# 结构定义
# ---------------------------------------------------------------------------


class SharedEncoderUNet(nn.Module):
    """U-Net 风格编码器, 返回每层特征 (包含输入端)."""

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

    def forward(self, x: Tensor) -> List[Tensor]:  # noqa: D401
        features: List[Tensor] = []
        out = self.stem(x)
        features.append(out)
        for down in self.down_blocks:
            out = self.dropout(down(out))
            features.append(out)
        return features


class SegmentationDecoderUNet(nn.Module):
    """分割分支解码器."""

    def __init__(
        self,
        feature_channels: List[int],
        out_channels: int,
        batchnorm: bool,
        dropout: float,
        use_bilinear: bool,
    ) -> None:
        super().__init__()
        ups: List[nn.Module] = []
        for idx in range(len(feature_channels) - 1, 0, -1):
            in_ch = feature_channels[idx]
            skip_ch = feature_channels[idx - 1]
            out_ch = feature_channels[idx - 1]
            ups.append(Up(in_ch, skip_ch, out_ch, batchnorm=batchnorm, use_bilinear=use_bilinear))
        self.up_blocks = nn.ModuleList(ups)
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        self.out_conv = OutConv(feature_channels[0], out_channels)

    def forward(self, features: List[Tensor]) -> Tensor:  # noqa: D401
        x = features[-1]
        for idx, block in enumerate(self.up_blocks):
            skip = features[-2 - idx]
            x = block(x, skip)
        return self.out_conv(self.dropout(x))


class ReconstructionDecoderUNet(nn.Module):
    """重建分支解码器, 结构与分割分支类似但输出通道等于输入通道."""

    def __init__(
        self,
        feature_channels: List[int],
        out_channels: int,
        batchnorm: bool,
        dropout: float,
        use_bilinear: bool,
    ) -> None:
        super().__init__()
        ups: List[nn.Module] = []
        for idx in range(len(feature_channels) - 1, 0, -1):
            in_ch = feature_channels[idx]
            skip_ch = feature_channels[idx - 1]
            out_ch = feature_channels[idx - 1]
            ups.append(Up(in_ch, skip_ch, out_ch, batchnorm=batchnorm, use_bilinear=use_bilinear))
        self.up_blocks = nn.ModuleList(ups)
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        self.out_conv = OutConv(feature_channels[0], out_channels)

    def forward(self, features: List[Tensor]) -> Tensor:  # noqa: D401
        x = features[-1]
        for idx, block in enumerate(self.up_blocks):
            skip = features[-2 - idx]
            x = block(x, skip)
        return self.out_conv(self.dropout(x))


@dataclass
class ForwardOutput:
    """模型前向输出的封装."""

    seg_logits: Tensor
    seg_probs: Tensor
    reconstruction: Tensor
    recon_error: Tensor
    fused_probs: Tensor


class ReGUNet(nn.Module):
    """ReG-UNet: 共享编码器 + 分割/重建双分支，使用重建误差门控分割概率的基线网络."""

    def __init__(
        self,
        n_channels: int = 3,
        n_classes: int = 1,
        n_filters: int = 32,
        depth: int = 5,
        batchnorm: bool = True,
        dropout: float = 0.1,
        use_bilinear: bool = False,
        tau: float = 1.0,
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

        feature_channels = [n_filters * (2 ** i) for i in range(depth)]
        self.seg_decoder = SegmentationDecoderUNet(
            feature_channels=feature_channels,
            out_channels=n_classes,
            batchnorm=batchnorm,
            dropout=dropout,
            use_bilinear=use_bilinear,
        )
        self.rec_decoder = ReconstructionDecoderUNet(
            feature_channels=feature_channels,
            out_channels=n_channels,
            batchnorm=batchnorm,
            dropout=dropout,
            use_bilinear=use_bilinear,
        )

    # ------------------------------------------------------------------
    # 前向推理
    # ------------------------------------------------------------------
    def forward(
        self,
        x: Tensor,
        *,
        tau: Optional[float] = None,
        fuse_outputs: bool = True,
    ) -> ForwardOutput:
        """执行前向推理并返回分割概率、重建结果与融合输出."""

        features = self.encoder(x)
        seg_logits = self.seg_decoder(features)
        reconstruction = self.rec_decoder(features)

        seg_probs = self._apply_activation(seg_logits)
        recon_error = torch.abs(x - reconstruction)

        if fuse_outputs:
            fused = self._fuse(seg_probs, recon_error, tau=tau)
        else:
            fused = seg_probs

        return ForwardOutput(
            seg_logits=seg_logits,
            seg_probs=seg_probs,
            reconstruction=reconstruction,
            recon_error=recon_error,
            fused_probs=fused,
        )

    # ------------------------------------------------------------------
    # 损失计算
    # ------------------------------------------------------------------
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
        """计算分割与重建损失。"""

        seg_loss = seg_loss_fn(seg_logits, targets_seg)

        # ---- 构建背景掩码 ----
        if fire_mask is None:
            fire_mask = self._infer_fire_mask(targets_seg)
        fire_mask = fire_mask.float()
        if fire_mask.dim() == 3:
            fire_mask = fire_mask.unsqueeze(1)
        background_mask = 1.0 - fire_mask
        if background_mask.shape[1] != inputs.shape[1]:
            background_mask = background_mask.expand(-1, inputs.shape[1], -1, -1)

        # ---- 计算重建损失 ----
        if recon_loss_fn is None:
            diff = torch.abs((reconstruction - inputs) * background_mask)
            if reduction == "sum":
                rec_loss = diff.sum()
            else:
                denom = background_mask.sum().clamp_min(1.0)
                rec_loss = diff.sum() / denom
        else:
            rec_loss = recon_loss_fn(reconstruction * background_mask, inputs * background_mask)

        total_loss = seg_loss + rec_loss
        return {
            "loss": total_loss,
            "seg_loss": seg_loss,
            "rec_loss": rec_loss,
        }

    # ------------------------------------------------------------------
    # 内部工具
    # ------------------------------------------------------------------
    def _apply_activation(self, seg_logits: Tensor) -> Tensor:
        if self.n_classes == 1:
            return torch.sigmoid(seg_logits)
        return F.softmax(seg_logits, dim=1)

    def _fuse(self, seg_probs: Tensor, recon_error: Tensor, tau: Optional[float]) -> Tensor:
        tau = self.tau if tau is None else tau
        if tau <= 0:
            raise ValueError("温度参数 tau 必须为正数")
        # 取重建误差的均值作为异常强度，经 Sigmoid 压缩到 [0,1]
        recon_scalar = recon_error.mean(dim=1, keepdim=True)
        weighting = torch.sigmoid(recon_scalar / tau)
        return seg_probs * weighting

    @staticmethod
    def _infer_fire_mask(targets_seg: Tensor) -> Tensor:
        if targets_seg.shape[1] == 1:
            return (targets_seg > 0.5).float()
        return torch.argmax(targets_seg, dim=1, keepdim=True).float()


# 对外导出
__all__ = ["ReGUNet", "ForwardOutput"]


if __name__ == "__main__":
    from utils import analyze_model_performance

    # 示例1：分析CPU性能
    analyze_model_performance(
        model=ReGUNet(n_channels=3, n_filters=32),
        input_shape=(1, 3, 256, 256),
        device='cpu'
    )
    
    # 示例2：分析指定GPU（如GPU 0）的性能
    analyze_model_performance(
        model=ReGUNet(n_channels=3, n_filters=32),
        input_shape=(64, 3, 256, 256),
        device='cuda',
        gpu_id=0
    )
