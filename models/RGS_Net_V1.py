"""RGS-Net V1 (renamed from Unet_2): shared-encoder dual-decoder U-Net.

此版本将原 Unet_2 架构重命名为 RGS_Net_V1，并移除旧版 RGS_Net_V1 实现。
提供分割 logits/prob、重建影像、重建误差与融合概率输出，并支持训练期的损失计算。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

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
        features: List[Tensor] = []
        out = self.stem(x)
        features.append(out)
        for down in self.down_blocks:
            out = self.dropout(down(out))
            features.append(out)
        return features


class SegmentationDecoderUNet(nn.Module):
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

    def forward(self, features: List[Tensor]) -> Tensor:
        x = features[-1]
        for idx, block in enumerate(self.up_blocks):
            skip = features[-2 - idx]
            x = block(x, skip)
        return self.out_conv(self.dropout(x))


class ReconstructionDecoderUNet(nn.Module):
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

    def forward(self, features: List[Tensor]) -> Tensor:
        x = features[-1]
        for idx, block in enumerate(self.up_blocks):
            skip = features[-2 - idx]
            x = block(x, skip)
        return self.out_conv(self.dropout(x))


@dataclass
class ForwardOutput:
    seg_logits: Tensor
    seg_probs: Tensor
    reconstruction: Tensor
    recon_error: Tensor
    fused_probs: Tensor


class RGSNetV1(nn.Module):
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

    def forward(
        self,
        x: Tensor,
        *,
        tau: Optional[float] = None,
        fuse_outputs: bool = True,
    ) -> ForwardOutput:
        features = self.encoder(x)
        seg_logits = self.seg_decoder(features)
        reconstruction = self.rec_decoder(features)

        seg_probs = self._apply_activation(seg_logits)
        recon_error = torch.abs(x - reconstruction)

        fused = self._fuse(seg_probs, recon_error, tau=tau) if fuse_outputs else seg_probs
        return ForwardOutput(
            seg_logits=seg_logits,
            seg_probs=seg_probs,
            reconstruction=reconstruction,
            recon_error=recon_error,
            fused_probs=fused,
        )

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

        total_loss = seg_loss + rec_loss
        return {"loss": total_loss, "seg_loss": seg_loss, "rec_loss": rec_loss}

    def _apply_activation(self, seg_logits: Tensor) -> Tensor:
        if self.n_classes == 1:
            return torch.sigmoid(seg_logits)
        return F.softmax(seg_logits, dim=1)

    def _fuse(self, seg_probs: Tensor, recon_error: Tensor, tau: Optional[float]) -> Tensor:
        tau = self.tau if tau is None else tau
        if tau <= 0:
            raise ValueError("温度参数 tau 必须为正数")
        recon_scalar = recon_error.mean(dim=1, keepdim=True)
        weighting = torch.sigmoid(recon_scalar / tau)
        return seg_probs * weighting

    @staticmethod
    def _infer_fire_mask(targets_seg: Tensor) -> Tensor:
        if targets_seg.shape[1] == 1:
            return (targets_seg > 0.5).float()
        return torch.argmax(targets_seg, dim=1, keepdim=True).float()


__all__ = ["RGSNetV1", "ForwardOutput"]


def analyze_v1_performance(
    *,
    input_shape: tuple = (1, 3, 256, 256),
    device: str = "cpu",
    gpu_id: int = 0,
    n_channels: int = 3,
    n_classes: int = 1,
    n_filters: int = 32,
    depth: int = 5,
    batchnorm: bool = True,
    dropout: float = 0.1,
    use_bilinear: bool = False,
    tau: float = 1.0,
) -> None:
    """Convenience helper to profile RGSNetV1 via utils.analyze_model_performance.

    This prints performance stats (params, FLOPs, time, memory). It does not return values.
    """
    # Import locally to avoid circular imports at module import time
    from utils import analyze_model_performance

    model = RGSNetV1(
        n_channels=n_channels,
        n_classes=n_classes,
        n_filters=n_filters,
        depth=depth,
        batchnorm=batchnorm,
        dropout=dropout,
        use_bilinear=use_bilinear,
        tau=tau,
    )
    analyze_model_performance(model=model, input_shape=input_shape, device=device, gpu_id=gpu_id)


if __name__ == "__main__":
    # Example runs: CPU and (if available) GPU profiling
    analyze_v1_performance(input_shape=(1, 3, 256, 256), device="cpu")

    if torch.cuda.is_available():
        # Use a moderate batch for quick GPU profiling; adjust if needed
        analyze_v1_performance(input_shape=(8, 3, 256, 256), device="cuda", gpu_id=0)
