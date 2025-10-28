"""RGS-Net V3

基于 V1 的共享编码器双解码结构，做出以下核心调整：
- 两个解码分支均输出单通道 mask，并使用 FocalTverskyLoss 训练；
- 重建分支（背景分支）在接收跳连前对编码器特征做 1-x 翻转；
- 融合阶段使用 seg_logits - rec_logits 形成最终 logits，再经 Sigmoid+阈值得到 0/1 掩膜。
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
		feats: List[Tensor] = []
		out = self.stem(x)
		feats.append(out)
		for down in self.down_blocks:
			out = self.dropout(down(out))
			feats.append(out)
		return feats


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


@dataclass
class ForwardOutputV3:
	seg_logits: Tensor
	seg_mask: Tensor
	rec_logits: Tensor
	rec_mask: Tensor
	fused_logits: Tensor
	fused_mask: Tensor
	binary_mask: Tensor


class RGSNetV3(nn.Module):
	def __init__(
		self,
		n_channels: int = 3,
		n_classes: int = 1,
		n_filters: int = 32,
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
		self.rec_decoder = SegmentationDecoderUNet(
			feature_channels=feature_channels,
			out_channels=1,
			batchnorm=batchnorm,
			dropout=dropout,
			use_bilinear=use_bilinear,
		)

	def forward(self, x: Tensor, *, binarize: bool = True) -> ForwardOutputV3:
		features = self.encoder(x)
		seg_logits = self.seg_decoder(features)

		flipped_feats = [1.0 - feat for feat in features]  # 翻转跳连特征，突出背景信息
		rec_logits = self.rec_decoder(flipped_feats)

		seg_mask = torch.sigmoid(seg_logits)
		rec_mask = torch.sigmoid(rec_logits)

		fused_logits = seg_logits - rec_logits
		fused_mask = torch.sigmoid(fused_logits)

		if binarize:
			binary_mask = (fused_mask >= self.binarize_threshold).float()
		else:
			binary_mask = fused_mask

		return ForwardOutputV3(
			seg_logits=seg_logits,
			seg_mask=seg_mask,
			rec_logits=rec_logits,
			rec_mask=rec_mask,
			fused_logits=fused_logits,
			fused_mask=fused_mask,
			binary_mask=binary_mask,
		)

	def compute_losses(
		self,
		*,
		seg_logits: Tensor,
		rec_logits: Tensor,
		targets_seg: Tensor,
		seg_loss_fn: Callable[[Tensor, Tensor], Tensor],
		recon_loss_fn: Optional[Callable[[Tensor, Tensor], Tensor]] = None,
		targets_recon: Optional[Tensor] = None,
	) -> Dict[str, Tensor]:
		targets_seg = self._ensure_channel_dim(targets_seg)
		targets_seg = targets_seg.float()

		if targets_recon is None:
			targets_recon = 1.0 - targets_seg
		else:
			targets_recon = self._ensure_channel_dim(targets_recon).float()

		seg_loss = seg_loss_fn(seg_logits, targets_seg)
		recon_fn = recon_loss_fn if recon_loss_fn is not None else seg_loss_fn
		rec_loss = recon_fn(rec_logits, targets_recon)

		total_loss = seg_loss + rec_loss
		return {"loss": total_loss, "seg_loss": seg_loss, "rec_loss": rec_loss}

	@staticmethod
	def _ensure_channel_dim(mask: Tensor) -> Tensor:
		if mask.dim() == 3:
			return mask.unsqueeze(1)
		return mask


__all__ = ["RGSNetV3", "ForwardOutputV3"]


def analyze_v3_performance(
	*,
	input_shape: tuple = (1, 3, 256, 256),
	device: str = "cpu",
	gpu_id: int = 0,
	n_channels: int = 3,
	n_filters: int = 32,
	depth: int = 5,
	batchnorm: bool = True,
	dropout: float = 0.1,
	use_bilinear: bool = False,
	binarize_threshold: float = 0.5,
) -> None:
	"""使用 utils.analyze_model_performance 对 RGSNetV3 做快速性能分析。"""

	from utils import analyze_model_performance

	model = RGSNetV3(
		n_channels=n_channels,
		n_filters=n_filters,
		depth=depth,
		batchnorm=batchnorm,
		dropout=dropout,
		use_bilinear=use_bilinear,
		binarize_threshold=binarize_threshold,
	)
	analyze_model_performance(model=model, input_shape=input_shape, device=device, gpu_id=gpu_id)


if __name__ == "__main__":
	analyze_v3_performance(input_shape=(1, 3, 256, 256), device="cpu")

	if torch.cuda.is_available():
		analyze_v3_performance(input_shape=(8, 3, 256, 256), device="cuda", gpu_id=3)
