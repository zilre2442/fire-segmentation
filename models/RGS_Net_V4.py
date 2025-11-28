"""RGS-Net V4

基于 ActiveFireUNetBaseline 结构扩展的双解码器版本:
Encoder + SegmentationDecoder 与 baseline UNet 完全一致。
新增 ReconstructionDecoder: 对每个尺度的 encoder 特征应用 TransformerBackgroundBlock 后再走与分割分支相同的上采样结构。
输出 seg_logits, rec_logits, fused_logits (seg_logits - weight * rec_logits)。

创新点:
1. 引入 TransformerBackgroundBlock 以捕获背景特征的全局上下文信息。
2. 双解码器结构: 分割分支和重建分支相辅相成，提升分割性能。
3. 融合策略: 使用重建分支的输出对分割结果进行校正。

与现有方法对比:
- 相较于传统 UNet，RGS-Net V4 在背景建模上更具优势。
- 双分支设计提升了模型的鲁棒性。

"""

from __future__ import annotations

from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

Tensor = torch.Tensor


# -------------------------------------------------
# 基础卷积块与 ActiveFireUNetBaseline 保持一致
# -------------------------------------------------
class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, batchnorm: bool = True):
        super().__init__()
        layers = [
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch) if batchnorm else nn.Identity(),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch) if batchnorm else nn.Identity(),
            nn.ReLU(inplace=True),
        ]
        self.block = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:  # noqa: D401
        return self.block(x)


class UpBlock(nn.Module):
    """上采样 + 拼接 + ConvBlock (与 baseline transpose conv + 2 conv 保持结构次序)。"""

    def __init__(self, in_ch: int, skip_ch: int, out_ch: int, batchnorm: bool, dropout: float):
        super().__init__()
        # 使用转置卷积对齐 baseline
        self.up = nn.ConvTranspose2d(in_ch, out_ch, 3, stride=2, padding=1, output_padding=1)
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        # 拼接后通道 = out_ch + skip_ch
        self.conv = ConvBlock(out_ch + skip_ch, out_ch, batchnorm=batchnorm)

    def forward(self, x: Tensor, skip: Tensor) -> Tensor:
        x = self.up(x)
        # 形状对齐 (一般应一致, 若有偏差做pad裁剪)
        if x.shape[-2:] != skip.shape[-2:]:
            diff_y = skip.size(-2) - x.size(-2)
            diff_x = skip.size(-1) - x.size(-1)
            x = F.pad(x, [diff_x // 2, diff_x - diff_x // 2, diff_y // 2, diff_y - diff_y // 2])
        x = torch.cat([x, skip], dim=1)
        x = self.drop(x)
        return self.conv(x)


# -------------------------------------------------
# Transformer 背景变换 (简化自 V3_2)
# -------------------------------------------------
class TransformerBackgroundBlock(nn.Module):
    """
    Transformer 背景变换模块。

    设计动机:
    - 捕获背景特征的全局上下文信息。
    - 使用窗口注意力机制提升计算效率。

    参数:
    - channels (int): 输入特征的通道数。
    - num_heads (int): 多头注意力的头数。
    - dropout (float): Dropout 概率。
    - window_size (int): 窗口大小。

    输入:
    - x (Tensor): 输入特征图，形状为 (B, C, H, W)。

    输出:
    - Tensor: 经过背景变换的特征图，形状与输入一致。

    """

    def __init__(self, channels: int, num_heads: int = 8, dropout: float = 0.1, window_size: int = 8):
        super().__init__()
        self.ws = window_size
        # 保证可以整除
        if channels % num_heads != 0:
            for h in [4, 2, 1]:
                if channels % h == 0:
                    num_heads = h
                    break
        self.norm1 = nn.LayerNorm(channels)
        self.attn = nn.MultiheadAttention(channels, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(channels)
        self.ffn = nn.Sequential(
            nn.Linear(channels, channels * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(channels * 4, channels),
            nn.Dropout(dropout),
        )

    def _partition(self, x: Tensor) -> Tuple[Tensor, int, int]:
        """
        将特征图划分为窗口。

        输入:
        - x (Tensor): 输入特征图，形状为 (B, C, H, W)。

        输出:
        - Tuple[Tensor, int, int]:
          - 窗口化后的特征，形状为 (B*nWin, ws*ws, C)。
          - 原始特征图的高度和宽度。
        """
        B, C, H, W = x.shape
        ws = self.ws
        x = x.view(B, C, H // ws, ws, W // ws, ws).permute(0, 2, 4, 3, 5, 1).contiguous()
        return x.view(-1, ws * ws, C), H, W

    def _reverse(self, windows: Tensor, H: int, W: int) -> Tensor:
        """
        将窗口还原为特征图。

        输入:
        - windows (Tensor): 窗口化后的特征，形状为 (B*nWin, ws*ws, C)。
        - H (int): 原始特征图的高度。
        - W (int): 原始特征图的宽度。

        输出:
        - Tensor: 还原后的特征图，形状为 (B, C, H, W)。
        """
        ws = self.ws
        num_win = (H // ws) * (W // ws)
        B = windows.shape[0] // num_win
        x = windows.view(B, H // ws, W // ws, ws, ws, -1).permute(0, 5, 1, 3, 2, 4).contiguous()
        return x.view(B, -1, H, W)

    def forward(self, x: Tensor) -> Tensor:
        B, C, h, w = x.shape
        ws = self.ws
        pad_r = (ws - w % ws) % ws
        pad_b = (ws - h % ws) % ws
        if pad_r or pad_b:
            x = F.pad(x, (0, pad_r, 0, pad_b))
        H, W = x.shape[-2:]
        tokens, H, W = self._partition(x)  # (B*nWin, ws*ws, C)
        y = self.attn(self.norm1(tokens), self.norm1(tokens), self.norm1(tokens))[0]
        tokens = tokens + y
        y2 = self.ffn(self.norm2(tokens))
        tokens = tokens + y2
        out = self._reverse(tokens, H, W)
        if pad_r or pad_b:
            out = out[:, :, :h, :w]
        return out


# -------------------------------------------------
# 主模型: RGSNetV4
# -------------------------------------------------
class RGSNetV4(nn.Module):
    """
    RGS-Net V4 主模型。

    设计动机:
    - 在分割任务中引入背景建模，提升对复杂场景的适应能力。
    - 双解码器结构: 分割分支和重建分支相辅相成。

    参数:
    - n_channels (int): 输入图像的通道数。
    - n_classes (int): 输出类别数，仅支持单通道分割。
    - base_filters (int): 基础卷积通道数。
    - dropout (float): Dropout 概率。
    - batchnorm (bool): 是否使用批归一化。

    输入:
    - x (Tensor): 输入图像，形状为 (B, C, H, W)。

    输出:
    - Tuple[Tensor, Tensor, Tensor]:
      - seg_logits: 分割分支的输出。
      - rec_logits: 重建分支的输出。
      - fused_logits: 融合后的输出。

    """

    def __init__(
        self,
        n_channels: int = 3,
        n_classes: int = 1,
        base_filters: int = 64,
        dropout: float = 0.1,
        batchnorm: bool = True,
    ) -> None:
        super().__init__()
        if n_classes != 1:
            raise ValueError("仅支持单通道分割")

        self.rec_weight = nn.Parameter(torch.tensor(1.0))

        # Encoder (4 down + bottleneck) 与 baseline 同结构
        self.enc1 = ConvBlock(n_channels, base_filters * 1, batchnorm)
        self.pool1 = nn.MaxPool2d(2); self.drop1 = nn.Dropout(dropout)
        self.enc2 = ConvBlock(base_filters * 1, base_filters * 2, batchnorm)
        self.pool2 = nn.MaxPool2d(2); self.drop2 = nn.Dropout(dropout)
        self.enc3 = ConvBlock(base_filters * 2, base_filters * 4, batchnorm)
        self.pool3 = nn.MaxPool2d(2); self.drop3 = nn.Dropout(dropout)
        self.enc4 = ConvBlock(base_filters * 4, base_filters * 8, batchnorm)
        self.pool4 = nn.MaxPool2d(2); self.drop4 = nn.Dropout(dropout)
        self.bottleneck = ConvBlock(base_filters * 8, base_filters * 16, batchnorm)

        # Segmentation decoder (mirrors baseline)
        self.up6_seg = UpBlock(base_filters * 16, base_filters * 8, base_filters * 8, batchnorm, dropout)
        self.up7_seg = UpBlock(base_filters * 8, base_filters * 4, base_filters * 4, batchnorm, dropout)
        self.up8_seg = UpBlock(base_filters * 4, base_filters * 2, base_filters * 2, batchnorm, dropout)
        self.up9_seg = UpBlock(base_filters * 2, base_filters * 1, base_filters * 1, batchnorm, dropout)
        self.seg_out = nn.Conv2d(base_filters, n_classes, 1)

        # Reconstruction decoder: encoder 特征先过 TransformerBackgroundBlock
        self.bg_transforms = nn.ModuleList([
            TransformerBackgroundBlock(base_filters * f) for f in [1, 2, 4, 8, 16]
        ])
        self.up6_rec = UpBlock(base_filters * 16, base_filters * 8, base_filters * 8, batchnorm, dropout)
        self.up7_rec = UpBlock(base_filters * 8, base_filters * 4, base_filters * 4, batchnorm, dropout)
        self.up8_rec = UpBlock(base_filters * 4, base_filters * 2, base_filters * 2, batchnorm, dropout)
        self.up9_rec = UpBlock(base_filters * 2, base_filters * 1, base_filters * 1, batchnorm, dropout)
        self.rec_out = nn.Conv2d(base_filters, 1, 1)

    def _encode(self, x: Tensor) -> List[Tensor]:
        """
        编码器部分。

        输入:
        - x (Tensor): 输入图像，形状为 (B, C, H, W)。

        输出:
        - List[Tensor]: 编码器的多尺度特征。
        """
        e1 = self.enc1(x)
        p1 = self.drop1(self.pool1(e1))
        e2 = self.enc2(p1)
        p2 = self.drop2(self.pool2(e2))
        e3 = self.enc3(p2)
        p3 = self.drop3(self.pool3(e3))
        e4 = self.enc4(p3)
        p4 = self.drop4(self.pool4(e4))
        b = self.bottleneck(p4)
        return [e1, e2, e3, e4, b]

    def _decode_seg(self, feats: List[Tensor]) -> Tensor:
        """
        分割分支解码器。

        输入:
        - feats (List[Tensor]): 编码器的多尺度特征。

        输出:
        - Tensor: 分割分支的输出，形状为 (B, n_classes, H, W)。
        """
        e1, e2, e3, e4, b = feats
        d6 = self.up6_seg(b, e4)
        d7 = self.up7_seg(d6, e3)
        d8 = self.up8_seg(d7, e2)
        d9 = self.up9_seg(d8, e1)
        return self.seg_out(d9)

    def _decode_rec(self, feats: List[Tensor]) -> Tensor:
        """
        重建分支解码器。

        输入:
        - feats (List[Tensor]): 编码器的多尺度特征。

        输出:
        - Tensor: 重建分支的输出，形状为 (B, 1, H, W)。
        """
        # 背景变换 (每尺度对应一个 Block)
        t_feats = [bg(f) for f, bg in zip(feats, self.bg_transforms)]
        t1, t2, t3, t4, tb = t_feats
        d6 = self.up6_rec(tb, t4)
        d7 = self.up7_rec(d6, t3)
        d8 = self.up8_rec(d7, t2)
        d9 = self.up9_rec(d8, t1)
        return self.rec_out(d9)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """
        前向传播。

        输入:
        - x (Tensor): 输入图像，形状为 (B, C, H, W)。

        输出:
        - Tuple[Tensor, Tensor, Tensor]:
          - seg_logits: 分割分支的输出。
          - rec_logits: 重建分支的输出。
          - fused_logits: 融合后的输出。
        """
        feats = self._encode(x)
        seg_logits = self._decode_seg(feats)
        rec_logits = self._decode_rec(feats)
        # 融合: seg - w * rec (保持 logit 空间操作)
        rec_for_fuse = rec_logits if rec_logits.shape[1] == 1 else rec_logits.mean(dim=1, keepdim=True)
        fused_logits = seg_logits - self.rec_weight * rec_for_fuse
        return seg_logits, rec_logits, fused_logits


__all__ = ["RGSNetV4", "TransformerBackgroundBlock"]

if __name__ == "__main__":
    model = RGSNetV4(n_channels=3, base_filters=32)
    x = torch.randn(2, 3, 256, 256)
    s, r, f = model(x)
    print("seg", s.shape, "rec", r.shape, "fused", f.shape, "weight", model.rec_weight.item())