import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, batchnorm: bool = True):
        super().__init__()
        layers = [
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels) if batchnorm else nn.Identity(),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels) if batchnorm else nn.Identity(),
            nn.ReLU(inplace=True),
        ]
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)

class ActiveFireUNetBaseline(nn.Module):
    """PyTorch reproduction of activefire Keras UNet (unet_64f_2conv_762 variant).

    Differences vs RGS models:
    - No reconstruction branch, single segmentation output.
    - Uses transpose conv for upsampling, 2 convs per block.
    - Returns raw logits; caller applies sigmoid for metrics/thresh.
    - Dropout applied after pooling and after concat (mirrors original code intent).
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
        self.n_channels = n_channels
        self.n_classes = n_classes

        # Encoder
        self.enc1 = ConvBlock(n_channels, base_filters * 1, batchnorm)
        self.pool1 = nn.MaxPool2d(2)
        self.drop1 = nn.Dropout(dropout)

        self.enc2 = ConvBlock(base_filters * 1, base_filters * 2, batchnorm)
        self.pool2 = nn.MaxPool2d(2)
        self.drop2 = nn.Dropout(dropout)

        self.enc3 = ConvBlock(base_filters * 2, base_filters * 4, batchnorm)
        self.pool3 = nn.MaxPool2d(2)
        self.drop3 = nn.Dropout(dropout)

        self.enc4 = ConvBlock(base_filters * 4, base_filters * 8, batchnorm)
        self.pool4 = nn.MaxPool2d(2)
        self.drop4 = nn.Dropout(dropout)

        self.bottleneck = ConvBlock(base_filters * 8, base_filters * 16, batchnorm)

        # Decoder
        self.up6 = nn.ConvTranspose2d(base_filters * 16, base_filters * 8, 3, stride=2, padding=1, output_padding=1)
        self.drop6 = nn.Dropout(dropout)
        self.dec6 = ConvBlock(base_filters * 16, base_filters * 8, batchnorm)

        self.up7 = nn.ConvTranspose2d(base_filters * 8, base_filters * 4, 3, stride=2, padding=1, output_padding=1)
        self.drop7 = nn.Dropout(dropout)
        self.dec7 = ConvBlock(base_filters * 8, base_filters * 4, batchnorm)

        self.up8 = nn.ConvTranspose2d(base_filters * 4, base_filters * 2, 3, stride=2, padding=1, output_padding=1)
        self.drop8 = nn.Dropout(dropout)
        self.dec8 = ConvBlock(base_filters * 4, base_filters * 2, batchnorm)

        self.up9 = nn.ConvTranspose2d(base_filters * 2, base_filters * 1, 3, stride=2, padding=1, output_padding=1)
        self.drop9 = nn.Dropout(dropout)
        self.dec9 = ConvBlock(base_filters * 2, base_filters * 1, batchnorm)

        self.out_conv = nn.Conv2d(base_filters, n_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        p1 = self.drop1(self.pool1(e1))

        e2 = self.enc2(p1)
        p2 = self.drop2(self.pool2(e2))

        e3 = self.enc3(p2)
        p3 = self.drop3(self.pool3(e3))

        e4 = self.enc4(p3)
        p4 = self.drop4(self.pool4(e4))

        b = self.bottleneck(p4)

        d6 = self.up6(b)
        d6 = torch.cat([d6, e4], dim=1)
        d6 = self.drop6(d6)
        d6 = self.dec6(d6)

        d7 = self.up7(d6)
        d7 = torch.cat([d7, e3], dim=1)
        d7 = self.drop7(d7)
        d7 = self.dec7(d7)

        d8 = self.up8(d7)
        d8 = torch.cat([d8, e2], dim=1)
        d8 = self.drop8(d8)
        d8 = self.dec8(d8)

        d9 = self.up9(d8)
        d9 = torch.cat([d9, e1], dim=1)
        d9 = self.drop9(d9)
        d9 = self.dec9(d9)

        logits = self.out_conv(d9)  # raw logits
        return logits

__all__ = ["ActiveFireUNetBaseline"]

if __name__ == "__main__":
    model = ActiveFireUNetBaseline(n_channels=3, base_filters=64)
    x = torch.randn(2, 3, 256, 256)
    y = model(x)
    print("Output shape:", y.shape)
