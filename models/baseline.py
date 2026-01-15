import sys
import torch
import torch.nn as nn
import torch.nn.functional as F


class UNet(nn.Module):
    def __init__(
        self,
        n_channels=3,
        n_classes=1,
        n_filters=64,
        dropout=0.1,
        batchnorm=True,
    ):
        super(UNet, self).__init__()

        # 编码器 (下采样)
        self.enc1 = self._double_conv(n_channels, n_filters, batchnorm)
        self.pool1 = nn.MaxPool2d(2)
        self.drop1 = nn.Dropout(dropout)

        self.enc2 = self._double_conv(n_filters, n_filters * 2, batchnorm)
        self.pool2 = nn.MaxPool2d(2)
        self.drop2 = nn.Dropout(dropout)

        self.enc3 = self._double_conv(n_filters * 2, n_filters * 4, batchnorm)
        self.pool3 = nn.MaxPool2d(2)
        self.drop3 = nn.Dropout(dropout)

        self.enc4 = self._double_conv(n_filters * 4, n_filters * 8, batchnorm)
        self.pool4 = nn.MaxPool2d(2)
        self.drop4 = nn.Dropout(dropout)

        self.enc5 = self._double_conv(n_filters * 8, n_filters * 16, batchnorm)

        # 解码器 (上采样)
        self.up6 = nn.ConvTranspose2d(
            n_filters * 16,
            n_filters * 8,
            kernel_size=3,
            stride=2,
            padding=1,
            output_padding=1,
        )
        self.drop6 = nn.Dropout(dropout)
        self.dec6 = self._double_conv(n_filters * 16, n_filters * 8, batchnorm)

        self.up7 = nn.ConvTranspose2d(
            n_filters * 8,
            n_filters * 4,
            kernel_size=3,
            stride=2,
            padding=1,
            output_padding=1,
        )
        self.drop7 = nn.Dropout(dropout)
        self.dec7 = self._double_conv(n_filters * 8, n_filters * 4, batchnorm)

        self.up8 = nn.ConvTranspose2d(
            n_filters * 4,
            n_filters * 2,
            kernel_size=3,
            stride=2,
            padding=1,
            output_padding=1,
        )
        self.drop8 = nn.Dropout(dropout)
        self.dec8 = self._double_conv(n_filters * 4, n_filters * 2, batchnorm)

        self.up9 = nn.ConvTranspose2d(
            n_filters * 2,
            n_filters,
            kernel_size=3,
            stride=2,
            padding=1,
            output_padding=1,
        )
        self.drop9 = nn.Dropout(dropout)
        self.dec9 = self._double_conv(n_filters * 2, n_filters, batchnorm)

        # 输出层 (不再包含 Sigmoid；返回 logits 以配合 BCEWithLogitsLoss)
        self.outc = nn.Conv2d(n_filters, n_classes, kernel_size=1)

    def _double_conv(self, in_channels, out_channels, batchnorm):
        """双卷积块: Conv -> BatchNorm -> ReLU"""
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels) if batchnorm else nn.Identity(),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels) if batchnorm else nn.Identity(),
            nn.ReLU(inplace=True)
        ]
        return nn.Sequential(*layers)

    def forward(self, x):
        # 编码器路径
        e1 = self.enc1(x)
        p1 = self.drop1(self.pool1(e1))

        e2 = self.enc2(p1)
        p2 = self.drop2(self.pool2(e2))

        e3 = self.enc3(p2)
        p3 = self.drop3(self.pool3(e3))

        e4 = self.enc4(p3)
        p4 = self.drop4(self.pool4(e4))

        e5 = self.enc5(p4)
        
        # 解码器路径（带跳跃连接）
        d6 = self.up6(e5)
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

        outputs = self.outc(d9)
        return outputs


if __name__ == "__main__":
    import os
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if PROJECT_ROOT not in sys.path:
        sys.path.insert(0, PROJECT_ROOT)
    from utils import analyze_model_performance

    # 示例1：分析CPU性能
    analyze_model_performance(
        model=UNet(n_channels=3, n_filters=64),
        input_shape=(1, 3, 256, 256),
        device='cpu'
    )
    
    # 示例2：分析指定GPU（如GPU 1）的性能
    analyze_model_performance(
        model=UNet(n_channels=3, n_filters=64),
        input_shape=(16, 3, 256, 256),
        device='cuda',
        gpu_id=0 
    )