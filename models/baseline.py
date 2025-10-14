import torch
import torch.nn as nn
import torch.nn.functional as F


class UNet(nn.Module):
    def __init__(self, n_channels=10, n_classes=1, n_filters=16, dropout=0.1, batchnorm=True):
        super(UNet, self).__init__()
        
        # 编码器 (下采样)
        self.enc1 = self._double_conv(n_channels, n_filters, batchnorm)
        self.enc2 = self._double_conv(n_filters, n_filters*2, batchnorm)
        self.enc3 = self._double_conv(n_filters*2, n_filters*4, batchnorm)
        self.enc4 = self._double_conv(n_filters*4, n_filters*8, batchnorm)
        self.enc5 = self._double_conv(n_filters*8, n_filters*16, batchnorm)
        
        self.pool = nn.MaxPool2d(2)
        self.dropout = nn.Dropout2d(dropout)
        
        # 解码器 (上采样)
        self.up6 = nn.ConvTranspose2d(n_filters*16, n_filters*8, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.dec6 = self._double_conv(n_filters*16, n_filters*8, batchnorm)
        
        self.up7 = nn.ConvTranspose2d(n_filters*8, n_filters*4, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.dec7 = self._double_conv(n_filters*8, n_filters*4, batchnorm)
        
        self.up8 = nn.ConvTranspose2d(n_filters*4, n_filters*2, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.dec8 = self._double_conv(n_filters*4, n_filters*2, batchnorm)
        
        self.up9 = nn.ConvTranspose2d(n_filters*2, n_filters, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.dec9 = self._double_conv(n_filters*2, n_filters, batchnorm)
        
        # 输出层
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
        p1 = self.dropout(self.pool(e1))
        
        e2 = self.enc2(p1)
        p2 = self.dropout(self.pool(e2))
        
        e3 = self.enc3(p2)
        p3 = self.dropout(self.pool(e3))
        
        e4 = self.enc4(p3)
        p4 = self.dropout(self.pool(e4))
        
        e5 = self.enc5(p4)
        
        # 解码器路径（带跳跃连接）
        d6 = self.up6(e5)
        d6 = torch.cat([d6, e4], dim=1)
        d6 = self.dropout(d6)
        d6 = self.dec6(d6)
        
        d7 = self.up7(d6)
        d7 = torch.cat([d7, e3], dim=1)
        d7 = self.dropout(d7)
        d7 = self.dec7(d7)
        
        d8 = self.up8(d7)
        d8 = torch.cat([d8, e2], dim=1)
        d8 = self.dropout(d8)
        d8 = self.dec8(d8)
        
        d9 = self.up9(d8)
        d9 = torch.cat([d9, e1], dim=1)
        d9 = self.dropout(d9)
        d9 = self.dec9(d9)
        
        return self.outc(d9)


if __name__ == "__main__":
    from utils import analyze_model_performance

    # 示例1：分析CPU性能
    analyze_model_performance(
        model=UNet(n_channels=3, n_filters=16),
        input_shape=(1, 3, 256, 256),
        device='cpu'
    )
    
    # 示例2：分析指定GPU（如GPU 1）的性能
    analyze_model_performance(
        model=UNet(n_channels=3, n_filters=16),
        input_shape=(64, 3, 256, 256),
        device='cuda',
        gpu_id=0
    )