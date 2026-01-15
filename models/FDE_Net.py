import torch
import torch.nn as nn
import torch.nn.functional as F

# ==============================================================================
# 1. Haar Wavelet Downsampling (HWD)
# ==============================================================================
class HWD(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(HWD, self).__init__()
        # 论文提到先通过Haar变换使通道数x4，然后通过1x1卷积调整通道数 [cite: 331, 332]
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels * 4, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # Haar Wavelet Decomposition
        # x: (B, C, H, W) -> LL, LH, HL, HH: (B, C, H/2, W/2)
        LL = (x[:, :, 0::2, 0::2] + x[:, :, 0::2, 1::2] + x[:, :, 1::2, 0::2] + x[:, :, 1::2, 1::2]) / 2
        LH = (x[:, :, 0::2, 0::2] + x[:, :, 0::2, 1::2] - x[:, :, 1::2, 0::2] - x[:, :, 1::2, 1::2]) / 2
        HL = (x[:, :, 0::2, 0::2] - x[:, :, 0::2, 1::2] + x[:, :, 1::2, 0::2] - x[:, :, 1::2, 1::2]) / 2
        HH = (x[:, :, 0::2, 0::2] - x[:, :, 0::2, 1::2] - x[:, :, 1::2, 0::2] + x[:, :, 1::2, 1::2]) / 2
        
        # 拼接 4 个子带 [cite: 330]
        x_cat = torch.cat([LL, LH, HL, HH], dim=1)  # (B, 4C, H/2, W/2)
        
        # 1x1 卷积调整通道 [cite: 332]
        return self.conv(x_cat)

# ==============================================================================
# 2. Convolutional Block Attention Module (CBAM)
# ==============================================================================
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        # Shared MLP [cite: 287]
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out # Element-wise sum [cite: 287]
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        # 7x7 Conv [cite: 314]
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # AvgPool and MaxPool along channel dimension [cite: 314]
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        x_out = self.conv1(x_cat)
        return self.sigmoid(x_out)

class CBAM(nn.Module):
    def __init__(self, in_planes, ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.ca = ChannelAttention(in_planes, ratio)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x):
        # Channel Attention First, then Spatial Attention [cite: 276]
        out = x * self.ca(x)
        out = out * self.sa(out)
        return out

# ==============================================================================
# 3. Self-Attention and Convolutional Mixture (ACmix)
# ==============================================================================
class ACmix(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_att=7, head=4, kernel_conv=3, stride=1, dilation=1):
        super(ACmix, self).__init__()
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.head = head
        self.kernel_att = kernel_att
        self.kernel_conv = kernel_conv
        self.stride = stride
        self.dilation = dilation
        # Start each path with equal weighting for stable training
        self.rate1 = torch.nn.Parameter(torch.tensor(1.0))
        self.rate2 = torch.nn.Parameter(torch.tensor(1.0))
        self.head_dim = out_planes // head

        # 1x1 Conv projection [cite: 227]
        self.conv1 = nn.Conv2d(in_planes, out_planes * 3, kernel_size=1)
        self.conv2 = nn.Conv2d(in_planes, out_planes, kernel_size=1)

        # Convolution Path (simulating shift with depthwise conv for efficiency as noted in [cite: 228])
        self.conv_path = nn.Conv2d(out_planes, out_planes, kernel_size=kernel_conv, 
                                   stride=stride, padding=kernel_conv//2, groups=out_planes, bias=False)
        
        # Self-Attention Path
        # Note: Using Local Window Attention (Sliding Window) to save memory
        # Global attention on 256x256 is too expensive (O(N^2))
        self.softmax = nn.Softmax(dim=-1)
        self.unfold = nn.Unfold(kernel_size=kernel_att, padding=kernel_att//2, stride=1)

    def forward(self, x):
        b, c, h, w = x.shape
        
        # 1. Projection
        qkv = self.conv1(x)
        q, k, v = torch.chunk(qkv, 3, dim=1) # Split into Q, K, V

        # 2. Convolution Path [cite: 227-236]
        # 论文中提到使用depthwise conv作为shift操作的轻量化替代 [cite: 228]
        f_conv = self.conv_path(q + k + v) # 简化的特征聚合，实际通常对中间特征做处理

        # 3. Self-Attention Path [cite: 223]
        # Local Window Attention implementation
        
        # Prepare Q: (B, Head, Dim, 1, L)
        q_att = q.view(b, self.head, self.head_dim, h * w).unsqueeze(3)
        
        # Prepare K, V: Unfold to get neighbors
        # (B, C, H, W) -> (B, C*K*K, L) -> (B, Head, Dim, K*K, L)
        k_unfold = self.unfold(k).view(b, self.head, self.head_dim, self.kernel_att**2, h * w)
        v_unfold = self.unfold(v).view(b, self.head, self.head_dim, self.kernel_att**2, h * w)
        
        # Calculate Energy: (B, Head, K*K, L)
        scaling = self.head_dim ** 0.5
        energy = (q_att * k_unfold).sum(dim=2) 
        
        # Softmax over neighbors (dim=2), NOT over pixels (dim=3)
        att = F.softmax(energy / scaling, dim=2)
        
        # Apply Attention: (B, Head, Dim, L)
        # att: (B, Head, KK, L) -> (B, Head, 1, KK, L)
        # v_unfold: (B, Head, Dim, KK, L)
        f_att = (att.unsqueeze(2) * v_unfold).sum(dim=3)
        
        f_att = f_att.reshape(b, self.out_planes, h, w)

        # 4. Fusion [cite: 241]
        # f_out = alpha * f_att + beta * f_conv
        out = self.rate1 * f_att + self.rate2 * f_conv
        return out

# ==============================================================================
# 4. Residual Block
# ==============================================================================
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.relu(out)
        return out

# ==============================================================================
# 5. FDE U-Net Architecture
# ==============================================================================
class FDE_UNet(nn.Module):
    def __init__(self, in_channels=3, num_classes=1):
        super(FDE_UNet, self).__init__()
        
        # --- Encoder ---
        # 初始层
        self.inc = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True)
        )
        
        # Stage 1 (256x256x16 -> 128x128x32)
        self.res1 = ResidualBlock(16, 16)
        self.acmix1 = ACmix(16, 16)
        self.cbam1 = CBAM(16)
        self.hwd1 = HWD(16, 32) # Downsampling

        # Stage 2 (128x128x32 -> 64x64x64)
        self.res2 = ResidualBlock(32, 32)
        self.acmix2 = ACmix(32, 32)
        self.cbam2 = CBAM(32)
        self.hwd2 = HWD(32, 64)

        # Stage 3 (64x64x64 -> 32x32x128)
        self.res3 = ResidualBlock(64, 64)
        self.acmix3 = ACmix(64, 64)
        self.cbam3 = CBAM(64)
        self.hwd3 = HWD(64, 128)

        # Stage 4 (32x32x128 -> 16x16x256)
        self.res4 = ResidualBlock(128, 128)
        self.acmix4 = ACmix(128, 128)
        self.cbam4 = CBAM(128)
        self.hwd4 = HWD(128, 256) # Bottleneck input

        # --- Bottleneck ---
        self.bridge = ResidualBlock(256, 256)

        # --- Decoder ---
        # Decoder 1
        self.up1 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.res_dec1 = ResidualBlock(256, 128) # 128 (up) + 128 (skip) = 256 in_channels

        # Decoder 2
        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.res_dec2 = ResidualBlock(128, 64) # 64 + 64 = 128

        # Decoder 3
        self.up3 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.res_dec3 = ResidualBlock(64, 32) # 32 + 32 = 64

        # Decoder 4
        self.up4 = nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2)
        self.res_dec4 = ResidualBlock(32, 16) # 16 + 16 = 32

        # Output Layer
        self.outc = nn.Conv2d(16, num_classes, kernel_size=1)

    def forward(self, x):
        # --- Encoder Path ---
        x1 = self.inc(x)             # 256x256x16
        
        # Block 1
        x1_res = self.res1(x1)
        x1_att = self.acmix1(x1_res)
        x1_att = self.cbam1(x1_att)  # Skip connection source 1
        x2 = self.hwd1(x1_att)       # 128x128x32

        # Block 2
        x2_res = self.res2(x2)
        x2_att = self.acmix2(x2_res)
        x2_att = self.cbam2(x2_att)  # Skip connection source 2
        x3 = self.hwd2(x2_att)       # 64x64x64

        # Block 3
        x3_res = self.res3(x3)
        x3_att = self.acmix3(x3_res)
        x3_att = self.cbam3(x3_att)  # Skip connection source 3
        x4 = self.hwd3(x3_att)       # 32x32x128

        # Block 4
        x4_res = self.res4(x4)
        x4_att = self.acmix4(x4_res)
        x4_att = self.cbam4(x4_att)  # Skip connection source 4
        x5 = self.hwd4(x4_att)       # 16x16x256

        # --- Bridge ---
        bridge = self.bridge(x5)

        # --- Decoder Path ---
        # Up 1
        d1 = self.up1(bridge)        # 16x16 -> 32x32 (128 ch)
        d1 = torch.cat([x4_att, d1], dim=1) # Concat: 128 + 128 = 256
        d1 = self.res_dec1(d1)

        # Up 2
        d2 = self.up2(d1)            # 32x32 -> 64x64 (64 ch)
        d2 = torch.cat([x3_att, d2], dim=1) # Concat: 64 + 64 = 128
        d2 = self.res_dec2(d2)

        # Up 3
        d3 = self.up3(d2)            # 64x64 -> 128x128 (32 ch)
        d3 = torch.cat([x2_att, d3], dim=1) # Concat: 32 + 32 = 64
        d3 = self.res_dec3(d3)

        # Up 4
        d4 = self.up4(d3)            # 128x128 -> 256x256 (16 ch)
        d4 = torch.cat([x1_att, d4], dim=1) # Concat: 16 + 16 = 32
        d4 = self.res_dec4(d4)

        # Output
        logits = self.outc(d4)
        return logits # Return logits for BCEWithLogitsLoss

# ==============================================================================
# 测试代码
# ==============================================================================
if __name__ == "__main__":
    # 模拟 Landsat-8 输入: Batch=1, Channels=10, Height=256, Width=256 [cite: 356]
    dummy_input = torch.randn(1, 10, 256, 256)
    
    model = FDE_UNet(in_channels=10, num_classes=1)
    
    # 打印输出尺寸
    output = model(dummy_input)
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}") # 预期: (1, 1, 256, 256)
    
    # 计算参数量
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total Parameters: {total_params / 1e6:.2f}M") 
    # 论文中提到参数量约为 2.25M