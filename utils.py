import os
import torch
from torch.utils.data import DataLoader, RandomSampler
import torch.nn.functional as F
from torch.profiler import profile, record_function, ProfilerActivity
import matplotlib.pyplot as plt
import numpy as np


def analyze_model_performance(
    model: torch.nn.Module,
    input_shape: tuple,
    device: str = "cpu",
    gpu_id: int = 0
):
    """
    分析模型性能（直接打印结果，不返回数据）
    
    Args:
        model: PyTorch模型
        input_shape: 输入张量形状 (batch_size, channels, height, width)
        device: 运行设备 ("cpu" 或 "cuda")
        gpu_id: 指定GPU ID（当device="cuda"时生效）
    """
    # 1. 设备设置
    if device == "cuda":
        assert torch.cuda.is_available(), "CUDA不可用！"
        torch.cuda.set_device(gpu_id)  # 指定GPU
        device = f"cuda:{gpu_id}"
    else:
        device = "cpu"
    
    model = model.to(device)
    dummy_input = torch.rand(*input_shape).to(device)

    # 2. 打印基础信息
    print("\n" + "="*60)
    print(f"{'模型性能分析':^60}")
    print("="*60)
    print(f"▪ 模型: {model.__class__.__name__}")
    print(f"▪ 输入形状: {input_shape}")
    print(f"▪ 运行设备: {device.upper()}")
    
    # 3. 参数量统计
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print("\n[参数量统计]")
    print(f"  Total: {total_params:,}")
    print(f"  Trainable: {trainable_params:,}")
    print(f"  Non-trainable: {total_params - trainable_params:,}")

    # 4. 运行性能分析 - 添加显存监控
    peak_memory = 0  # 初始化峰值显存
    
    with profile(
        activities=[ProfilerActivity.CUDA if "cuda" in device else ProfilerActivity.CPU],
        record_shapes=True,
        with_flops=True,
        profile_memory=True  # 启用内存分析
    ) as prof:
        with record_function("model_forward"):
            # 重置CUDA内存统计并运行模型
            if "cuda" in device:
                torch.cuda.reset_peak_memory_stats(device)
                model(dummy_input)
                # 获取峰值显存占用
                peak_memory = torch.cuda.max_memory_allocated(device)
            else:
                model(dummy_input)
    
    # 5. 计算量统计 - 使用多种方法
    print("\n[计算量统计]")
    
    # 方法1: 尝试从 profiler 获取
    flops_from_prof = 0
    try:
        flops_from_prof = prof.key_averages().total_average().flops or 0
        if flops_from_prof > 0:
            print(f"  GFLOPs (Profiler): {flops_from_prof / 1e9:.2f}")
    except (AttributeError, RuntimeError):
        pass
    
    # 方法2: 使用 fvcore (如果可用)
    try:
        from fvcore.nn import FlopCountAnalysis, flop_count_table
        import torch.nn as nn
        
        # 创建一个包装模块，只返回主要的 tensor 输出
        class ModelWrapper(nn.Module):
            def __init__(self, original_model):
                super().__init__()
                self.model = original_model
            
            def forward(self, x):
                output = self.model(x)
                # 如果输出是 dataclass 或有多个字段，只返回主要的 tensor
                if hasattr(output, 'seg_logits'):
                    return output.seg_logits
                elif hasattr(output, 'fused_logits'):
                    return output.fused_logits
                elif isinstance(output, (tuple, list)):
                    return output[0]
                else:
                    return output
        
        wrapped_model = ModelWrapper(model)
        flops_analyzer = FlopCountAnalysis(wrapped_model, dummy_input)
        total_flops = flops_analyzer.total()
        print(f"  GFLOPs (FVCore): {total_flops / 1e9:.2f}")
        # 可选：显示详细的每层FLOPS
        # print("\n[每层FLOPS详情]")
        # print(flop_count_table(flops_analyzer, max_depth=3))
    except ImportError:
        if flops_from_prof == 0:
            print("  提示: 安装 fvcore 以获得准确的 FLOPS 统计")
            print("  安装命令: pip install fvcore")
    except Exception as e:
        if flops_from_prof == 0:
            print(f"  FLOPS 计算失败: {e}")

    # 6. 时间统计
    time_stats = prof.key_averages().total_average()
    print("\n[时间统计]")
    if hasattr(time_stats, "cpu_time"):
        print(f"  CPU Time: {time_stats.cpu_time / 1000:.2f} ms")
    if hasattr(time_stats, "device_time"):
        print(f"  GPU Time: {time_stats.device_time / 1000:.2f} ms")

    # 7. 显存统计
    print("\n[显存统计]")
    if "cuda" in device:
        # 从CUDA直接获取的数据
        print(f"  峰值显存占用: {peak_memory / 1024**2:.2f} MB")
        
        # 从profiler获取的内存数据（可能更详细）
        mem_stats = prof.key_averages().total_average()
        if hasattr(mem_stats, "self_cuda_memory_usage"):
            print(f"  模型层显存占用: {mem_stats.self_cuda_memory_usage / 1024**2:.2f} MB")
    else:
        # CPU内存分析
        mem_stats = prof.key_averages().total_average()
        if hasattr(mem_stats, "self_cpu_memory_usage"):
            print(f"  CPU内存占用: {mem_stats.self_cpu_memory_usage / 1024**2:.2f} MB")

    # 8. 详细层统计
    print("\n[详细层性能]")
    print(prof.key_averages().table(
        sort_by="cuda_time_total" if "cuda" in device else "cpu_time_total",
        row_limit=10,
        # max_name_column_width=20
    ))
    
    print("="*60 + "\n")

def visualize_predictions(model, test_loader, device, threshold=0.5, num_samples=3, save_path='output.png'):
    """
    可视化预测    
    参数:
        model: 训练好的火点检测模型
        test_loader: 测试数据加载器
        device: 计算设备
        threshold: 火点概率阈值
        num_samples: 要可视化的样本数量
        save_path: 保存结果的路径
    """
    # 确保模型在评估模式
    model.eval()
    model.to(device)
    
    temp_loader = DataLoader(
        test_loader.dataset,
        batch_size=num_samples,
        sampler=RandomSampler(test_loader.dataset),
        num_workers=test_loader.num_workers
    )

    # 获取随机批次
    images, true_masks = next(iter(temp_loader))
    images, true_masks = images.to(device), true_masks.to(device)
    
    # 模型预测
    with torch.no_grad():
        pred_masks, _, _, _ = model(images)
        pred_masks = torch.sigmoid(pred_masks)
        pred_masks = (pred_masks > threshold).float()  # 二值化预测
    
    # 准备可视化
    fig, axes = plt.subplots(num_samples, images.shape[1] + 2, figsize=(12, 5 * num_samples))
    
    # 如果没有足够的行，确保axes是二维数组
    if num_samples == 1:
        axes = np.expand_dims(axes, axis=0)
    
    # 对每个样本进行可视化
    for i in range(num_samples):
        # 获取当前样本
        true_mask = true_masks[i].cpu().numpy().squeeze()
        pred_mask = pred_masks[i].cpu().numpy().squeeze()
        image = images[i].cpu().numpy()

        # 原图多通道展示
        for band in range(images.shape[1]):
            axes[i, band].imshow(image[band], cmap='gray')
            axes[i, band].set_title(f"C{band + 1}", fontsize=10)
            axes[i, band].axis('off')
        
        # 可视化真实火点标记
        axes[i, -2].imshow(true_mask, cmap='gray')
        axes[i, -2].set_title(f"GT | {np.sum(true_mask)} pixels", fontsize=10)
        axes[i, -2].axis('off')
        
        # 可视化模型预测结果
        axes[i, -1].imshow(pred_mask, cmap='gray')
        axes[i, -1].set_title(f"pred | {np.sum(pred_mask)} pixels", fontsize=10)
        axes[i, -1].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')

def adaptive_crop(images, masks, crop_size, max_crop_size):
    """
    基于火点位置进行自适应裁剪，无火点时返回中心裁剪
    images: [B, C, H, W]
    masks: [B, 1, H, W]
    crop_size: 当前目标裁剪尺寸
    """
    cropped_images = []
    cropped_masks = []
    
    for i in range(images.size(0)):
        img = images[i]
        mask = masks[i]
        
        # 检测火点位置（使用真实标签）
        fire_pixels = (mask[0] > 0).nonzero(as_tuple=False)
        
        if len(fire_pixels) > 0:
            # 随机选择一个火点作为中心
            center_idx = torch.randint(0, len(fire_pixels), (1,)).item()
            center_y, center_x = fire_pixels[center_idx]
            
            # 计算裁剪区域 (边界保护)
            H, W = img.shape[1], img.shape[2]
            half_size = crop_size // 2
            y_start = max(0, int(center_y - half_size))
            y_end = min(H, y_start + crop_size)
            x_start = max(0, int(center_x - half_size))
            x_end = min(W, x_start + crop_size)
            
            # 调整越界情况
            if y_end - y_start < crop_size:
                y_start = max(0, y_end - crop_size)
            if x_end - x_start < crop_size:
                x_start = max(0, x_end - crop_size)
            
            # 执行裁剪
            cropped_img = img[:, y_start:y_end, x_start:x_end]
            cropped_mask = mask[:, y_start:y_end, x_start:x_end]
        else:
            # 无火点：中心裁剪
            start_y = (img.shape[1] - crop_size) // 2
            start_x = (img.shape[2] - crop_size) // 2
            cropped_img = img[:, start_y:start_y+crop_size, start_x:start_x+crop_size]
            cropped_mask = mask[:, start_y:start_y+crop_size, start_x:start_x+crop_size]
        
        # 上采样回原始尺寸
        if crop_size != max_crop_size:
            cropped_img = F.interpolate(
                cropped_img.unsqueeze(0),
                size=(max_crop_size, max_crop_size),
                mode='bilinear',
                align_corners=True
            ).squeeze(0)
            cropped_mask = F.interpolate(
                cropped_mask.unsqueeze(0),
                size=(max_crop_size, max_crop_size),
                mode='nearest'  # 保持mask离散值
            ).squeeze(0)
        
        cropped_images.append(cropped_img)
        cropped_masks.append(cropped_mask)
    
    return torch.stack(cropped_images), torch.stack(cropped_masks)

