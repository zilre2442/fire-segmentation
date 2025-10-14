import os
import csv
import torch
import numpy as np
from torch.utils.data import Dataset
import rasterio


class LandsatFireDataset(Dataset):
    def __init__(self, csv_path, bands=(7, 6, 2)):
        """
        基于CSV分割文件的Landsat8火点检测数据集
        
        参数:
            csv_path: 包含图像和掩码路径对的CSV文件路径, 元组格式"(原图路径, 掩码路径)"
            bands: 需要读入的波段, (1, 2, ...), 总共10个波段, 从 1 开始
        """
        self.pairs = []
        self.bands = bands
        
        # 从CSV文件中读取图像-掩码对
        with open(csv_path, 'r') as f:
            reader = csv.reader(f)
            next(reader)  # 跳过表头
            for row in reader:
                self.pairs.append((row[0], row[1]))
        
        # 验证文件存在性
        for img_path, mask_path in self.pairs:
            if not os.path.exists(img_path):
                raise FileNotFoundError(f"图像文件不存在: {img_path}")
            if not os.path.exists(mask_path):
                raise FileNotFoundError(f"掩码文件不存在: {mask_path}")
    
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        img_path, mask_path = self.pairs[idx]
        
        # 读取火点掩码
        with rasterio.open(mask_path) as src:
            mask = src.read()
        
        # 读取热红外图像
        with rasterio.open(img_path) as src:
            thermal_img = src.read(self.bands)
        
        # 创建热红外图像 [C, H, W], 通道数等于波段数, 对于单通道图像需增加通道维度
        if thermal_img.ndim == 2:
            thermal_img = np.expand_dims(thermal_img, axis=0).astype(np.float32)
        
        # 归一化处理, landsat8数据为 16bit 量化
        thermal_img = thermal_img / 65535
        
        # 转换为PyTorch张量
        thermal_img = torch.from_numpy(thermal_img).float()
        mask = torch.from_numpy(mask).float()
        
        return thermal_img, mask

# 使用示例
if __name__ == "__main__":
    algorithms = ('Kumar-Roy', 'Murphy', 'Schroeder', 'intersection', 'voting')
    full_data_dir = "data/full"

    for algorithm in algorithms:
        # 示例用法
        train_dataset = LandsatFireDataset(f"{full_data_dir}/{algorithm}_train.csv")
        val_dataset = LandsatFireDataset(f"{full_data_dir}/{algorithm}_val.csv")
        test_dataset = LandsatFireDataset(f"{full_data_dir}/{algorithm}_test.csv")
        
        print(f"训练集大小: {len(train_dataset)}")
        print(f"验证集大小: {len(val_dataset)}")
        print(f"测试集大小: {len(test_dataset)}")
        
        # 获取一个样本
        sample_img, sample_mask = train_dataset[0]
        print(f"图像形状: {sample_img.shape}, 掩码形状: {sample_mask.shape}")
        print(f"img 唯一值: {torch.unique(sample_img)}, mask 唯一值: {torch.unique(sample_mask)}\n")