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

        try:
            # 读取掩膜（优先读取第1个波段，得到 [H, W]）
            with rasterio.open(mask_path) as msrc:
                if msrc.count < 1:
                    raise ValueError(f"掩膜没有可读取的波段: {mask_path}")
                mask = msrc.read(1)  # [H, W]

            # 读取影像指定波段
            with rasterio.open(img_path) as isrc:
                if len(self.bands) == 0:
                    raise ValueError("bands 参数不能为空")
                max_band = max(self.bands)
                if max_band > isrc.count:
                    raise ValueError(
                        f"请求的波段 {self.bands} 超过影像可用波段数 {isrc.count}: {img_path}"
                    )
                thermal_img = isrc.read(self.bands)  # [C, H, W]

            # 确保图像为 [C, H, W]
            if thermal_img.ndim == 2:
                thermal_img = np.expand_dims(thermal_img, axis=0).astype(np.float32)

            # 掩膜统一为 [1, H, W]
            if mask.ndim == 2:
                mask = np.expand_dims(mask, axis=0)

            # 归一化处理, landsat8数据为 16bit 量化（PNG 掩膜不需要归一化）
            thermal_img = thermal_img / 65535.0

            # 转换为 PyTorch 张量
            thermal_img = torch.from_numpy(thermal_img).float()
            mask = torch.from_numpy(mask).float()

            return thermal_img, mask

        except Exception as e:
            # 在 DataLoader worker 中抛出更可读的异常，便于定位问题样本
            raise RuntimeError(
                f"读取样本失败 idx={idx}: img='{img_path}', mask='{mask_path}' | 错误: {e}"
            )

# 使用示例
if __name__ == "__main__":
    algorithms = ('Kumar-Roy', 'Murphy', 'Schroeder', 'intersection', 'voting')
    full_data_dir = "data/splits_activefire"

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