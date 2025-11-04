import os
import random
import csv

algorithms = ('Kumar-Roy', 'Murphy', 'Schroeder', 'intersection', 'voting')
# full_data_dir = "/data/zhaoys/activefire-google/decompressed/"
full_data_dir = "dataset/activefire"

# 创建保存CSV的目录
output_dir = "data/splits_activefire"
os.makedirs(output_dir, exist_ok=True)

# 设置随机种子保证可重复性
random.seed(42)

for algorithm in algorithms:
    image_dir = os.path.join(full_data_dir, 'images', 'patches')
    if algorithm in ('Kumar-Roy', 'Murphy', 'Schroeder'):
        mask_dir = os.path.join(full_data_dir, 'masks', 'patches')
    else:
        mask_dir = os.path.join(full_data_dir, 'masks', algorithm)

    # 匹配文件，清洗 images
    all_masks = set(os.listdir(mask_dir)) # 加快查找
    images = []
    masks = []
    for img in os.listdir(image_dir):
        mask = img.split('_')
        mask.insert(7, algorithm)
        mask = '_'.join(mask)
        if mask in all_masks:
            images.append(img)
            masks.append(mask)

    # 验证文件配对
    assert len(images) == len(masks), "图像和掩码数量不匹配"
    for img, msk in zip(images, masks):
        assert os.path.splitext(img)[0] == os.path.splitext(msk.replace(f'_{algorithm}', ''))[0], f"文件不匹配: {img} vs {msk}"
    
    # 组成完整路径并打包成元组
    img_paths = [os.path.join(image_dir, img) for img in images]
    mask_paths = [os.path.join(mask_dir, mask) for mask in masks]
    paired_data = list(zip(img_paths, mask_paths))
    
    # 随机打乱数据
    random.shuffle(paired_data)
    
    # 计算分割点
    total = len(paired_data)
    test_split = int(0.5 * total)
    val_split = int(0.1 * total)
    
    # 划分数据集
    test = paired_data[:test_split]
    val = paired_data[test_split:test_split + val_split]
    train = paired_data[test_split + val_split:]
    
    # 保存为CSV文件的函数
    def save_to_csv(data, split_name):
        csv_path = os.path.join(output_dir, f"{algorithm}_{split_name}.csv")
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['image_path', 'mask_path'])  # 写入表头
            writer.writerows(data)
    
    save_to_csv(train, "train")
    save_to_csv(val, "val")
    save_to_csv(test, "test")
    
    print(f"{algorithm}数据集划分完成: 训练集{len(train)}, 验证集{len(val)}, 测试集{len(test)}")