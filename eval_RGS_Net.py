import os
import torch
from torch.utils.data import DataLoader, RandomSampler
from dataset import LandsatFireDataset
from models.RGS_Net import RGSNet
import time
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# GPU设置
os.environ["CUDA_VISIBLE_DEVICES"] = "2,3,4,5"

# 输出初始配置信息
print(f"========== 火灾检测评估启动 ==========")
print(f"当前时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
print(f"计算设备: {'GPU可用' if torch.cuda.is_available() else '仅限CPU'}")

DATA_ROOT = "data/full"
ALGORITHM = "voting"  # 可选项: 'Kumar-Roy', 'Murphy', 'Schroeder', 'intersection', 'voting'
SAVE_DIR = "output/RGS_Net/voting_202510101118"  # 训练文件夹
TH_FIRE  = 0.5  # 火点阈值
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
USE_FUSION = True  # 是否使用融合概率进行评估

TAU = 1  # 融合权重参数, 仅在USE_FUSION=True时有效


# 测试数据准备阶段 
print("\n[阶段 1/4] 准备测试数据")
test_data_csv = os.path.join(DATA_ROOT, f"{ALGORITHM}_test.csv")
print(f"├─ 算法标签: {ALGORITHM}")
print(f"├─ 测试集CSV: {test_data_csv}")

test_dataset = LandsatFireDataset(test_data_csv, bands=(7, 6, 5))
print(f"├─ 测试集样本数: {len(test_dataset)}")

test_loader = DataLoader(
    test_dataset,
    batch_size=128,
    shuffle=False,
    num_workers=4
)
print(f"└─ 数据加载器创建完成: {len(test_loader)} batches")




# 模型加载阶段
print("\n[阶段 2/4] 加载预训练模型")
model_name = "model_best"
param_path = os.path.join(SAVE_DIR, "weights", f"{model_name}.pth")
print(f"├─ 参数路径: {param_path}")

model = RGSNet(n_channels=3, n_filters=32, tau=TAU)
print(f"├─ 网络架构: {model.__class__.__name__}")
try:
    state_dict = torch.load(param_path)
    model.load_state_dict(state_dict)
    print(f"└─ 成功加载模型参数 (参数数量: {sum(p.numel() for p in model.parameters())})")
except Exception as e:
    print(f"!! 模型加载错误: {str(e)}")
    exit(1)




# 评估配置
print("\n[阶段 3/4] 配置评估参数")
print(f"├─ 推理设备: {DEVICE}")
print(f"├─ 火点阈值: {TH_FIRE}")
print(f"└─ 输出目录: {SAVE_DIR}")




# 执行评估
print("\n[阶段 4/4] 开始模型评估")
# model = torch.nn.DataParallel(model)
model.to(DEVICE)
model.eval()

# 初始化统计变量
num_samples = 0

# 初始化全局 TP/FP/FN 用于微平均
total_tp = 0
total_fp = 0
total_fn = 0

with torch.no_grad():
    test_bar = tqdm(test_loader, desc=f"Test Size [{len(test_dataset)}]")
    for batch_idx, (images, masks) in enumerate(test_bar):
        images = images.to(DEVICE)
        masks = masks.to(DEVICE)
        
        # 模型预测
        outputs = model(images, fuse_outputs=USE_FUSION)
        probs = outputs.fused_probs if USE_FUSION else outputs.seg_probs
        preds = (probs > TH_FIRE)
        
        # 计算批次指标
        batch_size = images.size(0)
        num_samples += batch_size
        
        pred_np = preds.cpu().numpy()
        mask_np = masks.cpu().numpy()

        tp = np.logical_and(pred_np, mask_np).sum()
        fp = np.logical_and(pred_np, np.logical_not(mask_np)).sum()
        fn = np.logical_and(np.logical_not(pred_np), mask_np).sum()

        total_tp += int(tp)
        total_fp += int(fp)
        total_fn += int(fn)

# 计算指标与汇总
result_filename = f"eval_{model_name}"
metrics = {}

precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

metrics.update({
    'precision': precision,
    'recall': recall,
    'f1': f1
})

report_path = os.path.join(SAVE_DIR, f"{result_filename}.txt")
with open(report_path, 'w') as f:
    f.write("="*50 + "\n")
    f.write("火点检测模型评估报告\n")
    f.write("="*50 + "\n\n")

    f.write(f"设备: {DEVICE}\n")
    f.write(f"阈值: {TH_FIRE}\n")
    f.write(f"测试样本数: {num_samples}\n\n")

    f.write("="*50 + "\n")
    f.write("评估指标 (微平均)\n")
    f.write("="*50 + "\n")
    f.write(f"精确率: {metrics['precision']:.4f}\n")
    f.write(f"召回率: {metrics['recall']:.4f}\n")
    f.write(f"F1分数: {metrics['f1']:.4f}\n")

# 打印结果
print("\n" + "="*50)
print("火点检测模型评估结果:")
print(f" - 精确率: {metrics['precision']:.4f}")
print(f" - 召回率: {metrics['recall']:.4f}")
print(f" - F1分数: {metrics['f1']:.4f}")
print(f"\n评估结果已保存至: {SAVE_DIR}")
print(f"- 文本报告: {report_path}")
print("="*50 + "\n")


# 抽样可视化
temp_loader = DataLoader(
    test_loader.dataset,
    batch_size=5,
    sampler=RandomSampler(test_loader.dataset),
    num_workers=test_loader.num_workers
)

# 获取随机批次
images, true_masks = next(iter(temp_loader))
images, true_masks = images.to(DEVICE), true_masks.to(DEVICE)

# 模型预测
with torch.no_grad():
    pred_outputs = model(images, fuse_outputs=USE_FUSION)
    pred_probs = pred_outputs.fused_probs if USE_FUSION else pred_outputs.seg_probs
    pred_masks = (pred_probs > TH_FIRE).float()  # 二值化预测

# 准备可视化
fig, axes = plt.subplots(5, images.shape[1] + 2, figsize=(12, 5 * 5))

# 对每个样本进行可视化
for i in range(5):
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
plt.savefig(os.path.join(SAVE_DIR, model_name + "_prediction.png"), dpi=300, bbox_inches='tight')

print(f"\n评估完成! 结果已保存至 {SAVE_DIR}")

# 最终状态
print("\n========== 评估流程结束 ==========")