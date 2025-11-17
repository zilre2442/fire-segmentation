import os
from datetime import datetime
from typing import Tuple
import logging

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import random
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

import sys
import os

# Ensure project root is on sys.path so top-level modules (dataset, loss, utils, models) can be imported
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from dataset import LandsatFireDataset
from models.RGS_Net_V3_1 import RGSNetV3_1
from loss import SpatialFocalTverskyLoss, MaskedL1Loss, get_criterion_info
from utils import adaptive_crop

# 使用示例:
# 1) 分布式训练 (推荐，torchrun 会自动设置 RANK/WORLD_SIZE/LOCAL_RANK 等环境变量)
#    假设在一台有 4 张 GPU 的机器上运行:
#
#    torchrun --nproc_per_node=4 exp/train_scripts/train_RGS_Net_V3_1.py
#
#    或者显式设置可见 GPU（例如使用 GPU 1,2）:
#
#    CUDA_VISIBLE_DEVICES=1,2 torchrun --nproc_per_node=2 exp/train_scripts/train_RGS_Net_V3_1.py --fire-category 
#
#    使用按火点数量划分的子数据集（例如训练 very_few 类别）:
#
#    torchrun --nproc_per_node=4 exp/train_scripts/train_RGS_Net_V3_1.py --fire-category very_few
#
#    或训练 many 类别的样本:
#
#    torchrun --nproc_per_node=4 exp/train_scripts/train_RGS_Net_V3_1.py --fire-category many --epochs 150
#
# 2) 单卡快速调试（需要手动设置环境变量供脚本读取）:
#
#    RANK=0 WORLD_SIZE=1 LOCAL_RANK=0 CUDA_VISIBLE_DEVICES=1 python3 exp/train_scripts/train_RGS_Net_V3_1.py
#
#    使用 few 类别数据集:
#
#    RANK=0 WORLD_SIZE=1 LOCAL_RANK=0 python3 exp/train_scripts/train_RGS_Net_V3_1.py --fire-category few
#
# 3) 其他可选参数:
#    --algorithm: 数据集算法类型 (voting, Kumar-Roy, Murphy, Schroeder, intersection, Land8Fire)
#    --fire-category: 按火点数量选择子集 (very_few, few, many, very_many)
#    --batch-size: 批大小
#    --epochs: 训练轮数
#    --lr: 学习率
#    --save-dir: 自定义保存目录
#
# 注意:
# - 脚本默认使用 NCCL 后端进行分布式训练 (GPU)，在 CPU-only 环境上请确保使用单卡模式并设置
#   相应环境变量。
# - 训练中使用的损失函数为 SpatialFocalTverskyLoss 和 MaskedL1Loss
# - 火点数量类别定义: very_few (1-10像素), few (10-100像素), many (100-1000像素), very_many (>1000像素)



import argparse

DATA_ROOT = 'data/splits_activefire'
ALGORITHM = 'voting'
RUN_ID = datetime.now().strftime("%Y%m%d%H%M")
SAVE_DIR = f"output/RGS_Net_V3_1/{ALGORITHM}_{RUN_ID}"
BANDS = (7, 6, 5)
BATCH_SIZE = 64
NUM_WORKERS = 4
SHUFFLE_TRAIN = True
EPOCHS = 200
LEARNING_RATE = 3e-4
VAL_INTERVAL = 1
SAVE_INTERVAL = 10
EARLY_STOPPING_PATIENCE = 20
EARLY_STOPPING_MIN_DELTA = 0

# 火点数量类别（用于选择按火点数量划分的子数据集）
# 可选值: None (使用全量数据), "very_few", "few", "many", "very_many"
FIRE_PIXEL_CATEGORY = None

zoom_start_epoch = 150
min_crop_size = 128
max_crop_size = 256
# 多尺度监督三阶段设置（阶段切换 epoch）
# 修改策略：为避免DDP参数未使用问题，所有epoch都计算所有尺度的损失，
# 但通过权重系数实现渐进式训练效果
#  - 阶段1 (epoch 0 ~ MULTI_SCALE_STAGE1_EPOCH-1): 多尺度损失权重=0.1（弱监督）
#  - 阶段2 (epoch MULTI_SCALE_STAGE1_EPOCH ~ MULTI_SCALE_STAGE2_EPOCH-1): 多尺度损失权重=0.5（中等监督）
#  - 阶段3 (epoch >= MULTI_SCALE_STAGE2_EPOCH): 多尺度损失权重=1.0（全监督）
MULTI_SCALE_STAGE1_EPOCH = 20   # 阶段2起始
MULTI_SCALE_STAGE2_EPOCH = 50   # 阶段3起始
# 各阶段的多尺度损失权重
MULTI_SCALE_WEIGHT_STAGE0 = 0.1  # 阶段1：弱监督
MULTI_SCALE_WEIGHT_STAGE1 = 0.5  # 阶段2：中等监督  
MULTI_SCALE_WEIGHT_STAGE2 = 1.0  # 阶段3：全监督


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="RGS-Net V3_1 分布式训练脚本")
    parser.add_argument("--data-root", type=str, default=DATA_ROOT, help="数据集根目录")
    parser.add_argument("--algorithm", type=str, default=ALGORITHM, 
                       choices=["voting", "Kumar-Roy", "Murphy", "Schroeder", "intersection", "Land8Fire"],
                       help="数据集算法类型")
    parser.add_argument("--fire-category", type=str, default=None,
                       choices=["very_few", "few", "many", "very_many"],
                       help="按火点像素数量选择子数据集 (very_few: 1-10, few: 10-100, many: 100-1000, very_many: >1000)")
    parser.add_argument("--bands", type=int, nargs=3, default=BANDS, help="选择的波段索引")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE, help="批大小")
    parser.add_argument("--num-workers", type=int, default=NUM_WORKERS, help="数据加载线程数")
    parser.add_argument("--epochs", type=int, default=EPOCHS, help="训练轮数")
    parser.add_argument("--lr", type=float, default=LEARNING_RATE, help="学习率")
    parser.add_argument("--save-dir", type=str, default=None, help="结果保存目录（默认自动生成）")
    return parser.parse_args()


def get_csv_paths(data_root: str, algorithm: str, fire_category: str = None):
    """
    获取训练/验证CSV文件路径
    
    Args:
        data_root: 数据集根目录
        algorithm: 算法类型
        fire_category: 火点数量类别 (None表示使用全量数据集)
    
    Returns:
        (train_csv, val_csv) 路径元组
    """
    if fire_category is None:
        # 使用全量数据集
        train_csv = os.path.join(data_root, f"{algorithm}_train.csv")
        val_csv = os.path.join(data_root, f"{algorithm}_val.csv")
    else:
        # 使用按火点数量划分的子数据集
        if algorithm == "Land8Fire":
            # Land8Fire 数据集路径结构
            subdir = os.path.join(data_root, "by_fire_pixels")
        else:
            # ActiveFire 数据集路径结构
            subdir = os.path.join(data_root, f"by_fire_pixels_{algorithm}")
        
        train_csv = os.path.join(subdir, "train", f"{algorithm}_train_{fire_category}.csv")
        val_csv = os.path.join(subdir, "val", f"{algorithm}_val_{fire_category}.csv")
    
    return train_csv, val_csv


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def setup_logging(rank: int, save_dir: str, enable_console: bool = False) -> logging.Logger:
    logger = logging.getLogger(f"rgs_v3_1_train_rank_{rank}")
    logger.setLevel(logging.DEBUG)

    log_dir = os.path.join(save_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)

    file_handler = logging.FileHandler(os.path.join(log_dir, f"rank_{rank}.log"), mode="w")
    file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(file_handler)

    if enable_console:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
        logger.addHandler(console_handler)

    return logger


def create_dataloaders(rank: int, world_size: int, data_root: str, algorithm: str, 
                       fire_category: str = None, bands: tuple = BANDS, 
                       batch_size: int = BATCH_SIZE, num_workers: int = NUM_WORKERS) -> Tuple[DataLoader, DataLoader, DistributedSampler]:
    train_csv, val_csv = get_csv_paths(data_root, algorithm, fire_category)

    train_dataset = LandsatFireDataset(train_csv, bands=bands)
    val_dataset = LandsatFireDataset(val_csv, bands=bands)

    train_sampler = DistributedSampler(
        train_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=SHUFFLE_TRAIN,
    )
    val_sampler = DistributedSampler(
        val_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=False,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True if num_workers > 0 else False,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        sampler=val_sampler,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True if num_workers > 0 else False,
        drop_last=True,
    )

    return train_loader, val_loader, train_sampler


def save_training_plot(train_losses, val_losses, save_dir: str) -> str:
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label="Training Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss Curves")
    plt.legend()
    plt.grid(True)
    plot_path = os.path.join(save_dir, "loss_curves.png")
    plt.savefig(plot_path)
    plt.close()
    return plot_path


def save_hyperparameters(
    save_dir: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    seg_criterion: torch.nn.Module,
    recon_criterion: torch.nn.Module,
    args = None,
) -> str:
    params_path = os.path.join(save_dir, "hyperparameters.txt")
    with open(params_path, "w") as f:
        f.write("======= 实验配置 =======\n")
        f.write(f"启动时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"实验ID: {RUN_ID}\n\n")

        f.write("======= 系统信息 =======\n")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        f.write(f"设备: {device}\n")
        if torch.cuda.is_available():
            f.write(f"GPU数量: {torch.cuda.device_count()}\n")
        f.write(f"PyTorch版本: {torch.__version__}\n\n")

        f.write("======= 模型信息 =======\n")
        f.write(f"模型架构: {type(model).__name__}\n")
        f.write(f"总参数量: {sum(p.numel() for p in model.parameters()):,}\n")
        f.write(f"可训练参数量: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}\n")
        f.write(f"重建权重参数: {model.module.rec_weight if isinstance(model, DDP) else model.rec_weight}\n\n")

        f.write("======= 数据配置 =======\n")
        if args:
            f.write(f"数据路径: {args.data_root}\n")
            f.write(f"算法类型: {args.algorithm}\n")
            f.write(f"火点数量类别: {args.fire_category if args.fire_category else '全量数据集'}\n")
            f.write(f"批大小: {args.batch_size}\n")
            f.write(f"数据加载线程数: {args.num_workers}\n")
            f.write(f"波段选择: {args.bands}\n")
        else:
            f.write(f"数据路径: {DATA_ROOT}\n")
            f.write(f"算法类型: {ALGORITHM}\n")
            f.write(f"火点数量类别: 全量数据集\n")
            f.write(f"批大小: {BATCH_SIZE}\n")
            f.write(f"数据加载线程数: {NUM_WORKERS}\n")
            f.write(f"波段选择: {BANDS}\n")
        f.write(f"训练集是否打乱: {'是' if SHUFFLE_TRAIN else '否'}\n\n")

        f.write("======= 训练配置 =======\n")
        epochs = args.epochs if args else EPOCHS
        f.write(f"总训练轮次: {epochs}\n")
        f.write("多尺度监督三阶段渐进配置（权重渐进策略）:\n")
        f.write(f"  - 阶段1 (epoch 0-{MULTI_SCALE_STAGE1_EPOCH-1}): 多尺度损失权重={MULTI_SCALE_WEIGHT_STAGE0}（弱监督）\n")
        f.write(f"  - 阶段2 (epoch {MULTI_SCALE_STAGE1_EPOCH}-{MULTI_SCALE_STAGE2_EPOCH-1}): 多尺度损失权重={MULTI_SCALE_WEIGHT_STAGE1}（中等监督）\n")
        f.write(f"  - 阶段3 (epoch {MULTI_SCALE_STAGE2_EPOCH}+): 多尺度损失权重={MULTI_SCALE_WEIGHT_STAGE2}（全监督）\n")
        f.write(f"验证间隔: 每 {VAL_INTERVAL} 个 epoch\n")
        f.write(f"保存间隔: 每 {SAVE_INTERVAL} 个 epoch\n")
        f.write("早停设置:\n")
        f.write(f"  - 耐心值: {EARLY_STOPPING_PATIENCE}\n")
        f.write(f"  - 最小改善: {EARLY_STOPPING_MIN_DELTA}\n")
        f.write("重建分支损失: MaskedL1Loss\n\n")

        f.write("======= 优化器配置 =======\n")
        f.write(f"优化器类型: {type(optimizer).__name__}\n")
        f.write(f"学习率: {optimizer.param_groups[0].get('lr', LEARNING_RATE)}\n")
        if "weight_decay" in optimizer.param_groups[0]:
            f.write(f"权重衰减: {optimizer.param_groups[0]['weight_decay']}\n")
        f.write("\n")

        f.write("======= 损失函数 =======\n")
        f.write("[Segmentation Loss]\n")
        f.write(get_criterion_info(seg_criterion) + "\n\n")
        f.write("[Reconstruction Loss]\n")
        f.write(get_criterion_info(recon_criterion) + "\n")
        f.write("\n" + "=" * 40 + "\n")

    return params_path


def train(rank: int, world_size: int, args) -> None:
    local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.device(f"cuda:{local_rank}")

    # 使用命令行参数更新全局配置
    data_root = args.data_root
    algorithm = args.algorithm
    fire_category = args.fire_category
    bands = tuple(args.bands)
    batch_size = args.batch_size
    num_workers = args.num_workers
    epochs = args.epochs
    learning_rate = args.lr
    
    # 生成保存目录
    if args.save_dir:
        save_dir = args.save_dir
    else:
        category_suffix = f"_{fire_category}" if fire_category else ""
        save_dir = f"output/RGS_Net_V3_1/{algorithm}{category_suffix}_{RUN_ID}"

    logger = setup_logging(rank, save_dir, enable_console=False)
    set_seed(42 + rank)

    try:
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            world_size=world_size,
            rank=rank,
        )
        logger.info(f"Rank {rank}: Process group initialized, world_size={world_size}")

        torch.cuda.set_device(device)
        logger.info(f"Rank {rank}/{world_size} using device: {device} (local_rank={local_rank})")

        model = RGSNetV3_1(n_channels=len(bands), n_filters=64).to(device)
        ddp_model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)

        # 分割分支使用SpatialFocalTverskyLoss
        seg_criterion = SpatialFocalTverskyLoss(
            alpha_tversky=0.5,  # Tversky FN 权重（漏报惩罚）
            beta_tversky=0.5,  # Tversky FP 权重（误报惩罚）
            gamma_focal=1.6,  # Focal Loss γ 参数
            focal_alpha=0.85,  # Focal Loss α 参数
            lambda_focal=0.4,  # Focal Loss λ 参数
            lambda_tversky=0.6,  # Tversky Loss λ 参数
            weight_min=1.0,  # 最小权重
            weight_max=4.0,  # 最大权重
            area_gamma=1.5,  # 区域平衡参数
            background_weight=1.0,  # 背景权重
            connectivity=8  # 连通性
        )
        
        # 重建分支使用MaskedL1Loss
        recon_criterion = MaskedL1Loss(reduction="mean")
        
        optimizer = torch.optim.AdamW(ddp_model.parameters(), lr=learning_rate, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.8, patience=5)

        train_loader, val_loader, train_sampler = create_dataloaders(
            rank, world_size, data_root, algorithm, fire_category, bands, batch_size, num_workers
        )
        logger.info(
            "Rank %d: Data loaders created | Train batches: %d, Val batches: %d",
            rank,
            len(train_loader),
            len(val_loader),
        )

        if zoom_start_epoch >= epochs and rank == 0:
            logger.warning(
                "Zoom curriculum never triggers (zoom_start_epoch=%d, EPOCHS=%d)",
                zoom_start_epoch,
                epochs,
            )

        weights_dir = os.path.join(save_dir, "weights")
        if rank == 0:
            os.makedirs(save_dir, exist_ok=True)
            os.makedirs(weights_dir, exist_ok=True)
            param_path = save_hyperparameters(save_dir, model, optimizer, seg_criterion, recon_criterion, args)
            logger.info(f"Hyperparameters saved to: {param_path}")

        train_losses = []
        val_losses = []
        train_seg_losses = []
        train_rec_losses = []
        val_seg_losses = []
        val_rec_losses = []

        best_val_loss = float("inf")
        epochs_no_improve = 0
        early_stop = False
        stop_signal = torch.tensor(0, device=device)

        for epoch in range(epochs):
            if early_stop:
                logger.info(f"Rank {rank}: Early stopping triggered at epoch {epoch}.")
                break

            current_crop_size = max_crop_size
            if epoch >= zoom_start_epoch:
                progress = min(1.0, (epoch - zoom_start_epoch) / max(1, epochs - zoom_start_epoch))
                current_crop_size = int(max_crop_size - (max_crop_size - min_crop_size) * progress)

            train_sampler.set_epoch(epoch)
            ddp_model.train()

            epoch_loss = 0.0
            epoch_seg_loss = 0.0
            epoch_rec_loss = 0.0
            num_batches = 0

            train_iter = tqdm(
                train_loader,
                total=len(train_loader),
                desc=f"Train {epoch+1}/{epochs}",
                disable=(rank != 0),
                leave=False,
            )
            for batch_idx, (images, masks) in enumerate(train_iter):
                if current_crop_size < max_crop_size:
                    images, masks = adaptive_crop(images, masks, current_crop_size, max_crop_size)

                images = images.to(device, non_blocking=True)
                masks = (masks > 0).float()
                if masks.dim() == 3:
                    masks = masks.unsqueeze(1)
                masks = masks.to(device, non_blocking=True)

                optimizer.zero_grad()
                
                outputs = ddp_model(images, binarize=False)
                seg_loss = seg_criterion(outputs.seg_logits, masks)
                rec_loss = recon_criterion(outputs.rec_logits, images, masks)
                # 融合分支也需要计算损失，确保 rec_weight 参数有梯度
                fused_loss = seg_criterion(outputs.fused_logits, masks)

                # 多尺度损失（按三阶段策略启用）
                # 策略修改：所有epoch都计算所有尺度，但用权重控制监督强度
                multi_scale_seg_loss = 0.0
                multi_scale_rec_loss = 0.0

                ms_seg = outputs.multi_scale_seg_logits or []
                ms_rec = outputs.multi_scale_rec_logits or []
                n_ms = len(ms_seg)

                if n_ms > 0:
                    # 根据当前epoch确定多尺度损失权重
                    if epoch < MULTI_SCALE_STAGE1_EPOCH:
                        ms_weight = MULTI_SCALE_WEIGHT_STAGE0  # 阶段1：弱监督
                    elif epoch < MULTI_SCALE_STAGE2_EPOCH:
                        ms_weight = MULTI_SCALE_WEIGHT_STAGE1  # 阶段2：中等监督
                    else:
                        ms_weight = MULTI_SCALE_WEIGHT_STAGE2  # 阶段3：全监督

                    # 优化：批量处理所有尺度，避免循环中的条件判断
                    # 计算所有尺度的多尺度分割损失
                    ms_seg_losses = []
                    for scale_logits in ms_seg:
                        if scale_logits.shape[2:] != masks.shape[2:]:
                            scale_logits = F.interpolate(
                                scale_logits, size=masks.shape[2:], 
                                mode='bilinear', align_corners=False
                            )
                        ms_seg_losses.append(seg_criterion(scale_logits, masks))
                    if ms_seg_losses:
                        multi_scale_seg_loss = (sum(ms_seg_losses) / len(ms_seg_losses)) * ms_weight

                    # 计算所有尺度的多尺度重建损失
                    ms_rec_losses = []
                    for scale_logits in ms_rec:
                        if scale_logits.shape[2:] != images.shape[2:]:
                            scale_logits = F.interpolate(
                                scale_logits, size=images.shape[2:], 
                                mode='bilinear', align_corners=False
                            )
                        ms_rec_losses.append(recon_criterion(scale_logits, images, masks))
                    if ms_rec_losses:
                        multi_scale_rec_loss = (sum(ms_rec_losses) / len(ms_rec_losses)) * ms_weight

                loss = seg_loss + rec_loss + fused_loss + multi_scale_seg_loss + multi_scale_rec_loss
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(ddp_model.parameters(), max_norm=0.5)
                optimizer.step()

                epoch_loss += loss.detach().item()
                epoch_seg_loss += (seg_loss.detach().item() + fused_loss.detach().item())
                epoch_rec_loss += rec_loss.detach().item()
                num_batches += 1

                if rank == 0:
                    train_iter.set_postfix(
                        loss=f"{loss.detach().item():.4f}",
                        seg=f"{seg_loss.detach().item():.4f}",
                        rec=f"{rec_loss.detach().item():.4f}",
                        rec_w=f"{ddp_model.module.rec_weight.item():.3f}" if hasattr(ddp_model.module, 'rec_weight') else "N/A"
                    )
            
            epoch_loss /= max(1, num_batches)
            avg_seg_loss = epoch_seg_loss / max(1, num_batches)
            avg_rec_loss = epoch_rec_loss / max(1, num_batches)
            train_losses.append(epoch_loss)
            train_seg_losses.append(avg_seg_loss)
            train_rec_losses.append(avg_rec_loss)
            logger.info(
                "Rank %d: Epoch %d/%d | Avg Train Loss: %.6f (seg: %.6f, rec: %.6f)",
                rank,
                epoch + 1,
                epochs,
                epoch_loss,
                avg_seg_loss,
                avg_rec_loss,
            )

            if epoch % VAL_INTERVAL == 0:
                ddp_model.eval()
                val_loss = 0.0
                val_seg_loss = 0.0
                val_rec_loss = 0.0
                val_samples = 0

                with torch.no_grad():
                    val_iter = tqdm(
                        val_loader,
                        total=len(val_loader),
                        desc=f"Val   {epoch+1}/{epochs}",
                        disable=(rank != 0),
                        leave=False,
                    )
                    for images, masks in val_iter:
                        images = images.to(device, non_blocking=True)
                        masks = (masks > 0).float()
                        if masks.dim() == 3:
                            masks = masks.unsqueeze(1)
                        masks = masks.to(device, non_blocking=True)

                        outputs = ddp_model(images, binarize=False)
                        seg_loss = seg_criterion(outputs.seg_logits, masks)
                        rec_loss = recon_criterion(outputs.rec_logits, images, masks)
                        # 融合分支也需要计算损失，确保 rec_weight 参数有梯度
                        fused_loss = seg_criterion(outputs.fused_logits, masks)

                        # 多尺度损失（按三阶段策略启用）
                        # 策略修改：所有epoch都计算所有尺度，但用权重控制监督强度
                        multi_scale_seg_loss = 0.0
                        multi_scale_rec_loss = 0.0

                        ms_seg = outputs.multi_scale_seg_logits or []
                        ms_rec = outputs.multi_scale_rec_logits or []
                        n_ms = len(ms_seg)

                        if n_ms > 0:
                            # 根据当前epoch确定多尺度损失权重（与训练保持一致）
                            if epoch < MULTI_SCALE_STAGE1_EPOCH:
                                ms_weight = MULTI_SCALE_WEIGHT_STAGE0
                            elif epoch < MULTI_SCALE_STAGE2_EPOCH:
                                ms_weight = MULTI_SCALE_WEIGHT_STAGE1
                            else:
                                ms_weight = MULTI_SCALE_WEIGHT_STAGE2

                            # 计算所有尺度的多尺度分割损失
                            for scale_logits in ms_seg:
                                if scale_logits.shape[2:] != masks.shape[2:]:
                                    scale_logits = F.interpolate(scale_logits, size=masks.shape[2:], mode='bilinear', align_corners=False)
                                multi_scale_seg_loss += seg_criterion(scale_logits, masks)
                            multi_scale_seg_loss = (multi_scale_seg_loss / len(ms_seg)) * ms_weight

                            # 计算所有尺度的多尺度重建损失
                            for scale_logits in ms_rec:
                                if scale_logits.shape[2:] != images.shape[2:]:
                                    scale_logits = F.interpolate(scale_logits, size=images.shape[2:], mode='bilinear', align_corners=False)
                                multi_scale_rec_loss += recon_criterion(scale_logits, images, masks)
                            multi_scale_rec_loss = (multi_scale_rec_loss / len(ms_rec)) * ms_weight

                        loss = seg_loss + rec_loss + fused_loss + multi_scale_seg_loss + multi_scale_rec_loss
                        batch_size = images.size(0)
                        val_loss += loss.item() * batch_size
                        val_seg_loss += (seg_loss.item() + fused_loss.item()) * batch_size
                        val_rec_loss += rec_loss.item() * batch_size
                        val_samples += batch_size

                        if rank == 0:
                            val_iter.set_postfix(
                                loss=f"{loss.item():.4f}",
                                seg=f"{seg_loss.item():.4f}",
                                rec=f"{rec_loss.item():.4f}",
                                rec_weight=f"{ddp_model.module.rec_weight.item():.4f}" if hasattr(ddp_model.module, 'rec_weight') else "N/A"
                            )

                val_loss_tensor = torch.tensor(val_loss, device=device)
                val_seg_tensor = torch.tensor(val_seg_loss, device=device)
                val_rec_tensor = torch.tensor(val_rec_loss, device=device)
                val_samples_tensor = torch.tensor(val_samples, device=device)
                dist.all_reduce(val_loss_tensor, op=dist.ReduceOp.SUM)
                dist.all_reduce(val_seg_tensor, op=dist.ReduceOp.SUM)
                dist.all_reduce(val_rec_tensor, op=dist.ReduceOp.SUM)
                dist.all_reduce(val_samples_tensor, op=dist.ReduceOp.SUM)

                avg_val_loss = val_loss_tensor.item() / max(1, val_samples_tensor.item())
                avg_val_seg = val_seg_tensor.item() / max(1, val_samples_tensor.item())
                avg_val_rec = val_rec_tensor.item() / max(1, val_samples_tensor.item())
                scheduler.step(avg_val_loss)
                val_losses.append(avg_val_loss)
                val_seg_losses.append(avg_val_seg)
                val_rec_losses.append(avg_val_rec)
                logger.info(
                    "Rank %d: Epoch %d/%d | Avg Val Loss: %.6f (seg: %.6f, rec: %.6f)",
                    rank,
                    epoch + 1,
                    epochs,
                    avg_val_loss,
                    avg_val_seg,
                    avg_val_rec,
                )

                if rank == 0:
                    # 仅在进入阶段三（epoch >= MULTI_SCALE_STAGE2_EPOCH）后启用早停监控
                    early_stop_enabled = (epoch >= MULTI_SCALE_STAGE2_EPOCH)

                    # 在阶段三开始的第一个 epoch，将计数器与 best 清零，避免早期阶段影响早停判断
                    if epoch == MULTI_SCALE_STAGE2_EPOCH:
                        best_val_loss = float("inf")
                        epochs_no_improve = 0
                        logger.info(
                            "Entering Stage 3: enabling early stopping; reset best_val_loss and epochs_no_improve."
                        )

                    if best_val_loss - avg_val_loss > EARLY_STOPPING_MIN_DELTA:
                        best_val_loss = avg_val_loss
                        epochs_no_improve = 0
                        best_model_path = os.path.join(weights_dir, "model_best.pth")
                        torch.save(ddp_model.module.state_dict(), best_model_path)
                        logger.info(f"New best model saved! Val loss improved to {best_val_loss:.6f}")
                    else:
                        if early_stop_enabled:
                            epochs_no_improve += 1
                            logger.info(
                                "No improvement in validation loss for %d/%d epochs (early stopping active)",
                                epochs_no_improve,
                                EARLY_STOPPING_PATIENCE,
                            )
                            if epochs_no_improve >= EARLY_STOPPING_PATIENCE:
                                stop_signal.fill_(1)
                                logger.info(
                                    "Early stopping triggered (Stage 3+): No improvement for %d consecutive epochs",
                                    EARLY_STOPPING_PATIENCE,
                                )
                        else:
                            # 阶段 1/2 不累计早停计数，仅记录日志
                            logger.info(
                                "Stage < 3: early stopping disabled; not counting no-improve this epoch."
                            )

                dist.broadcast(stop_signal, src=0)
                if stop_signal.item() == 1:
                    early_stop = True
                    if rank != 0:
                        logger.info(f"Rank {rank}: Received early stop signal")

            if (epoch + 1) % SAVE_INTERVAL == 0 and rank == 0:
                checkpoint_path = os.path.join(weights_dir, f"model_epoch_{epoch + 1}.pth")
                torch.save(ddp_model.module.state_dict(), checkpoint_path)
                logger.info(f"Saved checkpoint to: {checkpoint_path}")

        if rank == 0:
            final_model_path = os.path.join(weights_dir, "model_final.pth")
            torch.save(ddp_model.module.state_dict(), final_model_path)
            logger.info(f"Final model saved to: {final_model_path}")

            plot_path = save_training_plot(train_losses, val_losses, save_dir)
            logger.info(f"Training plot saved to: {plot_path}")

            logger.info(f"Training completed. Results saved to: {save_dir}")
            if train_losses:
                logger.info(f"Final Training Loss: {train_losses[-1]:.6f}")
            if val_losses:
                logger.info(f"Final Validation Loss: {val_losses[-1]:.6f}")

        logger.info(f"Rank {rank}: Training process completed")

    except Exception:
        if "logger" in locals():
            logger.exception("Rank %d: Unhandled exception during training", rank)
        raise
    finally:
        if dist.is_initialized():
            try:
                dist.barrier()
            except Exception:
                pass
            dist.destroy_process_group()
        if "logger" in locals():
            logger.info(f"Rank {rank}: Process group destroyed")


if __name__ == "__main__":
    # 解析命令行参数
    args = parse_args()
    
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    if rank == 0:
        logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
        main_logger = logging.getLogger("rgs_v3_1_main")
        main_logger.info(f"Starting distributed RGS-Net V3_1 training with {world_size} GPUs via torchrun")
        main_logger.info(f"算法: {args.algorithm}, 火点类别: {args.fire_category if args.fire_category else '全量数据集'}")
        category_suffix = f"_{args.fire_category}" if args.fire_category else ""
        save_dir = args.save_dir if args.save_dir else f"output/RGS_Net_V3_1/{args.algorithm}{category_suffix}_{RUN_ID}"
        main_logger.info(f"Results will be saved to: {save_dir}")

    train(rank, world_size, args)