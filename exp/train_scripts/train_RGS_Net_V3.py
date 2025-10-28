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
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from dataset import LandsatFireDataset
from models.RGS_Net_V3 import RGSNetV3
from loss import FocalTverskyLoss, get_criterion_info
from utils import adaptive_crop

# 使用示例:
# 1) 分布式训练 (推荐，torchrun 会自动设置 RANK/WORLD_SIZE/LOCAL_RANK 等环境变量)
#    假设在一台有 4 张 GPU 的机器上运行:
#
#    torchrun --nproc_per_node=4 exp/train_scripts/train_RGS_Net_V3.py
#
#    或者显式设置可见 GPU（例如使用 GPU 3-5）:
#
#    CUDA_VISIBLE_DEVICES=3 torchrun --nproc_per_node=1 exp/train_scripts/train_RGS_Net_V3.py
#
# 2) 单卡快速调试（需要手动设置环境变量供脚本读取）:
#
#    RANK=0 WORLD_SIZE=1 LOCAL_RANK=0 python3 exp/train_scripts/train_RGS_Net_V3.py
#
# 3) 自定义参数/结果目录: 在脚本顶部修改常量 (DATA_ROOT, BANDS, SAVE_DIR 等) 或将这些参数
#    改造成命令行参数（当前脚本使用常量配置）。
#
# 注意:
# - 脚本默认使用 NCCL 后端进行分布式训练 (GPU)，在 CPU-only 环境上请确保使用单卡模式并设置
#   相应环境变量。
# - 训练中使用的损失函数为 FocalTverskyLoss，输入到损失的应当是 raw logits（脚本已按此规则实现）。



DATA_ROOT = "data/full"
ALGORITHM = "voting"
RUN_ID = datetime.now().strftime("%Y%m%d%H%M")
SAVE_DIR = f"output/RGS_Net_V3/{ALGORITHM}_{RUN_ID}"
BANDS = (7, 6, 5)
BATCH_SIZE = 64
NUM_WORKERS = 4
SHUFFLE_TRAIN = True
EPOCHS = 200
LEARNING_RATE = 3e-4
VAL_INTERVAL = 1
SAVE_INTERVAL = 10
EARLY_STOPPING_PATIENCE = 15
EARLY_STOPPING_MIN_DELTA = 0.001

zoom_start_epoch = 1500
min_crop_size = 128
max_crop_size = 256


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def setup_logging(rank: int, save_dir: str, enable_console: bool = False) -> logging.Logger:
    logger = logging.getLogger(f"rgs_v3_train_rank_{rank}")
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


def create_dataloaders(rank: int, world_size: int) -> Tuple[DataLoader, DataLoader, DistributedSampler]:
    train_csv = os.path.join(DATA_ROOT, f"{ALGORITHM}_train.csv")
    val_csv = os.path.join(DATA_ROOT, f"{ALGORITHM}_val.csv")

    train_dataset = LandsatFireDataset(train_csv, bands=BANDS)
    val_dataset = LandsatFireDataset(val_csv, bands=BANDS)

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
        batch_size=BATCH_SIZE,
        sampler=train_sampler,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=True if NUM_WORKERS > 0 else False,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        sampler=val_sampler,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=True if NUM_WORKERS > 0 else False,
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
        f.write(f"可训练参数量: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}\n\n")

        f.write("======= 数据配置 =======\n")
        f.write(f"数据路径: {DATA_ROOT}\n")
        f.write(f"算法类型: {ALGORITHM}\n")
        f.write(f"批大小: {BATCH_SIZE}\n")
        f.write(f"数据加载线程数: {NUM_WORKERS}\n")
        f.write(f"训练集是否打乱: {'是' if SHUFFLE_TRAIN else '否'}\n")
        f.write(f"波段选择: {BANDS}\n\n")

        f.write("======= 训练配置 =======\n")
        f.write(f"总训练轮次: {EPOCHS}\n")
        f.write(f"验证间隔: 每 {VAL_INTERVAL} 个 epoch\n")
        f.write(f"保存间隔: 每 {SAVE_INTERVAL} 个 epoch\n")
        f.write("早停设置:\n")
        f.write(f"  - 耐心值: {EARLY_STOPPING_PATIENCE}\n")
        f.write(f"  - 最小改善: {EARLY_STOPPING_MIN_DELTA}\n")
        f.write("重建分支损失: FocalTverskyLoss (与分割一致)\n\n")

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


def train(rank: int, world_size: int) -> None:
    local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.device(f"cuda:{local_rank}")

    logger = setup_logging(rank, SAVE_DIR, enable_console=False)
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

        model = RGSNetV3(n_channels=len(BANDS), n_filters=64).to(device)
        ddp_model = DDP(model, device_ids=[local_rank], find_unused_parameters=False)

        seg_criterion = FocalTverskyLoss()
        recon_criterion = FocalTverskyLoss()
        optimizer = torch.optim.AdamW(ddp_model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.8, patience=5)

        train_loader, val_loader, train_sampler = create_dataloaders(rank, world_size)
        logger.info(
            "Rank %d: Data loaders created | Train batches: %d, Val batches: %d",
            rank,
            len(train_loader),
            len(val_loader),
        )

        if zoom_start_epoch >= EPOCHS and rank == 0:
            logger.warning(
                "Zoom curriculum never triggers (zoom_start_epoch=%d, EPOCHS=%d)",
                zoom_start_epoch,
                EPOCHS,
            )

        weights_dir = os.path.join(SAVE_DIR, "weights")
        if rank == 0:
            os.makedirs(SAVE_DIR, exist_ok=True)
            os.makedirs(weights_dir, exist_ok=True)
            param_path = save_hyperparameters(SAVE_DIR, model, optimizer, seg_criterion, recon_criterion)
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

        for epoch in range(EPOCHS):
            if early_stop:
                logger.info(f"Rank {rank}: Early stopping triggered at epoch {epoch}.")
                break

            current_crop_size = max_crop_size
            if epoch >= zoom_start_epoch:
                progress = min(1.0, (epoch - zoom_start_epoch) / max(1, EPOCHS - zoom_start_epoch))
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
                desc=f"Train {epoch+1}/{EPOCHS}",
                disable=(rank != 0),
                leave=False,
            )
            for images, masks in train_iter:
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
                background = 1.0 - masks
                rec_loss = recon_criterion(outputs.rec_logits, background)
                loss = seg_loss + rec_loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(ddp_model.parameters(), max_norm=0.5)
                optimizer.step()

                epoch_loss += loss.detach().item()
                epoch_seg_loss += seg_loss.detach().item()
                epoch_rec_loss += rec_loss.detach().item()
                num_batches += 1

                if rank == 0:
                    train_iter.set_postfix(
                        loss=f"{loss.detach().item():.4f}",
                        seg=f"{seg_loss.detach().item():.4f}",
                        rec=f"{rec_loss.detach().item():.4f}",
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
                EPOCHS,
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
                        desc=f"Val   {epoch+1}/{EPOCHS}",
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
                        background = 1.0 - masks
                        rec_loss = recon_criterion(outputs.rec_logits, background)
                        loss = seg_loss + rec_loss
                        batch_size = images.size(0)
                        val_loss += loss.item() * batch_size
                        val_seg_loss += seg_loss.item() * batch_size
                        val_rec_loss += rec_loss.item() * batch_size
                        val_samples += batch_size

                        if rank == 0:
                            val_iter.set_postfix(
                                loss=f"{loss.item():.4f}",
                                seg=f"{seg_loss.item():.4f}",
                                rec=f"{rec_loss.item():.4f}",
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
                    EPOCHS,
                    avg_val_loss,
                    avg_val_seg,
                    avg_val_rec,
                )

                if rank == 0:
                    if best_val_loss - avg_val_loss > EARLY_STOPPING_MIN_DELTA:
                        best_val_loss = avg_val_loss
                        epochs_no_improve = 0
                        best_model_path = os.path.join(weights_dir, "model_best.pth")
                        torch.save(ddp_model.module.state_dict(), best_model_path)
                        logger.info(f"New best model saved! Val loss improved to {best_val_loss:.6f}")
                    else:
                        epochs_no_improve += 1
                        logger.info(
                            "No improvement in validation loss for %d/%d epochs",
                            epochs_no_improve,
                            EARLY_STOPPING_PATIENCE,
                        )
                        if epochs_no_improve >= EARLY_STOPPING_PATIENCE:
                            stop_signal.fill_(1)
                            logger.info(
                                "Early stopping triggered! No improvement for %d consecutive epochs",
                                EARLY_STOPPING_PATIENCE,
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
            final_model_path = os.path.join(SAVE_DIR, "model_final.pth")
            torch.save(ddp_model.module.state_dict(), final_model_path)
            logger.info(f"Final model saved to: {final_model_path}")

            plot_path = save_training_plot(train_losses, val_losses, SAVE_DIR)
            logger.info(f"Training plot saved to: {plot_path}")

            logger.info(f"Training completed. Results saved to: {SAVE_DIR}")
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
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    if rank == 0:
        logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
        main_logger = logging.getLogger("rgs_v3_main")
        main_logger.info(f"Starting distributed RGS-Net V3 training with {world_size} GPUs via torchrun")
        main_logger.info(f"Results will be saved to: {SAVE_DIR}")

    train(rank, world_size)
