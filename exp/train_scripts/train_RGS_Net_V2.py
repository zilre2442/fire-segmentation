import os
import argparse
from datetime import datetime
from typing import Tuple, Optional
import torch
import torch.distributed as dist
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader

from dataset import LandsatFireDataset
from models.RGS_Net_V2 import RGSNetV2
from utils import adaptive_crop
from loss import FocalTverskyLoss,MaskedL1Loss, get_criterion_info
import logging

# 参考 Unet_2 的分布式训练脚本，适配 RGSNetV2（V2 模型）

# 全局超参数配置
DATA_ROOT = "data/full"
ALGORITHM = "voting"  # 可选项: 'Kumar-Roy', 'Murphy', 'Schroeder', 'intersection', 'voting'
RUN_ID = datetime.now().strftime('%Y%m%d%H%M')
SAVE_DIR = f"output/RGS_Net_V2/{ALGORITHM}_{RUN_ID}"
BATCH_SIZE = 64
NUM_WORKERS = 4
SHUFFLE_TRAIN = True
EPOCHS = 200
LEARNING_RATE = 3e-4
VAL_INTERVAL = 1
SAVE_INTERVAL = 10

# 早停
EARLY_STOPPING_PATIENCE = 15
EARLY_STOPPING_MIN_DELTA = 0.001

# 分割-重建混合损失权重
RECON_LOSS_WEIGHT = 1.0

# 融合温度
TAU = 1.0

# Zoom Curriculum Learning（若不需要，将 zoom_start_epoch 设为极大）
zoom_start_epoch = 1500
min_crop_size = 128
max_crop_size = 256


def compute_total_loss(
    outputs,
    images: torch.Tensor,
    masks: torch.Tensor,
    seg_loss_fn: torch.nn.Module,
    recon_loss_fn: torch.nn.Module,
    recon_weight: float,
    reduction: str = "mean",
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """组合分割与掩膜 L1 重建损失."""
    seg_loss = seg_loss_fn(outputs.seg_logits, masks)
    rec_loss = recon_loss_fn(outputs.reconstruction, images, masks, reduction=reduction)
    total_loss = seg_loss + recon_weight * rec_loss
    return total_loss, seg_loss, rec_loss


def setup_logging(rank, save_dir, enable_console=False):
    logger = logging.getLogger(f'train_rank_{rank}')
    logger.setLevel(logging.DEBUG)
    log_dir = os.path.join(save_dir, 'logs')
    os.makedirs(log_dir, exist_ok=True)
    file_handler = logging.FileHandler(os.path.join(log_dir, f'rank_{rank}.log'), mode='w')
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)
    # 训练过程不在终端输出日志，仅写文件；如需开启控制台输出，设置 enable_console=True
    if enable_console:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(logging.Formatter('%(levelname)s: %(message)s'))
        logger.addHandler(console_handler)
    return logger


def create_dataloaders(rank, world_size):
    train_csv = os.path.join(DATA_ROOT, f"{ALGORITHM}_train.csv")
    val_csv = os.path.join(DATA_ROOT, f"{ALGORITHM}_val.csv")

    train_dataset = LandsatFireDataset(train_csv, bands=(7, 6, 5))
    val_dataset = LandsatFireDataset(val_csv, bands=(7, 6, 5))

    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=SHUFFLE_TRAIN)
    val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        sampler=train_sampler,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        sampler=val_sampler,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
    )
    return train_loader, val_loader, train_sampler


def save_training_plot(train_losses, val_losses, save_dir):
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss Curves (RGS_Net_V2)')
    plt.legend()
    plt.grid(True)
    plot_path = os.path.join(save_dir, 'loss_curves.png')
    plt.savefig(plot_path)
    plt.close()
    return plot_path


def save_hyperparameters(save_dir, model, optimizer, criterion, tau: Optional[float] = None):
    params_path = os.path.join(save_dir, 'hyperparameters.txt')
    with open(params_path, 'w') as f:
        f.write("======= 实验配置 (RGS_Net_V2) =======\n")
        f.write(f"启动时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"实验ID: {RUN_ID}\n\n")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        f.write("======= 系统信息 =======\n")
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
        f.write(f"批大小(Batch Size): {BATCH_SIZE}\n")
        f.write(f"数据加载线程数: {NUM_WORKERS}\n")
        f.write(f"训练集是否打乱: {'是' if SHUFFLE_TRAIN else '否'}\n\n")
        f.write("======= 训练配置 =======\n")
        f.write(f"总训练轮次: {EPOCHS}\n")
        f.write(f"验证间隔: 每 {VAL_INTERVAL} 个epoch验证一次\n")
        f.write(f"保存间隔: 每 {SAVE_INTERVAL} 个epoch保存一次\n")
        f.write("早停设置:\n")
        f.write(f"  - 耐心值(Patience): {EARLY_STOPPING_PATIENCE}\n")
        f.write(f"  - 最小改善(Min Delta): {EARLY_STOPPING_MIN_DELTA}\n\n")
        f.write(f"重建损失权重: {RECON_LOSS_WEIGHT}\n")
        f.write(f"融合温度 (tau): {tau if tau is not None else TAU}\n\n")
        f.write("======= 优化器配置 =======\n")
        if optimizer is not None:
            f.write(f"优化器类型: {type(optimizer).__name__}\n")
            f.write(f"学习率: {optimizer.param_groups[0].get('lr', LEARNING_RATE)}\n")
            if 'weight_decay' in optimizer.param_groups[0]:
                f.write(f"权重衰减: {optimizer.param_groups[0]['weight_decay']}\n")
            if 'betas' in optimizer.param_groups[0]:
                f.write(f"Betas参数: {optimizer.param_groups[0]['betas']}\n")
        else:
            f.write("优化器: 未提供\n")
        f.write("\n======= 损失函数 =======\n")
        f.write(get_criterion_info(criterion))
        f.write("\n\n" + "=" * 40 + "\n")
    return params_path


def train(rank, world_size, tau: Optional[float] = None):
    try:
        # 不在终端输出日志，仅写日志文件
        logger = setup_logging(rank, SAVE_DIR, enable_console=False)

        # 初始化分布式
        dist.init_process_group(backend="nccl", init_method="env://", world_size=world_size, rank=rank)
        logger.info(f"Rank {rank}: Process group initialized (world_size={world_size})")

        local_rank = int(os.environ["LOCAL_RANK"])
        device = torch.device(f'cuda:{local_rank}' if torch.cuda.is_available() else 'cpu')
        if torch.cuda.is_available():
            torch.cuda.set_device(device)
        logger.info(f"Rank {rank}: using device {device}")

        # 模型/优化器
        model_tau = TAU if tau is None else tau
        model = RGSNetV2(n_channels=3, n_classes=1, n_filters=32, tau=model_tau).to(device)
        ddp_model = DDP(model, device_ids=[local_rank] if torch.cuda.is_available() else None, find_unused_parameters=False)

        seg_criterion = FocalTverskyLoss(
            alpha=0.6,##alpha超参数迁移
            beta=0.4,##beta超参数迁移
            gamma=1.6,##gamma超参数迁移
            focal_alpha=0.85,##focal_alpha超参数迁移
            lambda_focal=0.3,##lambda_focal超参数迁移
            lambda_tversky=0.7##lambda_tversky超参数迁移
        ).to(device)  # 确保损失函数在正确设备上
        # seg_criterion = torch.nn.DiceLoss().to(device)
        recon_criterion = MaskedL1Loss()

        optimizer = torch.optim.AdamW(ddp_model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.8, patience=5)

        train_loader, val_loader, train_sampler = create_dataloaders(rank, world_size)
        logger.info(f"Rank {rank}: Data loaders ready | train={len(train_loader)}, val={len(val_loader)}")

        if rank == 0:
            os.makedirs(SAVE_DIR, exist_ok=True)
            param_path = save_hyperparameters(SAVE_DIR, model, optimizer, seg_criterion, tau=model_tau)
            logger.info(f"Hyperparameters saved to: {param_path}")

        train_losses = []
        val_losses = []
        best_val_loss = float('inf')
        epochs_no_improve = 0
        early_stop = False
        stop_signal = torch.tensor(0, device=device)

        for epoch in range(EPOCHS):
            if early_stop:
                logger.info(f"Rank {rank}: Early stop at epoch {epoch}")
                break

            # Zoom curriculum
            if epoch < zoom_start_epoch:
                current_crop_size = max_crop_size
            else:
                progress = min(1.0, (epoch - zoom_start_epoch) / (EPOCHS - zoom_start_epoch))
                current_crop_size = int(max_crop_size - (max_crop_size - min_crop_size) * progress)

            train_sampler.set_epoch(epoch)
            ddp_model.train()
            epoch_loss = 0.0
            epoch_seg = 0.0
            epoch_rec = 0.0
            n_batches = 0

            # 仅在 rank0 上显示进度条，其他进程静默
            train_iter = tqdm(
                train_loader,
                total=len(train_loader),
                desc=f"Train {epoch+1}/{EPOCHS}",
                disable=(rank != 0)
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
                outputs = ddp_model(images, fuse_outputs=False)
                loss, seg_loss, rec_loss = compute_total_loss(
                    outputs, images, masks, seg_criterion, recon_criterion, RECON_LOSS_WEIGHT, reduction="mean"
                )
                loss.backward()
                torch.nn.utils.clip_grad_norm_(ddp_model.parameters(), max_norm=0.5)
                optimizer.step()

                batch_total = loss.detach().item()
                batch_seg = seg_loss.detach().item()
                batch_rec = rec_loss.detach().item()
                epoch_loss += batch_total
                epoch_seg += batch_seg
                epoch_rec += batch_rec
                n_batches += 1
                if rank == 0:
                    train_iter.set_postfix(
                        loss=f"{batch_total:.4f}", seg=f"{batch_seg:.4f}", rec=f"{batch_rec:.4f}"
                    )

            epoch_loss /= max(1, n_batches)
            epoch_seg /= max(1, n_batches)
            epoch_rec /= max(1, n_batches)
            train_losses.append(epoch_loss)
            if rank == 0:
                logger.info(
                    f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {epoch_loss:.6f} (seg: {epoch_seg:.6f}, rec: {epoch_rec:.6f})"
                )

            # 验证
            if epoch % VAL_INTERVAL == 0:
                ddp_model.eval()
                val_loss = 0.0
                val_seg = 0.0
                val_rec = 0.0
                val_samples = 0
                with torch.no_grad():
                    val_iter = tqdm(
                        val_loader,
                        total=len(val_loader),
                        desc=f"Val   {epoch+1}/{EPOCHS}",
                        disable=(rank != 0)
                    )
                    for images, masks in val_iter:
                        bs = images.size(0)
                        images = images.to(device, non_blocking=True)
                        masks = (masks > 0).float()
                        if masks.dim() == 3:
                            masks = masks.unsqueeze(1)
                        masks = masks.to(device, non_blocking=True)

                        outputs = ddp_model(images, fuse_outputs=False)
                        tot, seg_l, rec_l = compute_total_loss(
                            outputs, images, masks, seg_criterion, recon_criterion, RECON_LOSS_WEIGHT, reduction="mean"
                        )
                        b_tot = tot.item()
                        b_seg = seg_l.item()
                        b_rec = rec_l.item()
                        val_loss += b_tot * bs
                        val_seg += b_seg * bs
                        val_rec += b_rec * bs
                        val_samples += bs
                        if rank == 0:
                            val_iter.set_postfix(loss=f"{b_tot:.4f}", seg=f"{b_seg:.4f}", rec=f"{b_rec:.4f}")

                # all-reduce
                val_loss_t = torch.tensor(val_loss, device=device)
                val_seg_t = torch.tensor(val_seg, device=device)
                val_rec_t = torch.tensor(val_rec, device=device)
                val_cnt_t = torch.tensor(val_samples, device=device)
                dist.all_reduce(val_loss_t, op=dist.ReduceOp.SUM)
                dist.all_reduce(val_seg_t, op=dist.ReduceOp.SUM)
                dist.all_reduce(val_rec_t, op=dist.ReduceOp.SUM)
                dist.all_reduce(val_cnt_t, op=dist.ReduceOp.SUM)

                avg_val_loss = (val_loss_t / val_cnt_t).item() if val_cnt_t.item() > 0 else float('nan')
                avg_val_seg = (val_seg_t / val_cnt_t).item() if val_cnt_t.item() > 0 else float('nan')
                avg_val_rec = (val_rec_t / val_cnt_t).item() if val_cnt_t.item() > 0 else float('nan')
                val_losses.append(avg_val_loss)
                scheduler.step(avg_val_loss)

                if rank == 0:
                    logger.info(
                        f"Epoch {epoch+1}/{EPOCHS} | Val Loss: {avg_val_loss:.6f} (seg: {avg_val_seg:.6f}, rec: {avg_val_rec:.6f})"
                    )

                # 早停 & 保存 best
                if rank == 0:
                    if best_val_loss - avg_val_loss > EARLY_STOPPING_MIN_DELTA:
                        best_val_loss = avg_val_loss
                        epochs_no_improve = 0
                        best_path = os.path.join(SAVE_DIR, "model_best.pth")
                        torch.save(ddp_model.module.state_dict(), best_path)
                        logger.info(f"New best saved to {best_path}")
                    else:
                        epochs_no_improve += 1
                        logger.info(
                            f"No improvement for {epochs_no_improve}/{EARLY_STOPPING_PATIENCE} epochs (best: {best_val_loss:.6f})"
                        )
                        if epochs_no_improve >= EARLY_STOPPING_PATIENCE:
                            stop_signal.fill_(1)
                            logger.info("Early stopping triggered by rank 0")

                dist.broadcast(stop_signal, src=0)
                if stop_signal.item() == 1:
                    early_stop = True

            if (epoch + 1) % SAVE_INTERVAL == 0 and rank == 0:
                ckpt_path = os.path.join(SAVE_DIR, f"model_epoch_{epoch+1}.pth")
                torch.save(ddp_model.module.state_dict(), ckpt_path)
                logger.info(f"Saved checkpoint: {ckpt_path}")

        # 完结保存
        if rank == 0:
            final_model_path = os.path.join(SAVE_DIR, "model_final.pth")
            torch.save(ddp_model.module.state_dict(), final_model_path)
            logger.info(f"Final model saved: {final_model_path}")
            plot_path = save_training_plot(train_losses, val_losses, SAVE_DIR)
            logger.info(f"Training plot saved: {plot_path}")
    finally:
        if dist.is_initialized():
            try:
                dist.barrier()
            except Exception:
                pass
            try:
                dist.destroy_process_group()
            except Exception:
                pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tau", type=float, default=None, help="融合温度 tau（覆盖文件中 TAU 的默认值）")
    args = parser.parse_args()

    rank = int(os.environ['RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    # 不在终端配置全局日志，避免终端输出；仅写入文件日志
    try:
        train(rank, world_size, tau=args.tau)
    except Exception:
        try:
            os.makedirs(os.path.join(SAVE_DIR, 'logs'), exist_ok=True)
            import traceback
            tb = traceback.format_exc()
            err_path = os.path.join(SAVE_DIR, 'logs', f'exception_rank_{os.environ.get("LOCAL_RANK","0")}.txt')
            with open(err_path, 'w') as ef:
                ef.write(tb)
        except Exception:
            pass
        raise

# CUDA_VISIBLE_DEVICES=3,4,5,6 torchrun --nproc_per_node=4 exp/train_scripts/train_RGS_Net_V2.py --tau 1
