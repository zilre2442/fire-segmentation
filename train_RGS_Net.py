import os
from datetime import datetime
from typing import Tuple
import torch
import torch.distributed as dist
import matplotlib.pyplot as plt
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
from dataset import LandsatFireDataset
from models.RGS_Net import RGSNet
from utils import adaptive_crop
from loss import FocalTverskyLoss, MaskedL1Loss, get_criterion_info
import logging

# GPU设置
os.environ["CUDA_VISIBLE_DEVICES"] = "2,3,5"

# 全局超参数配置
DATA_ROOT = "data/full"
ALGORITHM = "voting"  # 可选项: 'Kumar-Roy', 'Murphy', 'Schroeder', 'intersection', 'voting'
RUN_ID = datetime.now().strftime('%Y%m%d%H%M')
SAVE_DIR = f"output/RGS_Net/{ALGORITHM}_{RUN_ID}"
BATCH_SIZE = 128
NUM_WORKERS = 4
SHUFFLE_TRAIN = True
EPOCHS = 200
LEARNING_RATE = 3e-4  #LEARNING_RATE超参数迁移

VAL_INTERVAL = 1
SAVE_INTERVAL = 10  # 每10个epoch保存一次
# ================ 新增全局早停参数 ================
EARLY_STOPPING_PATIENCE = 15    # 允许验证损失未改善的epoch数
EARLY_STOPPING_MIN_DELTA = 0.001  # 视为改善的最小损失变化阈值

BANDS = (7, 6, 2)  # 使用的波段组合
RECON_LOSS_WEIGHT = 1.0  # 重建损失权重 
TAU = 1.0  # 重建分支融合系数（越大重建分支影像越小，对分割概率的调整越温和）

# Zoom Curriculum Learning
zoom_start_epoch = 1500    # 开始应用Zoom的epoch, 不使用就给大一点
min_crop_size = 128        # 最小裁剪尺寸
max_crop_size = 256        # 最大裁剪尺寸

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
    """为每个进程设置单独的日志记录器"""
    logger = logging.getLogger(f'train_rank_{rank}')
    logger.setLevel(logging.DEBUG)
    
    # 创建日志目录
    log_dir = os.path.join(save_dir, 'logs')
    os.makedirs(log_dir, exist_ok=True)
    
    # 所有进程的文件日志
    file_handler = logging.FileHandler(
        os.path.join(log_dir, f'rank_{rank}.log'), 
        mode='w'
    )
    file_handler.setFormatter(logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s'
    ))
    logger.addHandler(file_handler)
    
    # 可选控制台输出
    if enable_console:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(logging.Formatter(
            '%(levelname)s: %(message)s'
        ))
        logger.addHandler(console_handler)
    
    return logger

def create_dataloaders(rank, world_size):
    train_csv = os.path.join(DATA_ROOT, f"{ALGORITHM}_train.csv")
    val_csv = os.path.join(DATA_ROOT, f"{ALGORITHM}_val.csv")

    train_dataset = LandsatFireDataset(train_csv, bands=BANDS)
    val_dataset = LandsatFireDataset(val_csv, bands=BANDS)

    train_sampler = DistributedSampler(
        train_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=SHUFFLE_TRAIN
    )
    
    val_sampler = DistributedSampler(
        val_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=False
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        sampler=train_sampler,
        num_workers=NUM_WORKERS,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        sampler=val_sampler,
        num_workers=NUM_WORKERS,
        drop_last=True  # 确保所有进程有相同数量的batch
    )
    
    return train_loader, val_loader, train_sampler

def save_training_plot(train_losses, val_losses, save_dir):
    """保存训练曲线图"""
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss Curves')
    plt.legend()
    plt.grid(True)
    plot_path = os.path.join(save_dir, 'loss_curves.png')
    plt.savefig(plot_path)
    plt.close()
    return plot_path

def save_hyperparameters(
    save_dir,
    model,
    optimizer,
    seg_criterion=None,
    recon_criterion=None,
):
    """保存详细的超参数配置到TXT文件

    Args:
        save_dir (str): 保存目录路径
        model (nn.Module): PyTorch模型实例
        optimizer (torch.optim.Optimizer, optional): 优化器实例
        seg_criterion (torch.nn.Module, optional): 分割分支损失函数
        recon_criterion (torch.nn.Module, optional): 重建分支损失函数
    """
    params_path = os.path.join(save_dir, 'hyperparameters.txt')
    with open(params_path, 'w') as f:
        # 写入基础信息
        f.write("======= 实验配置 =======\n")
        f.write(f"启动时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"实验ID: {RUN_ID}\n\n")
        
        # 系统信息部分
        f.write("======= 系统信息 =======\n")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        f.write(f"设备: {device}\n")
        if torch.cuda.is_available():
            f.write(f"GPU数量: {torch.cuda.device_count()}\n")
        f.write(f"PyTorch版本: {torch.__version__}\n\n")
        
        # 模型信息部分
        f.write("======= 模型信息 =======\n")
        f.write(f"模型架构: {type(model).__name__}\n")
        f.write(f"总参数量: {sum(p.numel() for p in model.parameters()):,}\n")
        f.write(f"可训练参数量: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}\n")
        f.write("\n")
        
        # 数据信息部分
        f.write("======= 数据配置 =======\n")
        f.write(f"数据路径: {DATA_ROOT}\n")
        f.write(f"算法类型: {ALGORITHM}\n")
        f.write(f"批大小(Batch Size): {BATCH_SIZE}\n")
        f.write(f"数据加载线程数: {NUM_WORKERS}\n")
        f.write(f"训练集是否打乱: {'是' if SHUFFLE_TRAIN else '否'}\n\n")
        
        # 训练配置部分
        f.write("======= 训练配置 =======\n")
        f.write(f"总训练轮次: {EPOCHS}\n")
        f.write(f"验证间隔: 每 {VAL_INTERVAL} 个epoch验证一次\n")
        f.write(f"保存间隔: 每 {SAVE_INTERVAL} 个epoch保存一次\n")
        f.write("早停设置:\n")
        f.write(f"  - 耐心值(Patience): {EARLY_STOPPING_PATIENCE}\n")
        f.write(f"  - 最小改善(Min Delta): {EARLY_STOPPING_MIN_DELTA}\n")
        f.write(f"重建损失权重: {RECON_LOSS_WEIGHT}\n")
        f.write(f"TAU (重建分支融合系数): {TAU}\n\n")
        f.write(f"Bands used: {BANDS}\n\n")
        
        # 优化器信息
        f.write("======= 优化器配置 =======\n")
        if optimizer is not None:
            f.write(f"优化器类型: {type(optimizer).__name__}\n")
            f.write(f"学习率: {optimizer.param_groups[0].get('lr', LEARNING_RATE)}\n")
            if 'weight_decay' in optimizer.param_groups[0]:
                f.write(f"权重衰减: {optimizer.param_groups[0]['weight_decay']}\n")
            if 'momentum' in optimizer.param_groups[0]:
                f.write(f"动量: {optimizer.param_groups[0]['momentum']}\n")
            if 'betas' in optimizer.param_groups[0]:
                f.write(f"Betas参数: {optimizer.param_groups[0]['betas']}\n")
            if 'eps' in optimizer.param_groups[0]:
                f.write(f"Epsilon: {optimizer.param_groups[0]['eps']}\n")
        else:
            f.write("优化器: 未提供\n")
        f.write("\n")
        
        # 损失函数信息
        f.write("======= 损失函数 =======\n")
        if seg_criterion is not None:
            f.write("[Segmentation Loss]\n")
            f.write(get_criterion_info(seg_criterion) + "\n")
        else:
            f.write("Segmentation Loss: 未提供\n")

        f.write("\n")
        if recon_criterion is not None:
            f.write("[Reconstruction Loss]\n")
            f.write(get_criterion_info(recon_criterion) + "\n")
        else:
            f.write("Reconstruction Loss: 未提供\n")
        f.write("\n")
        
        # 添加分隔线
        f.write("\n" + "="*40 + "\n")
    
    return params_path

def train(rank, world_size):
    try:
        # 设置日志
        logger = setup_logging(rank, SAVE_DIR, enable_console=(rank == 0))
        
        # === 初始化进程组 ===
        dist.init_process_group(
            backend="nccl",  # GPU 推荐使用 nccl
            init_method="env://",
            world_size=world_size,
            rank=rank
        )
        logger.info(f"Rank {rank}: Process group initialized by torchrun, world_size={world_size}")
        
        # === 直接使用环境变量中的 LOCAL_RANK ===
        local_rank = int(os.environ["LOCAL_RANK"])
        device = torch.device(f'cuda:{local_rank}')
        torch.cuda.set_device(device)
        
        logger.info(f"Rank {rank}/{world_size} using device: {device} (local_rank={local_rank})")
        
        # 创建模型和优化器
        model = RGSNet(n_channels=3, n_classes=1, n_filters=32,tau=TAU).to(device)  # n_channels、n_classes、n_filters超参数迁移
        ddp_model = DDP(
            model,
            device_ids=[local_rank],
            find_unused_parameters=False  # 允许检测未使用的参数
        )  # 使用local_rank
        
        # seg_criterion = SpatialFocalLoss(
        #     gamma=1.6,
        #     alpha=0.85,
        #     include_pred_mask=True,
        #     weight_max=7.0,
        #     weight_gamma=1.2,
        # ).to(device)
        recon_criterion = MaskedL1Loss()
        seg_criterion = FocalTverskyLoss(
            alpha=0.8,##alpha超参数迁移
            beta=0.4,##beta超参数迁移
            gamma=1.6,##gamma超参数迁移
            focal_alpha=0.85,##focal_alpha超参数迁移
            lambda_focal=0.3,##lambda_focal超参数迁移
            lambda_tversky=0.7##lambda_tversky超参数迁移
        ).to(device)  # 确保损失函数在正确设备上

        optimizer = torch.optim.AdamW(
            ddp_model.parameters(),
            lr=LEARNING_RATE,
            weight_decay=1e-4
        )

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.8,
            patience=5
        )

        train_loader, val_loader, train_sampler = create_dataloaders(rank, world_size)
        logger.info(f"Rank {rank}: Data loaders created | Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
        
        # 确保保存目录存在（仅rank0）
        if rank == 0:
            os.makedirs(SAVE_DIR, exist_ok=True)
            # 训练开始前就保存超参数
            param_path = save_hyperparameters(
                SAVE_DIR,
                model,
                optimizer,
                seg_criterion=seg_criterion,
                recon_criterion=recon_criterion,
            )
            logger.info(f"Hyperparameters saved to: {param_path}")

        # 初始化记录器
        train_losses = []
        val_losses = []
        batch_losses = []

        # ================== 早停机制初始化 ==================
        best_val_loss = float('inf')  # 初始设为无穷大
        epochs_no_improve = 0         # 未改善周期计数器
        early_stop = False            # 停止标志
        # 用于跨进程同步的停止信号Tensor
        stop_signal = torch.tensor(0).to(device)
        # =================================================
        
        # 训练循环
        for epoch in range(EPOCHS):
            # 检查早停信号
            if early_stop:
                logger.info(f"Rank {rank}: Early stopping triggered. Exiting training at epoch {epoch}.")
                break

            # 应用Zoom Curriculum Learning计算当前裁剪尺寸
            if epoch < zoom_start_epoch:
                current_crop_size = max_crop_size  # 前期使用全图
            else:
                progress = min(1.0, (epoch - zoom_start_epoch) / (EPOCHS - zoom_start_epoch))
                current_crop_size = int(max_crop_size - (max_crop_size - min_crop_size) * progress)

            train_sampler.set_epoch(epoch)
            ddp_model.train()
            epoch_train_loss = 0.0
            num_batches = 0
            
            epoch_seg_loss = 0.0
            epoch_rec_loss = 0.0

            for batch_idx, (images, masks) in enumerate(train_loader):
                # 应用Zoom Curriculum Learning裁剪
                if current_crop_size < max_crop_size:
                    images, masks = adaptive_crop(images, masks, current_crop_size, max_crop_size)

                # 数据移动到设备device
                images, masks = images.to(device), masks.to(device)
                masks = (masks > 0).float()

                optimizer.zero_grad()

                # ---- 前向传播 ----
                outputs = ddp_model(images, fuse_outputs=False)

                loss, seg_loss, rec_loss = compute_total_loss(
                    outputs,
                    images,
                    masks,
                    seg_criterion,
                    recon_criterion,
                    RECON_LOSS_WEIGHT,
                    reduction="mean",
                )

                seg_loss_value = seg_loss.detach().item()
                rec_loss_value = rec_loss.detach().item()
                total_loss_value = loss.detach().item()

                loss.backward()
                
                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(ddp_model.parameters(), max_norm=0.5)##max_norm超参数迁移
                
                optimizer.step()
                
                epoch_train_loss += total_loss_value
                epoch_seg_loss += seg_loss_value
                epoch_rec_loss += rec_loss_value
                num_batches += 1
                
                # 每10个batch记录一次（仅rank0）
                if batch_idx % 10 == 0:
                    batch_losses.append(total_loss_value)
                    logger.debug(
                        "Rank %d: Epoch %d/%d | Batch %d/%d | Loss: %.6f (seg: %.6f, rec: %.6f)",
                        rank,
                        epoch + 1,
                        EPOCHS,
                        batch_idx,
                        len(train_loader),
                        total_loss_value,
                        seg_loss_value,
                        rec_loss_value,
                    )
            
            # 计算平均训练损失
            epoch_train_loss /= num_batches
            avg_seg_loss = epoch_seg_loss / num_batches
            avg_rec_loss = epoch_rec_loss / num_batches
            train_losses.append(epoch_train_loss)
            logger.info(
                "Rank %d: Epoch %d/%d | Avg Train Loss: %.6f (seg: %.6f, rec: %.6f)",
                rank,
                epoch + 1,
                EPOCHS,
                epoch_train_loss,
                avg_seg_loss,
                avg_rec_loss,
            )

            # 验证阶段
            if epoch % VAL_INTERVAL == 0:
                ddp_model.eval()
                val_loss = 0.0
                val_seg = 0.0
                val_rec = 0.0
                val_samples = 0
                with torch.no_grad():
                    for images, masks in val_loader:
                        batch_size = images.size(0)
                        # 移动到设备device
                        images, masks = images.to(device), masks.to(device)
                        masks = (masks > 0).float()

                        # ---- 前向传播 ----
                        outputs = ddp_model(images, fuse_outputs=False)

                        total_loss, seg_loss_val_tensor, rec_loss_val_tensor = compute_total_loss(
                            outputs,
                            images,
                            masks,
                            seg_criterion,
                            recon_criterion,
                            RECON_LOSS_WEIGHT,
                            reduction="mean",
                        )

                        seg_loss_val = seg_loss_val_tensor.item()
                        rec_loss_val = rec_loss_val_tensor.item()
                        total_loss_val = total_loss.item()

                        val_loss += total_loss_val * batch_size
                        val_seg += seg_loss_val * batch_size
                        val_rec += rec_loss_val * batch_size
                        val_samples += batch_size
                
                # 同步所有进程的验证损失
                val_loss_tensor = torch.tensor(val_loss, device=device)
                val_seg_tensor = torch.tensor(val_seg, device=device)
                val_rec_tensor = torch.tensor(val_rec, device=device)
                val_samples_tensor = torch.tensor(val_samples).to(device)
                
                dist.all_reduce(val_loss_tensor, op=dist.ReduceOp.SUM)
                dist.all_reduce(val_seg_tensor, op=dist.ReduceOp.SUM)
                dist.all_reduce(val_rec_tensor, op=dist.ReduceOp.SUM)
                dist.all_reduce(val_samples_tensor, op=dist.ReduceOp.SUM)
                
                # 计算加权平均损失
                if val_samples_tensor > 0:
                    avg_val_loss = val_loss_tensor.item() / val_samples_tensor.item()
                    avg_val_seg = val_seg_tensor.item() / val_samples_tensor.item()
                    avg_val_rec = val_rec_tensor.item() / val_samples_tensor.item()
                else:
                    avg_val_loss = float('nan')
                    avg_val_seg = float('nan')
                    avg_val_rec = float('nan')
                
                scheduler.step(avg_val_loss)

                val_losses.append(avg_val_loss)
                logger.info(
                    "Rank %d: Epoch %d/%d | Avg Val Loss: %.6f (seg: %.6f, rec: %.6f)",
                    rank,
                    epoch + 1,
                    EPOCHS,
                    avg_val_loss,
                    avg_val_seg,
                    avg_val_rec,
                )

                # ================ 早停判断逻辑 ================
                # 仅在rank 0上决策早停
                if rank == 0:
                    # 检查是否达到最佳验证损失
                    if best_val_loss - avg_val_loss > EARLY_STOPPING_MIN_DELTA:
                        best_val_loss = avg_val_loss
                        epochs_no_improve = 0
                        # 保存最佳模型
                        best_model_path = os.path.join(SAVE_DIR, "weights", "model_best.pth")
                        torch.save(ddp_model.module.state_dict(), best_model_path)
                        logger.info(f"New best model saved! Val loss improved to {best_val_loss:.6f}")
                    else:
                        epochs_no_improve += 1
                        logger.info(f"No improvement in validation loss for {epochs_no_improve}/{EARLY_STOPPING_PATIENCE} epochs")
                        
                        # 检查是否触发早停
                        if epochs_no_improve >= EARLY_STOPPING_PATIENCE:
                            stop_signal.fill_(1)  # 设置停止信号
                            logger.info(f"Early stopping triggered! No improvement for {EARLY_STOPPING_PATIENCE} consecutive epochs")

                # 广播停止信号给所有进程
                dist.broadcast(stop_signal, src=0)
                
                # 所有进程检查停止信号
                if stop_signal.item() == 1:
                    early_stop = True
                    # 非零秩进程只需跳出循环
                    if rank != 0:
                        logger.info(f"Rank {rank}: Received early stop signal")

            # 保存一些特定轮数的模型
            if (epoch + 1) % SAVE_INTERVAL == 0 and rank == 0:
                checkpoint_path = os.path.join(SAVE_DIR, "weights", f"model_epoch_{epoch+1}.pth")
                torch.save(ddp_model.module.state_dict(), checkpoint_path)
                logger.info(f"Saved checkpoint to: {checkpoint_path}")
        
        # 训练结束处理（仅rank0）
        if rank == 0:
            # 保存最终模型
            final_model_path = os.path.join(SAVE_DIR, "model_final.pth")
            torch.save(ddp_model.module.state_dict(), final_model_path)
            logger.info(f"Final model saved to: {final_model_path}")
            
            # 保存训练曲线
            plot_path = save_training_plot(train_losses, val_losses, SAVE_DIR)
            logger.info(f"Training plot saved to: {plot_path}")
            
            # 记录最终性能
            logger.info(f"Training completed. Results saved to: {SAVE_DIR}")
            if train_losses:
                logger.info(f"Final Training Loss: {train_losses[-1]:.6f}")
            if val_losses:
                logger.info(f"Final Validation Loss: {val_losses[-1]:.6f}")
        
        logger.info(f"Rank {rank}: Training process completed")

    finally:
        # 确保在所有情况下都清理进程组
        if dist.is_initialized():
            # 等待所有进程完成
            dist.barrier(device_ids=[local_rank])
            dist.destroy_process_group()
            if 'logger' in locals():  # 确保logger已定义
                logger.info(f"Rank {rank}: Process group destroyed")

if __name__ == "__main__":
    # 从环境变量获取分布式参数
    rank = int(os.environ['RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    
    # 初始化主日志（仅用于启动信息）
    if rank == 0:
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        main_logger = logging.getLogger('main')
        main_logger.info(f"Starting distributed training with {world_size} GPUs via torchrun")
        main_logger.info(f"Results will be saved to: {SAVE_DIR}")
    
    # 启动训练
    train(rank, world_size)

# 命令行相关指令
# torchrun --nproc_per_node=3 train.py或者python -m torch.distributed.run --nproc_per_node=3 train_.py
# nohup torchrun --nproc_per_node=3 CaraUnet_2_train.py > CaraUnet_2_train_$(date +%Y%m%d%H%M).log 2>&1 &
# disown  # 立即执行：将进程从Shell作业列表中剥离
# nvidia-smi  # 查看GPU状态
# ps aux | grep CaraUnet_2_train.py  # 查看进程数