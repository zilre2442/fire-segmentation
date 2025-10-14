from collections import deque
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalTverskyLoss(nn.Module):
    def __init__(self, alpha=0.7, beta=0.3, gamma=2.0, focal_alpha=0.8,
                 lambda_focal=0.6, lambda_tversky=0.4, smooth=1e-6):
        """
        组合损失: λ1 * Focal Loss + λ2 * Tversky Loss\n
        此损失函数自带sigmod
        Args:
            alpha: Tversky假阴性权重(漏报惩罚)
            beta: Tversky假阳性权重(误报惩罚)
            gamma: Focal Loss聚焦参数
            focal_alpha: Focal Loss正样本权重
            lambda_focal: Focal Loss组合权重
            lambda_tversky: Tversky Loss组合权重
            smooth: 平滑系数防除零

        Example:
            >>> # 使用示例
            >>> criterion = FocalTverskyLoss(
            ...     alpha=0.7,  # 高漏报惩罚
            ...     beta=0.3,  # 低误报惩罚
            ...     gamma=2.0,  # 聚焦困难样本
            ...     focal_alpha=0.85,  # 正样本权重
            ...     lambda_focal=0.7,
            ...     lambda_tversky=0.3
            ... )
        """
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.focal_alpha = focal_alpha
        self.lambda_focal = lambda_focal
        self.lambda_tversky = lambda_tversky
        self.smooth = smooth

    def forward(self, pred, target):
        # 输入检查
        assert pred.size() == target.size(), "预测与标签尺寸不一致"

        # 获取概率图 (假设pred未归一化)
        pred = torch.sigmoid(pred)

        # ===== 1. 计算Focal Loss =====
        ce_loss = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
        p_t = torch.where(target == 1, pred, 1 - pred)
        focal_weight = (1 - p_t) ** self.gamma
        focal_loss = focal_weight * ce_loss

        # 正样本加权
        alpha_factor = torch.where(target == 1, self.focal_alpha, 1 - self.focal_alpha)
        focal_loss = alpha_factor * focal_loss

        # ===== 2. 计算Tversky Loss =====
        # 计算TP/FP/FN
        tp = torch.sum(pred * target)  # 真阳性
        fp = torch.sum(pred * (1 - target))  # 假阳性
        fn = torch.sum((1 - pred) * target)  # 假阴性

        # Tversky系数 (数值稳定版)
        tversky = (tp + self.smooth) / (tp + self.alpha * fn + self.beta * fp + self.smooth)
        tversky_loss = 1 - tversky

        # ===== 3. 加权组合 =====
        batch_focal = torch.mean(focal_loss)  # 批平均
        combined_loss = (
                self.lambda_focal * batch_focal +
                self.lambda_tversky * tversky_loss
        )

        return combined_loss

class FocalOHEMLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=0.8, ohem_ratio=0.1, mode='focal'):
        """
        OHEM增强的损失函数(支持FocalBCE)
        Args:
            gamma: Focal Loss参数
            alpha: 正样本权重(Focal)
            ohem_ratio: 选取的困难样本比例(0 - 1)
            mode: 'focal' 或 'bce'
        
        Example:
            >>> # 版本1 Focal Loss + OHEM (适合极稀疏场景)
            >>> criterion_ohem_focal = OHEM_FocalLoss(
            ...     gamma=2.0,
            ...     alpha=0.9,
            ...     ohem_ratio=0.15,  # 仅用15%最困难像素
            ...     mode='focal'
            ... )
            >>> 
            >>> # 版本2 标准BCE + OHEM (计算更轻量)
            >>> criterion_ohem_bce = OHEM_FocalLoss(
            ...     ohem_ratio=0.1,
            ...     mode='bce'
            ... )
        """
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.ohem_ratio = ohem_ratio
        self.mode = mode
        self.base_loss = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, pred, target):
        # 计算逐像素基础损失
        pixel_loss = self.base_loss(pred, target)

        # Focal Loss加权
        if self.mode == 'focal':
            probas = torch.sigmoid(pred)
            p_t = torch.where(target == 1, probas, 1 - probas)
            focal_weight = (1 - p_t) ** self.gamma
            alpha_factor = torch.where(target == 1, self.alpha, 1 - self.alpha)
            pixel_loss = focal_weight * alpha_factor * pixel_loss

        # OHEM样本选择
        n_pixels = torch.numel(pixel_loss)
        n_select = int(n_pixels * self.ohem_ratio)

        if n_select > 0:
            flattened_loss = pixel_loss.view(-1)
            
            # 大图采样策略避免内存溢出
            if n_pixels > 1e6:
                # 随机采样50万像素或全部像素(如果不足50万)
                n_samples = min(500000, n_pixels)
                rand_idx = torch.randint(0, n_pixels, (n_samples,), device=flattened_loss.device)
                sampled_loss = flattened_loss[rand_idx]
                topk_val, _ = torch.topk(sampled_loss, n_select)
                threshold = topk_val[-1] if topk_val.numel() > 0 else 0
            else:
                topk_val, _ = torch.topk(flattened_loss, n_select)
                threshold = topk_val[-1] if topk_val.numel() > 0 else 0

            # 创建掩码(仅高损失区域)
            ohem_mask = (pixel_loss >= threshold).float()
            selected_loss = pixel_loss * ohem_mask
            final_loss = torch.sum(selected_loss) / (torch.sum(ohem_mask) + 1e-6)
        else:
            final_loss = torch.mean(pixel_loss)

        return final_loss

def get_criterion_info(criterion):
    """动态获取 criterion 的参数信息"""
    criterion_name = type(criterion).__name__
    info = f"Criterion: {criterion_name}\n"
    
    if criterion_name == "FocalTverskyLoss":
        info += f"    alpha (Tversky FN): {criterion.alpha}\n"
        info += f"    beta (Tversky FP): {criterion.beta}\n"
        info += f"    gamma (Focal): {criterion.gamma}\n"
        info += f"    focal_alpha: {criterion.focal_alpha}\n"
        info += f"    lambda_focal: {criterion.lambda_focal}\n"
        info += f"    lambda_tversky: {criterion.lambda_tversky}\n"
        info += f"    smooth: {criterion.smooth}\n"
    elif criterion_name == "FocalOHEMLoss":
        info += f"    gamma: {criterion.gamma}\n"
        info += f"    alpha: {criterion.alpha}\n"
        info += f"    ohem_ratio: {criterion.ohem_ratio}\n"
        info += f"    mode: {criterion.mode}\n"
    elif criterion_name == "SpatialFocalLoss":
        info += f"    gamma: {criterion.gamma}\n"
        info += f"    alpha: {criterion.alpha}\n"
        info += f"    include_pred_mask: {criterion.include_pred_mask}\n"
        info += f"    target_threshold: {criterion.target_threshold}\n"
        info += f"    pred_threshold: {criterion.pred_threshold}\n"
        info += f"    weight_min: {criterion.weight_min}\n"
        info += f"    weight_max: {criterion.weight_max}\n"
        info += f"    weight_gamma: {criterion.weight_gamma}\n"
        info += f"    background_weight: {criterion.background_weight}\n"
        info += f"    connectivity: {criterion.connectivity}\n"
        info += f"    eps: {criterion.eps}\n"    
    elif criterion_name == "MaskedL1Loss":
        info += f"    reduction: {criterion.reduction}\n"
        info += f"    eps: {criterion.eps}\n"
    
    else:
        info += "(custom or unknown parameters)"
    
    return info.strip()


class SpatialFocalLoss(nn.Module):
    """带空间连通域加权的 Focal Loss。

    该损失在标准 Focal Loss 的基础上，通过连通域面积为孤立火点提供额外权重。
    小面积连通域 (孤立像素/细小火点) 会获得更大的权重，从而提升模型对小目标的关注度。

    Args:
        gamma: Focal Loss 的聚焦参数。
        alpha: 正样本权重 (二分类场景)。若为 ``None`` 则不进行 alpha 加权。
        include_pred_mask: 是否将预测概率图一起用于连通域统计。
        target_threshold: 将标签二值化的阈值。
        pred_threshold: 当 ``include_pred_mask`` 为真时，对预测概率二值化的阈值。
        weight_min: 空间权重下限 (背景与大连通域的最低权重)。
        weight_max: 空间权重上限 (最小连通域的最大权重)。
        weight_gamma: 控制权重随面积衰减的幂指数，数值越大代表对小目标的加权越强。
        background_weight: 背景像素的权重。
        connectivity: 连通域邻域类型，``4`` 或 ``8``。
        eps: 数值稳定项。

    Note:
        当前实现仅支持二分类分割 (预测张量 ``pred`` 的通道数应为 1)。
    """

    def __init__(
        self,
        gamma: float = 2.0,
        alpha: float = 0.8,
        include_pred_mask: bool = False,
        target_threshold: float = 0.5,
        pred_threshold: float = 0.5,
        weight_min: float = 1.0,
        weight_max: float = 8.0,
        weight_gamma: float = 1.5,
        background_weight: float = 1.0,
        connectivity: int = 8,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        if connectivity not in (4, 8):
            raise ValueError("connectivity 仅支持 4 或 8")
        if weight_max < weight_min:
            raise ValueError("weight_max 应大于等于 weight_min")

        self.gamma = gamma
        self.alpha = alpha
        self.include_pred_mask = include_pred_mask
        self.target_threshold = target_threshold
        self.pred_threshold = pred_threshold
        self.weight_min = weight_min
        self.weight_max = weight_max
        self.weight_gamma = weight_gamma
        self.background_weight = background_weight
        self.connectivity = connectivity
        self.eps = eps

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        assert pred.size() == target.size(), "预测与标签尺寸不一致"
        if pred.dim() != 4 or pred.size(1) != 1:
            raise ValueError("SpatialFocalLoss 目前仅支持形状为 [B,1,H,W] 的二分类任务")

        target = target.float()
        probas = torch.sigmoid(pred)

        # Focal Loss 核心部分
        bce_loss = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
        p_t = torch.where(target == 1, probas, 1 - probas)
        focal_weight = (1 - p_t).clamp(min=0.0).pow(self.gamma)

        if self.alpha is not None:
            alpha_factor = torch.where(target == 1, self.alpha, 1 - self.alpha)
            focal_loss = alpha_factor * focal_weight * bce_loss
        else:
            focal_loss = focal_weight * bce_loss

        # 空间权重
        with torch.no_grad():
            weight_map = self._build_spatial_weight(pred=probas, target=target)

        weighted_loss = focal_loss * weight_map
        norm = weight_map.sum().clamp_min(self.eps)
        return weighted_loss.sum() / norm

    def _build_spatial_weight(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        target_mask = (target > self.target_threshold).squeeze(1)
        if self.include_pred_mask:
            pred_mask = (pred > self.pred_threshold).squeeze(1)
            union_mask = torch.logical_or(target_mask.bool(), pred_mask.bool())
        else:
            union_mask = target_mask.bool()

        batch_weights = []
        for mask in union_mask:
            weight_np = self._component_weight_2d(mask.cpu().numpy().astype(np.uint8))
            batch_weights.append(torch.from_numpy(weight_np))

        weight_tensor = torch.stack(batch_weights, dim=0)
        return weight_tensor.unsqueeze(1).to(target.device)

    def _component_weight_2d(self, binary_mask: np.ndarray) -> np.ndarray:
        if binary_mask.ndim != 2:
            raise ValueError("内部错误: binary_mask 应为二维数组")

        if binary_mask.sum() == 0:
            return np.full_like(binary_mask, fill_value=self.background_weight, dtype=np.float32)

        labels = np.zeros_like(binary_mask, dtype=np.int32)
        current_label = 0
        neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        if self.connectivity == 8:
            neighbors += [(-1, -1), (-1, 1), (1, -1), (1, 1)]

        h, w = binary_mask.shape
        for y in range(h):
            for x in range(w):
                if binary_mask[y, x] and labels[y, x] == 0:
                    current_label += 1
                    self._flood_fill(binary_mask, labels, y, x, current_label, neighbors)

        # 统计面积并转换为权重
        counts = np.bincount(labels.flatten())
        counts = counts.astype(np.float32)
        counts[0] = np.inf  # 背景标签不参与权重计算
        area_map = counts[labels]
        inv_area = 1.0 / np.maximum(area_map, 1.0)
        inv_area[~np.isfinite(inv_area)] = 0.0

        if np.any(inv_area > 0):
            inv_area /= inv_area.max()

        weights = self.weight_min + (self.weight_max - self.weight_min) * (inv_area ** self.weight_gamma)
        weights[labels == 0] = self.background_weight
        return weights.astype(np.float32)

    def _flood_fill(
        self,
        binary_mask: np.ndarray,
        labels: np.ndarray,
        start_y: int,
        start_x: int,
        label_id: int,
        neighbors,
    ) -> None:
        queue = deque([(start_y, start_x)])
        labels[start_y, start_x] = label_id
        h, w = binary_mask.shape

        while queue:
            y, x = queue.popleft()
            for dy, dx in neighbors:
                ny, nx = y + dy, x + dx
                if 0 <= ny < h and 0 <= nx < w and binary_mask[ny, nx] and labels[ny, nx] == 0:
                    labels[ny, nx] = label_id
                    queue.append((ny, nx))


class MaskedL1Loss(nn.Module):
    """针对遥感火点分割任务的掩膜 L1 损失。

    仅对背景区域 (``fire_mask == 0``) 计算 L1，避免火点区域干扰重建分支。

    Args:
        reduction: ``"mean"`` 或 ``"sum"``，控制损失归约方式。
        eps: 防止除零的平滑项。
    """

    def __init__(self, reduction: str = "mean", eps: float = 1e-6) -> None:
        super().__init__()
        if reduction not in {"mean", "sum"}:
            raise ValueError("reduction 仅支持 'mean' 或 'sum'")
        self.reduction = reduction
        self.eps = eps

    def forward(
    self,
    reconstruction: torch.Tensor,
    inputs: torch.Tensor,
    fire_mask: torch.Tensor,
    reduction: Optional[str] = None,
    ) -> torch.Tensor:
        if reconstruction.shape != inputs.shape:
            raise ValueError("reconstruction 与 inputs 的形状必须一致")

        if fire_mask.dim() == 3:
            fire_mask = fire_mask.unsqueeze(1)
        if fire_mask.shape[1] not in {1, reconstruction.shape[1]}:
            raise ValueError("fire_mask 的通道数需为 1 或与重建张量一致")

        background_mask = (1.0 - fire_mask).float()
        if background_mask.shape[1] != reconstruction.shape[1]:
            background_mask = background_mask.expand(-1, reconstruction.shape[1], -1, -1)

        diff = torch.abs(reconstruction - inputs) * background_mask
        reduce_mode = self.reduction if reduction is None else reduction

        if reduce_mode == "sum":
            return diff.sum()

        denom = background_mask.sum().clamp_min(self.eps)
        return diff.sum() / denom