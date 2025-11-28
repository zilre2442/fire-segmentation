from collections import deque
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalTverskyLoss(nn.Module):
    """标准 Focal + Tversky 组合损失（修正 BCE 用法）。

    原版本错误地对已经 sigmoid 的概率使用 binary_cross_entropy_with_logits。
    这里保持接口不变：输入 pred 为 logits，内部再 sigmoid 得概率。
    """
    def __init__(self, alpha=0.64, beta=0.6, gamma=1.6, focal_alpha=0.85,
                 lambda_focal=0.4, lambda_tversky=0.6, smooth=1e-6):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.focal_alpha = focal_alpha
        self.lambda_focal = lambda_focal
        self.lambda_tversky = lambda_tversky
        self.smooth = smooth

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        assert pred.size() == target.size(), "预测与标签尺寸不一致"
        target = target.float()
        probas = torch.sigmoid(pred)

        # Focal BCE (逐像素)
        bce_loss = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
        p_t = torch.where(target == 1, probas, 1 - probas)
        focal_weight = (1 - p_t).pow(self.gamma)
        alpha_factor = torch.where(target == 1, self.focal_alpha, 1 - self.focal_alpha)
        focal_loss_pixel = alpha_factor * focal_weight * bce_loss
        focal_loss = focal_loss_pixel.mean()

        # Tversky
        tp = (probas * target).sum()
        fp = (probas * (1 - target)).sum()
        fn = ((1 - probas) * target).sum()
        tversky = (tp + self.smooth) / (tp + self.alpha * fn + self.beta * fp + self.smooth)
        tversky_loss = 1 - tversky
        return self.lambda_focal * focal_loss + self.lambda_tversky * tversky_loss

class TverskyLoss(nn.Module):
    """标准 Tversky 损失 (仅 logits 输入)。"""

    def __init__(self, alpha: float = 0.64, beta: float = 0.6, smooth: float = 1e-6) -> None:
        super().__init__()
        if not 0 <= alpha <= 1:
            raise ValueError("alpha 范围应在 [0,1]")
        if not 0 <= beta <= 1:
            raise ValueError("beta 范围应在 [0,1]")
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        target = target.float()
        probas = torch.sigmoid(pred)
        tp = (probas * target).sum()
        fp = (probas * (1 - target)).sum()
        fn = ((1 - probas) * target).sum()
        tversky = (tp + self.smooth) / (tp + self.alpha * fn + self.beta * fp + self.smooth)
        return 1 - tversky

class FocalLoss(nn.Module):
    """标准二值 Focal Loss（基于 BCEWithLogits）。

    Args:
        gamma: 聚焦参数，控制对难样本的加权强度。
        alpha: 正样本权重（None 表示不使用 alpha 加权）。
        reduction: 归约方式，"mean" 或 "sum"。
    """
    def __init__(self, gamma: float = 1.6, alpha: Optional[float] = 0.85, reduction: str = "mean") -> None:
        super().__init__()
        if reduction not in {"mean", "sum"}:
            raise ValueError("reduction 仅支持 'mean' 或 'sum'")
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        target = target.float()
        bce = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
        prob = torch.sigmoid(pred)
        p_t = torch.where(target == 1, prob, 1 - prob)
        focal_weight = (1 - p_t).pow(self.gamma)
        if self.alpha is not None:
            alpha_factor = torch.where(target == 1, self.alpha, 1 - self.alpha)
            loss = alpha_factor * focal_weight * bce
        else:
            loss = focal_weight * bce
        if self.reduction == "sum":
            return loss.sum()
        return loss.mean()


class DiceLoss(nn.Module):
    """Dice 损失 (1 - Dice 系数)。"""

    def __init__(self, smooth: float = 1e-6) -> None:
        super().__init__()
        self.smooth = smooth

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        target = target.float()
        probas = torch.sigmoid(pred)
        intersection = (probas * target).sum()
        union = probas.sum() + target.sum()
        dice = (2 * intersection + self.smooth) / (union + self.smooth)
        return 1 - dice

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
    elif criterion_name == "TverskyLoss":
        info += f"    alpha: {criterion.alpha}\n"
        info += f"    beta: {criterion.beta}\n"
        info += f"    smooth: {criterion.smooth}\n"
    elif criterion_name == "FocalOHEMLoss":
        info += f"    gamma: {criterion.gamma}\n"
        info += f"    alpha: {criterion.alpha}\n"
        info += f"    ohem_ratio: {criterion.ohem_ratio}\n"
        info += f"    mode: {criterion.mode}\n"
    elif criterion_name == "FocalLoss":
        info += f"    gamma: {criterion.gamma}\n"
        info += f"    alpha: {criterion.alpha}\n"
        info += f"    reduction: {criterion.reduction}\n"
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
        info += f"    weight_strategy: {criterion.weight_strategy}\n"
        info += f"    weight_update_freq: {criterion.weight_update_freq}\n"
        info += f"    eps: {criterion.eps}\n"    
    elif criterion_name == "DiceLoss":
        info += f"    smooth: {criterion.smooth}\n"
    elif criterion_name == "MaskedL1Loss":
        info += f"    reduction: {criterion.reduction}\n"
        info += f"    eps: {criterion.eps}\n"
    elif criterion_name == "SpatialFocalTverskyLoss":
        info += f"    alpha_tversky: {criterion.alpha_tversky}\n"
        info += f"    beta_tversky: {criterion.beta_tversky}\n"
        info += f"    gamma_focal: {criterion.gamma_focal}\n"
        info += f"    focal_alpha: {criterion.focal_alpha}\n"
        info += f"    lambda_focal: {criterion.lambda_focal}\n"
        info += f"    lambda_tversky: {criterion.lambda_tversky}\n"
        info += f"    weight_min: {criterion.weight_min}\n"
        info += f"    weight_max: {criterion.weight_max}\n"
        info += f"    area_gamma: {criterion.area_gamma}\n"
        info += f"    background_weight: {criterion.background_weight}\n"
        info += f"    connectivity: {criterion.connectivity}\n"
        info += f"    smooth: {criterion.smooth}\n"
        info += f"    eps: {criterion.eps}\n"
    
    else:
        info += "(custom or unknown parameters)"
    
    return info.strip()

class SpatialFocalLoss(nn.Module):
    """带空间连通域加权的 Focal Loss。

    该损失在标准 Focal Loss 的基础上,通过连通域面积为孤立火点提供额外权重。
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
        weight_strategy: 连通域加权策略，可选 ``"small"`` (小目标高权重)、
                        ``"large"`` (大目标高权重)、``"uniform"`` (不加权)。
        eps: 数值稳定项。

    Note:
        当前实现仅支持二分类分割 (预测张量 ``pred`` 的通道数应为 1)。
    """

    def __init__(
        self,
        gamma: float = 1.5,
        alpha: float = 0.8,
        include_pred_mask: bool = False,
        target_threshold: float = 0.5,
        pred_threshold: float = 0.5,
        weight_min: float = 1.0,
        weight_max: float = 4.0,
        weight_gamma: float = 1.5,
        background_weight: float = 1.0,
        connectivity: int = 8,
        eps: float = 1e-6,
        weight_update_freq: int = 1,
        weight_strategy: str = "small",
    ) -> None:
        super().__init__()
        if connectivity not in (4, 8):
            raise ValueError("connectivity 仅支持 4 或 8")
        if weight_max < weight_min:
            raise ValueError("weight_max 应大于等于 weight_min")
        if weight_strategy not in {"small", "large", "uniform"}:
            raise ValueError("weight_strategy 必须是 'small', 'large' 或 'uniform'")

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
        self.weight_update_freq = weight_update_freq
        self.weight_strategy = weight_strategy

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

        # 空间权重（每个 batch 重新计算）
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

        # uniform 策略：快速路径，直接返回常数权重
        if self.weight_strategy == "uniform":
            weights = torch.where(
                union_mask,
                torch.ones_like(union_mask, dtype=torch.float32),
                torch.full_like(union_mask, self.background_weight, dtype=torch.float32),
            )
            return weights.unsqueeze(1)

        # 原尺寸计算连通域权重
        batch_weights = self._component_weight_2d_gpu(union_mask)
        return batch_weights.unsqueeze(1)

    def _component_weight_2d_gpu(self, binary_masks: torch.Tensor) -> torch.Tensor:
        """使用 OpenCV 高效计算连通域权重（CPU优化后传回GPU）
        
        Args:
            binary_masks: 形状为 [B, H, W] 的二值掩膜张量
            
        Returns:
            权重图，形状为 [B, H, W]
        """
        import cv2
        
        device = binary_masks.device
        B, H, W = binary_masks.shape
        
        weights_batch = []
        
        for b in range(B):
            # 将mask转到CPU并转为numpy（uint8格式）
            mask_np = binary_masks[b].cpu().numpy().astype(np.uint8)
            
            if mask_np.sum() == 0:
                # 全背景情况
                weight_map = np.full_like(mask_np, self.background_weight, dtype=np.float32)
                weights_batch.append(torch.from_numpy(weight_map))
                continue
            
            # 使用OpenCV的连通域标记（高度优化）
            connectivity_cv = 4 if self.connectivity == 4 else 8
            num_labels, labels = cv2.connectedComponents(mask_np, connectivity=connectivity_cv)
            
            if num_labels <= 1:  # 只有背景
                weight_map = np.full_like(mask_np, self.background_weight, dtype=np.float32)
                weights_batch.append(torch.from_numpy(weight_map))
                continue
            
            # 统计每个连通域的面积
            areas = np.bincount(labels.flatten()).astype(np.float32)
            areas[0] = np.inf  # 背景不参与计算

            # 构建面积映射（前景像素处为该连通域面积）
            area_map = areas[labels]

            # 根据策略计算权重
            if self.weight_strategy == "small":
                # 小目标更高权重：使用归一化反面积
                inv_area = 1.0 / np.maximum(area_map, 1.0)
                inv_area[~np.isfinite(inv_area)] = 0.0
                if np.any(inv_area > 0):
                    inv_area = inv_area / inv_area.max()
                weights = self.weight_min + (self.weight_max - self.weight_min) * (inv_area ** self.weight_gamma)
            elif self.weight_strategy == "large":
                # 大连通域更高权重：使用归一化面积
                area_norm = area_map.copy()
                fg = (labels > 0)
                max_area = area_norm[fg].max() if np.any(fg) else 1.0
                if max_area <= 0:
                    max_area = 1.0
                area_norm = area_norm / max_area
                weights = self.weight_min + (self.weight_max - self.weight_min) * (np.power(area_norm, self.weight_gamma))
            else:  # uniform
                # 前景统一权重 1.0（不做连通域加权）
                weights = np.ones_like(area_map, dtype=np.float32)

            # 背景权重单独指定
            weights[labels == 0] = self.background_weight
            
            weights_batch.append(torch.from_numpy(weights.astype(np.float32)))
        
        # 将结果堆叠并传回GPU
        return torch.stack(weights_batch, dim=0).to(device)

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

class SpatialFocalTverskyLoss(nn.Module):
    """融合 small 连通域加权策略的 Focal+Tversky 损失（仅用于单通道二分类）。

    权重图专注于小面积连通域：权重 = w_min + (w_max - w_min) * (inv_area_norm ** area_gamma)
    背景像素统一给 background_weight。

    公式：Loss = λ_focal * (Σ_i focal_i * w_i / Σ_i w_i) + λ_tversky * (1 - Tversky)

    Args:
        alpha_tversky: Tversky FN 权重（漏报惩罚）
        beta_tversky: Tversky FP 权重（误报惩罚）
        gamma_focal: Focal 聚焦参数
        focal_alpha: 正样本 alpha 权重
        lambda_focal: Focal 部分权重
        lambda_tversky: Tversky 部分权重
        weight_min, weight_max: 空间权重范围
        area_gamma: 控制小面积增强强度（越大越强调小目标）
        background_weight: 背景像素权重
        connectivity: 连通域 4 或 8 邻域
        smooth: 数值稳定项
        eps: 加权归一化防除零
    """
    def __init__(
        self,
        alpha_tversky: float = 0.6,
        beta_tversky: float = 0.4,
        gamma_focal: float = 1.6,
        focal_alpha: float = 0.85,
        lambda_focal: float = 0.4,
        lambda_tversky: float = 0.6,
        weight_min: float = 1.0,
        weight_max: float = 4.0,
        area_gamma: float = 1.5,
        background_weight: float = 1.0,
        connectivity: int = 8,
        smooth: float = 1e-6,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        if connectivity not in (4, 8):
            raise ValueError("connectivity 仅支持 4 或 8")
        if weight_max < weight_min:
            raise ValueError("weight_max 必须 >= weight_min")
        self.alpha_tversky = alpha_tversky
        self.beta_tversky = beta_tversky
        self.gamma_focal = gamma_focal
        self.focal_alpha = focal_alpha
        self.lambda_focal = lambda_focal
        self.lambda_tversky = lambda_tversky
        self.weight_min = weight_min
        self.weight_max = weight_max
        self.area_gamma = area_gamma
        self.background_weight = background_weight
        self.connectivity = connectivity
        self.smooth = smooth
        self.eps = eps

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if pred.shape != target.shape:
            raise ValueError("pred 与 target 形状需一致")
        if pred.dim() != 4 or pred.size(1) != 1:
            raise ValueError("仅支持 [B,1,H,W] 二分类掩膜")
        target = target.float()
        probas = torch.sigmoid(pred)

        # Focal (逐像素)
        bce = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
        p_t = torch.where(target == 1, probas, 1 - probas)
        focal_mod = (1 - p_t).pow(self.gamma_focal)
        alpha_factor = torch.where(target == 1, self.focal_alpha, 1 - self.focal_alpha)
        focal_pix = alpha_factor * focal_mod * bce  # [B,1,H,W]

        # 空间权重
        with torch.no_grad():
            weights = self._compute_small_weights(target)  # [B,1,H,W]

        focal_weighted = (focal_pix * weights).sum() / (weights.sum().clamp_min(self.eps))

        # Tversky（未加权，保持全局统计）
        tp = (probas * target).sum()
        fp = (probas * (1 - target)).sum()
        fn = ((1 - probas) * target).sum()
        tversky = (tp + self.smooth) / (tp + self.alpha_tversky * fn + self.beta_tversky * fp + self.smooth)
        tversky_loss = 1 - tversky
        return self.lambda_focal * focal_weighted + self.lambda_tversky * tversky_loss

    def _compute_small_weights(self, target: torch.Tensor) -> torch.Tensor:
        # target 已是 [B,1,H,W]
        mask = (target > 0.5).squeeze(1)  # [B,H,W]
        B, H, W = mask.shape
        device = mask.device
        weights_list = []
        import cv2
        for b in range(B):
            m = mask[b].cpu().numpy().astype(np.uint8)
            if m.sum() == 0:
                weights_list.append(torch.full((H, W), self.background_weight, dtype=torch.float32))
                continue
            num_labels, labels = cv2.connectedComponents(m, connectivity=self.connectivity)
            if num_labels <= 1:
                weights_list.append(torch.full((H, W), self.background_weight, dtype=torch.float32))
                continue
            areas = np.bincount(labels.flatten()).astype(np.float32)
            areas[0] = np.inf  # 背景
            area_map = areas[labels]
            inv_area = 1.0 / np.maximum(area_map, 1.0)
            inv_area[~np.isfinite(inv_area)] = 0.0
            if (inv_area > 0).any():
                inv_area /= inv_area.max()
            weights = self.weight_min + (self.weight_max - self.weight_min) * (inv_area ** self.area_gamma)
            weights[labels == 0] = self.background_weight
            weights_list.append(torch.from_numpy(weights.astype(np.float32)))
        weights_tensor = torch.stack(weights_list, dim=0).unsqueeze(1).to(device)
        return weights_tensor