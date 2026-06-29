import torch
import torch.nn as nn
from torch import Tensor
from mmdet.registry import MODELS
from typing import Optional
from .utils import weighted_loss

@weighted_loss
def smooth_l1_loss(pred: Tensor, target: Tensor, beta: float = 1.0) -> Tensor:
    """Smooth L1 loss.

    Args:
        pred (Tensor): The prediction.
        target (Tensor): The learning target of the prediction.
        beta (float, optional): The threshold in the piecewise function.
            Defaults to 1.0.

    Returns:
        Tensor: Calculated loss
    """
    assert beta > 0
    if target.numel() == 0:
        return pred.sum() * 0

    assert pred.size() == target.size()
    diff = torch.abs(pred - target)
    loss = torch.where(diff < beta, 0.5 * diff * diff / beta, diff - 0.5 * beta)
    return loss

@MODELS.register_module()
class CustomSmoothL1Loss(nn.Module):
    """Custom Smooth L1 loss for teeth detection.

    Args:
        beta (float, optional): The threshold in the piecewise function.
            Defaults to 1.0.
        reduction (str, optional): The method to reduce the loss.
            Options are "none", "mean" and "sum". Defaults to "mean".
        loss_weight (float, optional): The weight of loss.
        crown_weight (float, optional): Weight for crown part.
        root_weight (float, optional): Weight for root part.
    """

    def __init__(self,
                 beta: float = 1.0,
                 reduction: str = 'mean',
                 loss_weight: float = 1.0,
                 crown_weight: float = 1.0,
                 root_weight: float = 1.5) -> None:
        super().__init__()
        self.beta = beta
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.crown_weight = crown_weight
        self.root_weight = root_weight

    def forward(self,
                pred: Tensor,
                target: Tensor,
                weight: Optional[Tensor] = None,
                avg_factor: Optional[int] = None,
                reduction_override: Optional[str] = None,
                tooth_labels: Optional[Tensor] = None,
                **kwargs) -> Tensor:
        """Forward function.

        Args:
            pred (Tensor): The prediction.
            target (Tensor): The learning target of the prediction.
            weight (Tensor, optional): The weight of loss for each
                prediction. Defaults to None.
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
            reduction_override (str, optional): The reduction method used to
                override the original reduction method of the loss.
                Defaults to None.
            tooth_labels (Tensor, optional): Labels for each tooth to determine
                crown or root. Defaults to None.

        Returns:
            Tensor: Calculated loss
        """
        if weight is not None and not torch.any(weight > 0):
            if pred.dim() == weight.dim() + 1:
                weight = weight.unsqueeze(1)
            return (pred * weight).sum()
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = reduction_override if reduction_override else self.reduction

        # 根据牙齿标签和边界框中间位置计算牙根和牙冠的权重
        if tooth_labels is not None:
            # 计算边界框的中间位置
            bbox_centers = (pred[:, :2] + pred[:, 2:]) / 2.0

            # 根据FDI编号和边界框中间位置判断是牙冠还是牙根
            for i, label in enumerate(tooth_labels):
                if 11 <= label <= 18 or 21 <= label <= 28:
                    # 上颌牙齿，牙冠部分
                    if bbox_centers[i, 1] < 0.5:  # 假设上半部分为牙冠
                        weight[i] = self.crown_weight
                    else:
                        weight[i] = self.root_weight
                elif 31 <= label <= 38 or 41 <= label <= 48:
                    # 下颌牙齿，牙根部分
                    if bbox_centers[i, 1] > 0.5:  # 假设下半部分为牙根
                        weight[i] = self.root_weight
                    else:
                        weight[i] = self.crown_weight

        loss_bbox = self.loss_weight * smooth_l1_loss(
            pred,
            target,
            weight,
            beta=self.beta,
            reduction=reduction,
            avg_factor=avg_factor,
            **kwargs)
        return loss_bbox