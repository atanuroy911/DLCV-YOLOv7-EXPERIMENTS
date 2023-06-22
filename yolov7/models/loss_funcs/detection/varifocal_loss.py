from typing import *

import torch
import torch.nn as nn
import torch.nn.functional as F


def varifocal_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    weight: Optional[torch.Tensor]=None,
    alpha: float=0.75,
    gamma: float=2.0,
    iou_weighted: bool=True,
    
):
    assert logits.size == labels.size()
    logits_prob = logits.sigmoid()
    labels = labels.type_as(logits)
    if iou_weighted:
        focal_weight = labels * (labels > 0.0).float() + \
            alpha * (logits_prob - labels).abs().pow(gamma) * \
            (labels <= 0.0).float()

    else:
        focal_weight = (labels > 0.0).float() + \
            alpha * (logits_prob - labels).abs().pow(gamma) * \
            (labels <= 0.0).float()

    loss = F.binary_cross_entropy_with_logits(
        logits, labels, reduction='none') * focal_weight
    loss = loss * weight if weight is not None else loss
    return loss


class VariFocalLoss(nn.Module):
    """
    Args:
        alpha: a hyperparameter to weight easy and hard samples
        gamma: a hyperparameter for focusing the easy examples (modulating factor)
        iou_weighted: whether to weight the loss of the positive samples with the iou target
        reduction: the output type that can be selected from none, sum and mean
    Forward:
        logits: a prediction with torch tensor type and has shape of (N, ...)
        labels: a label with torch tensor type and has shape of (M, ...)
    Examples:
        >>> loss_func = VariFocalLoss()
        >>> outputs = model(images)
        >>> loss = loss_func(outputs, labels)
        >>> loss.backward()
    """
    def __init__(
        self, 
        alpha: float=0.75, 
        gamma: float=2.0, 
        iou_weighted: bool=True, 
        reduction: str='mean',
    ):
        super(VariFocalLoss, self).__init__()
        assert reduction in ('mean', 'sum', 'none')
        assert alpha >= 0.0
        self.alpha = alpha
        self.gamma = gamma
        self.iou_weighted = iou_weighted
        self.reduction = reduction

    def forward(self, logits, labels):
        loss = varifocal_loss(logits, labels, self.alpha, self.gamma, self.iou_weighted)

        if self.reduction == 'sum':
            return loss.sum()
        elif self.reduction == 'mean':
            return loss.mean()
        else:
            return loss