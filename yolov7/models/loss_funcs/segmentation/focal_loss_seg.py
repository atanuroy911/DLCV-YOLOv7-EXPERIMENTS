from typing import *

import torch
import torch.nn as nn


class FocalLoss(nn.Module):
    """
    Args:
        alpha: a hyperparameter to weight easy and hard samples
        gamma: a hyperparameter for focusing the easy examples
        weight: the list consist of the weighted values for each classes
        reduction: the output type that can be selected from none, sum and mean
    Forward:
        logits: a prediction with torch tensor type and has shape of (B, N, W, H)
        labels: a label with torch tensor type and has shape of (B, W, H)
            where B is batch size, N is number of classes, W and H is the width and height of outputs and labels
    Examples:
        >>> criteria = FocalLoss()
        >>> outputs = model(images)
        >>> loss = criteria(outputs, labels)
    """
    def __init__(self, alpha: float=0.25, gamma: float=2., weight: Optional[torch.Tensor]=None, reduction: str='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.criteria = nn.CrossEntropyLoss(weight=weight, reduction='none')

    def forward(self, logits: torch.Tensor, labels: torch.Tensor):
        ce_loss = self.criteria(logits, labels)
        pt = torch.exp(-ce_loss)
        loss = self.alpha * (1 - pt) ** self.gamma

        if self.reduction == 'none':
            return loss
        elif self.reduction == 'mean':
            return loss.mean()
        else:
            return loss.sum()