from typing import *

import torch
import torch.nn as nn
import torch.nn.functional as F


def distribution_focal_loss(logits, labels, weight):
    dis_left = labels.long()
    dis_right = dis_left + 1
    weight_left = dis_right.float() - labels
    weight_right = labels - dis_left.float()
    loss = F.cross_entropy(logits, dis_left, reduction='none') * weight_left + \
        F.cross_entropy(logits, dis_right, reduction='none') * weight_right
    loss = loss * weight if weight is not None else loss
    return loss

class DistributionFocalLoss(nn.Module):

    def __init__(self, reduction: str='none', weight=Optional[torch.Tensor]=None):
        super(DistributionFocalLoss, self).__init__()
        assert reduction in ('sum', 'mean', 'none')
        self.reduction = reduction
        self.weight = weight

    def forward(self, logits: torch.Tensor, labels: torch.Tensor):
        loss = distribution_focal_loss(logits, labels, self.weight)
        
        if self.reduction == 'sum':
            return loss.sum()
        elif self.reduction == 'mean':
            return loss.mean()
        else:
            return loss