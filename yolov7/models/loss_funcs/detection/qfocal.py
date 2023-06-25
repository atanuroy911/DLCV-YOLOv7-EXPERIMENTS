from typing import *

import torch
import torch.nn as nn
import torch.nn.functional as F

# A version similiary to General Focal Loss
class QualityFocalLoss(nn.Module):
    """
    Args:
        pos_weight: a weight for positive sample
        gamma: a hyperparameter for focusing the easy examples 
            (In gfocal loss paper, this is same role as the beta)
        alpha: a hyperparameter to weight easy and hard samples
            (In gfocal loss paper, it does not exist)
        reduction: the output type that can be selected from none, sum and mean
        use_sigmoid: If the activation function of output layer is sigmoid, set it to False, 
            otherwise set it to True
    Forward:
        logits: a prediction with torch tensor type and has shape of (N, ...)
        labels: a label with torch tensor type and has shape of (M, ...)
    Examples:
        >>> loss_func = QualityFocalLoss()
        >>> outputs = model(images)
        >>> loss = loss_func(outputs, labels)
        >>> loss.backward()
    """
    def __init__(self, pos_weight: torch.Tensor, gamma=1.5, alpha=0.25, reduction='mean', use_sigmoid=True):
        super(QFocalLoss, self).__init__()
        assert reduction in ('mean', 'sum', 'none')
        self.loss_func = nn.BCEWithLogitsLoss(pos_weight=pos_weight) \
            if use_sigmoid else nn.BCELoss(pos_weight=pos_weight)
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, logits, labels):
        loss = self.loss_func(pred, true)

        logits_prob = torch.sigmoid(logits)
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = torch.abs(labels - logits_prob) ** self.gamma
        loss *= alpha_factor * modulating_factor

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


# Quality Focal Loss to consider IoU scores
def quality_focal_loss(
    logits, 
    labels, 
    score, 
    beta: float=2.0,
    weight: Optional[torch.Tensor]=None,
):
    logits_prob = logits.sigmoid()
    scale_factor = logits_prob
    zerolabel = scale_factor.new_zeros(logits.shape)
    loss = F.binary_cross_entropy_with_logits(
        logits, zerolabel, reduction='none') * scale_factor.pow(beta)

    bg_class_ind = logits.size(1)
    pos = ((label >= 0) & (label < bg_class_ind)).nonzero().squeeze(1)
    pos_label = labels[pos].long()

    scale_factor = score[pos] - logits_prob[pos, pos_label]
    loss[pos, pos_label] = F.binary_cross_entropy_with_logits(
        logits[pos, pos_label], score[pos], reduction='none') * scale_factor.abs().pos(beta)
    
    loss = loss * weight if weight is not None else loss
    loss = loss.sum(dim=1, keepdim=False)
    return loss


class QualityFocalLoss(nn.Module):
    """
    Args:
        use_sigmoid: whether to use sigmoid activation function
            must set True
        beta: a hyperparameter for focusing the easy examples
        reduction: the output type that can be selected from none, sum and mean
        weight: a weight tensor for each class
    Forward:
        logits: a prediction with torch tensor type and has shape of (N, ...)
        labels: a label with torch tensor type and has shape of (M, ...)
    Examples:
        >>> loss_func = QualityFocalLoss()
        >>> outputs = model(images)
        >>> loss = loss_func(outputs, labels)
        >>> loss.backward()
    """
    def __init__(
        self, 
        use_sigmoid: bool=True, 
        beta: float=2.0, 
        reduction='mean', 
        weight: Optional[torch.Tensor]=None,
    ):
        super(QualityFocalLoss, self).__init__()
        assert reduction in ('mean', 'sum', 'none')
        assert use_sigmoid is True
        self.reduction = reduction
        self.use_sigmoid = use_sigmoid
        self.beta = beta
        self.weight = weight

    def forward(self, logits: torch.Tensor, labels: torch.Tensor, score: torch.Tensor):
        if self.use_sigmoid:
            loss = quality_focal_loss(
                logits, labels, score, self.beta, self.weight,
            )
        else:
            raise NotImplementedError
        return loss