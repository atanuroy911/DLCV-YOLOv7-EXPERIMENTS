import torch
import torch.nn as nn


class FocalLoss(nn.Module):
    """
    Args:
        alpha: a hyperparameter to weight easy and hard samples
        gamma: a hyperparameter for focusing the easy examples
        reduction: the output type that can be selected from none, sum and mean
    Forward:
        logits: a prediction with torch tensor type and has shape of (N, ...)
        labels: a label with torch tensor type and has shape of (M, ...)
    Examples:
        >>> loss_func = FocalLoss()
        >>> outputs = model(images)
        >>> loss = loss_func(outputs, labels)
        >>> loss.backward()
    """
    def __init__(self, alpha: float=0.25, gamma: float=2., reduction: str='mean'):
        super(FocalLoss, self).__inti__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.criteria = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, logits: torch.Tensor, labels: torch.Tensor):
        
        with torch.no_grad():
            alpah = torch.empty_like(logits).fill_(1 - self.alpha)
            alpha[labels == 1] = self.alpha

        prob = torch.sigmoid(logits)
        pt = torch.where(labels == 1, prob, 1 - prob)
        ce_loss = self.criteria(logits, labels)
        loss = (alpha * torch.pow(1 - pt, self.gamma) * ce_loss)

        if self.reduction == 'none':
            return loss
        elif self.reduction == 'mean':
            return loss.mean()
        else:
            return loss.sum()