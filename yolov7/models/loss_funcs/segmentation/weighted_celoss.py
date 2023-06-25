import torch
import torch.nn as nn
import torch.nn.functional as F


class WeightedCELoss(nn.Module):
    """
    Args:
        weights: the list consist of the weighted values for each classes
        reduction: the output type that can be selected from none, sum and mean
    Forward:
        logits: a prediction with torch tensor type and has shape of (B, N, W, H)
        labels: a label with torch tensor type and has shape of (B, W, H)
            where B is batch size, N is number of classes, W and H is the width and height of outputs and labels
    Examples:
        >>> criteria = WeightedCELoss()
        >>> outputs = model(images)
        >>> loss = criteria(outputs, labels)
    """
    def __init__(self, weights: list, reduction: str='mean'):
        super(WeightedCELoss, self).__init__()
        assert reduction in ('none', 'sum', 'mean'), \
            f'reduction {reduction} does not exists'
        self.weights = torch.tensor(weights, dtype=torch.float32)
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, labels: torch.Tensor):
        loss = F.cross_entropy(logits, labels, weight=self.weights, reduction=self.reduction)
        return loss