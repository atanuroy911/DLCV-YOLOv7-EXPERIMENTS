import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    """
    Args:
        smooth: a value to avoid dividing by zero
        reduction: the output type that can be selected from none, sum and mean

    Forward:
        logits: a prediction with torch tensor type and has shape of (B, N, W, H)
        labels: a label with torch tensor type and has shape of (B, W, H)
            where B is batch size, N is number of classes, W and H is the width and height of outputs and labels

    Examples:
        >>> criteria = DiceLoss()
        >>> outputs = model(images)
        >>> loss = criteria(outputs, labels)
    """
    def __init__(self, smooth: float=1e-9, reduction: str='mean'):
        super(DiceLoss, self).__init__()
        assert reduction in ('none', 'sum', 'mean'), \
            f'reduction {reduction} does not exists'

        self.smooth = smooth
        self.reduction = reduction
    
    def forward(logits: torch.Tensor, labels: torch.Tensor):

        num_classes = logits.size(1)

        logits = F.softmax(logits, dim=1)
        one_hot = torch.eye(num_classes)[labels.cpu().squeeze(dim=1)]
        one_hot = one_hot.permute(0,3,1,2).float()
        one_hot = one_hot.type(logits.type())
        dims = (0,) + tuple(range(2, labels.ndimension()))

        intersection = torch.sum(logits * one_hot, dims)
        summatation = torch.sum(logits + one_hot, dims)

        dice = (2. * intersection + self.smooth) / (summatation + self.smooth)
        loss = 1. - dice

        if self.reduction == 'none':
            return loss
        elif self.reduction == 'mean':
            return loss.mean()
        else:
            return loss.sum()