import torch
import torch.nn as nn

from iou import iou_func


class GIoULoss(nn.Module):
    """
    Args:
        smooth: a value to avoid dividing by zero
        reduction: the output type that can be selected from none, sum and mean
    Forward:
        logits: a prediction with torch tensor type and has shape of (N, 4)
        labels: a label with torch tensor type and has shape of (M, 4)
    Examples:
        >>> giou_loss = GIoULoss()
        >>> outputs = model(images)
        >>> loss = giou_loss(outputs, labels)
        >>> loss.backward()
    """
    def __init__(self, smooth: float=1e-9, reduction: str='mean'):
        super(GIoULoss, self).__init__()
        self.smooth = smooth
        self.reduction = reduction
    
    def forward(self, logits: torch.Tensor, labels: torch.Tensor):
        giou = giou_func(logits, labels, self.smooth)
        loss = 1 - giou
        
        if self.reduction == 'none':
            return loss
        elif self.reduction == 'mean':
            return loss.mean()
        else:
            return loss.sum()


def giou_func(logits, labels, smooth):
    iou = iou_func(logits, labels, smooth)
    
    lt = torch.max(labels[:, :2], logits[:, :2])
    rb = torch.min(labels[:, 2:], logits[:, 2:])
    wh = (rb - lt + smooth).clamp(min=0)
    
    enclosure = wh[:, 0] * wh[:, 1]

    giou = iou - (enclosure - union) / enclosure
    return giou