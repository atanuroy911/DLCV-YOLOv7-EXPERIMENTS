import torch
import torch.nn as nn


class IoULoss(nn.Module):
    """
    Args:
        smooth: a value to avoid dividing by zero
        reduction: the output type that can be selected from none, sum and mean
    Forward:
        logits: a prediction with torch tensor type and has shape of (N, 4)
        labels: a label with torch tensor type and has shape of (M, 4)
            where the format of logits and labels have to has the type of (x1, y1, x2, y2) with integers
    Examples:
        >>> iou_loss = IoULoss()
        >>> outputs = model(images)
        >>> loss = iou_loss(outputs, labels)
        >>> loss.backward()
    """
    def __init__(self, smooth: float=1e-9, reduction: str='mean'):
        super(IoULoss, self).__init__()
        assert reduction in ('none', 'mean', 'sum'), \
            f'{reduction} does not exists, choice between (none, mean, sum).'
        self.smooth = smooth
        self.reduction = reduction
    
    def forward(self, logits: torch.Tensor, labels: torch.Tensor):
        iou = iou_func(logits, labels, self.smooth)
        loss = 1 - iou

        if self.reduction == 'none':
            return loss
        elif self.reduction == 'mean':
            return loss.mean()
        else:
            return loss.sum()


def iou_func(logits, labels, smooth):
    logits_area = (logits[:, 2] - logits[:, 0]) * (logits[:, 3] - logits[:, 1])
    labels_area = (labels[:, 2] - labels[:, 0]) * (labels[:, 3] - labels[:, 1])

    lt = torch.max(labels[:, :2], logits[:, :2])
    rb = torch.min(labels[:, 2:], logits[:, 2:])
    wh = (rb - lt + smooth).clamp(min=0)
    
    intersection = wh[:, 0] * wh[:, 1]
    union = logits_area + labels_area - intersection
    iou = intersection / union
    return iou