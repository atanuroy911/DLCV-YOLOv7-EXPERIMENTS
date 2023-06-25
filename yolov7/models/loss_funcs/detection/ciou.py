import math

import torch
import torch.nn as nn

from iou import iou_func
from diou import diou_func


class CIoULoss(nn.Module):
    """
    Args:
        smooth: a value to avoid dividing by zero
        reduction: the output type that can be selected from none, sum and mean
    Forward:
        logits: a prediction with torch tensor type and has shape of (N, 4)
        labels: a label with torch tensor type and has shape of (M, 4)
    Examples:
        >>> ciou_loss = CIoULoss()
        >>> outputs = model(images)
        >>> loss = ciou_loss(outputs, labels)
        >>> loss.backward()
    """
    def __init__(self, smooth: float=1e-9, reduction='none'):
        super(CIoULoss, self).__init__()
        assert reduction in ('sum', 'mean', 'none')
        self.reduction = reduction
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, labels: torch.Tensor):
        ciou_loss = ciou_loss_func(logits, labels, self.smooth)

        if self.reduction == 'sum':
            return ciou_loss.sum()
        elif self.reduction == 'mean':
            return ciou_loss.mean()
        else:
            return ciou_loss


def ciou_loss_func(logits, labels, smooth):
    iou = iou_func(logits, labels, smooth)
    diou = diou_func(logits, labels, smooth)

    logits_center_x, logits_center_y = logits[:, 0], logits[:, 1]
    labels_center_x, labels_center_y = labels[:, 0], labels[:, 1]

    logits_x1, logits_y1 = logits[:, 0] - logits[:, 2] / 2, logits[:, 1] - logits[:, 3] / 2
    logits_x2, logits_y2 = logits[:, 0] + logits[:, 2] / 2, logits[:, 1] + logits[:, 3] / 2

    labels_x1, labels_y1 = labels[:, 0] - labels[:, 2] / 2, labels[:, 1] - labels[:, 3] / 2
    labels_x2, labels_y2 = labels[:, 0] + labels[:, 2] / 2, labels[:, 1] + labels[:, 3] / 2

    logits_width, logits_height = logits_x2 - logits_x1, logits_y2 - logits_y1
    labels_width, labels_height = labels_x2 - labels_x1, labels_y2 - labels_y1

    v = (4 / math.pi ** 2) * torch.pow(
        torch.atan(labels_width / (labels_height + smooth)) - torch.atan(logits_width / (logits_height + smooth)), 2)
    
    with torch.no_grad():
        alpha = v / (1 - iou + v + smooth)
    
    ciou_loss = 1 - diou + alpha * v
    return ciou_loss