import torch
import torch.nn as nn

from iou import iou_func


class DIoULoss(nn.Module):
    """
    Args:
        smooth: a value to avoid dividing by zero
        reduction: the output type that can be selected from none, sum and mean
    Forward:
        logits: a prediction with torch tensor type and has shape of (N, 4)
        labels: a label with torch tensor type and has shape of (M, 4)
    Examples:
        >>> diou_loss = DIoULoss()
        >>> outputs = model(images)
        >>> loss = diou_loss(outputs, labels)
        >>> loss.backward()
    """
    def __init__(self, smooth: float=1e-9, reduction: str='mean'):
        super(DIoULoss, self).__init__()
        self.smooth = smooth
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, labels: torch.Tensor):
        diou = diou_func(logits, labels, self.smooth)
        loss = 1 - diou

        if self.reduction == 'none':
            return loss
        elif self.reduction == 'mean':
            return loss.mean()
        else:
            return loss.sum()
            

def diou_func(logits, labels, smooth):
    iou = iou_func(logits, labels)

    # central points of bunding boxes
    logit_center_x = logits[:, 0::2].mean(dim=-1)
    logit_center_y = logits[:, 1::2].mean(dim=-1)
    label_center_x = labels[:, 0::2].mean(dim=-1)
    label_center_y = labels[:, 1::2].mean(dim=-1)
    center_dist = (label_center_x - logit_center_x).pow(2.) + (label_center_y - logit_center_y).pow(2.)

    # diagonal length
    lt = torch.max(labels[:, :2], logits[:, :2])
    rb = torch.min(labels[:, 2:], logits[:, 2:])
    diag_len = (lt - rb).pow(2.).sum(dim=-1)

    # calculate diou score
    penalty = center_dist / (diag_len + smooth)
    diou = iou - penalty
    return diou