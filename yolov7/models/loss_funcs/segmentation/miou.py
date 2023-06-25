import numpy as np
import torch
import torch.nn as nn


class MeanIOU(nn.Module):
    """
    Args:
        num_classes: the number of classes of dataset
        smooth: a value to avoid dividing by zero

    Forward:
        logits: a prediction with torch tensor type and has shape of (B, N, W, H)
        labels: a label with torch tensor type and has shape of (B, W, H)
            where B is batch size, N is number of classes, W and H is the width and height of outputs and labels

    Examples:
        >>> metric = MeanIOU(num_classes)
        >>> outputs = model(images)
        >>> miou = metric(outputs, labels)
    """
    def __init__(self, num_classes: int, smooth: float=1e-9):
        super(MeanIOU, self).__init__()
        self.num_classes = num_classes
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, labels: torch.Tensor):
        logits = torch.argmax(logits, dim=1)

        logits = logits.contiguous().view(-1)
        labels = labels.contiguous().view(-1)

        iou_per_class = []

        for clas in range(self.num_classes):
            true_class = logits == clas
            true_label = labels == clas

            if true_label.long().sum().item() == 0:
                iou_per_class.append(np.nan)
            else:
                intersect = torch.logical_and(true_class, true_label).sum().float().item()
                union = torch.logical_or(true_class, true_label).sum().float().item()

                iou = (intersect + self.smooth) / (union + self.smooth)
                iou_per_class.append(iou)

        return np.nanmean(iou_per_class)