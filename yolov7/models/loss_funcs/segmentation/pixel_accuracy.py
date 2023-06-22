import torch
import torch.nn as nn
import torch.nn.functional as F


class PixelAccuracy(nn.Module):
    """
    Args:
        num_classes: the number of classes of dataset
        smooth: a value to avoid dividing by zero

    Forward:
        logits: a prediction with torch tensor type and has shape of (B, N, W, H)
        labels: a label with torch tensor type and has shape of (B, W, H)
            where B is batch size, N is number of classes, W and H is the width and height of outputs and labels

    Examples:
        >>> metric = PixelAccuracy(num_classes)
        >>> outputs = model(images)
        >>> pix_acc = metric(outputs, labels)
    """
    def __init__(self, num_classes: int, smooth: float=1e-9):
        super(PixelAccuracy, self).__init__()
        self.num_classes = num_classes
        self.smooth = smooth

    def forward(self, logits, labels):
        logits = F.softmax(logits, dim=1)
        logits = torch.argmax(logits, dim=1)

        correct = torch.eq(logits.view(-1), labels.view(-1)).int()
        accuracy = correct.sum() / correct.numel()

        return accuracy