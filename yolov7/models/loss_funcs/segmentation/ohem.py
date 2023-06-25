import torch
import torch.nn as nn


class OhemCELoss(nn.Module):
    """
    Args:
        thresh: a threshold for mining of negative examples
        ignore_index: a ignore class number when calculating loss

    Forward:
        logits: a prediction with torch tensor type and has shape of (B, N, W, H)
        labels: a label with torch tensor type and has shape of (B, W, H)
            where B is batch size, N is number of classes, W and H is the width and height of outputs and labels

    Examples:
        >>> criteria = OhemCELoss()
        >>> outputs = model(images)
        >>> loss = criteria(outputs, labels)
    """
    def __init__(self, thresh: float=0.7, ignore_index: int=255):
        super(OhemCELoss, self).__init__()
        self.thresh = -torch.log(torch.tensor(thresh, requires_grad=False, dtype=torch.float)).cuda()
        self.ignore_index = ignore_index
        self.criteria = nn.CrossEntropyLoss(ignore_index=ignore_index, reduction='none')

    def forward(self, logits: torch.Tensor, labels: torch.Tensor):
        n_min = labels[labels != self.ignore_index].numel() // 16
        loss = self.criteria(logits, labels).view(-1)
        loss_hard = loss[loss > self.thresh]
        if loss_hard.numel() < n_min:
            loss_hard, _ = loss.topk(n_min)
        return torch.mean(loss_hard)