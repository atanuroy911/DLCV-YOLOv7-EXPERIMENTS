# PyTorch losses and metrics
- Write losses and metrics for Detection and Segmentation task

### Segmentation task  
- Weighted Cross Entropy Loss [(code)](https://github.com/Sangh0/pytorch-loss-metric/blob/main/segmentation/weighted_celoss.py)
- Ohem Cross Entropy Loss [(code)](https://github.com/Sangh0/pytorch-loss-metric/blob/main/segmentation/ohem.py)
- Focal Loss [(code)](https://github.com/Sangh0/pytorch-loss-metric/blob/main/segmentation/focal_loss_seg.py)
- Dice Loss [(code)](https://github.com/Sangh0/pytorch-loss-metric/blob/main/segmentation/dice.py)
- Mean IOU [(code)](https://github.com/Sangh0/pytorch-loss-metric/blob/main/segmentation/miou.py)
- Pixel Accuracy [(code)](https://github.com/Sangh0/pytorch-loss-metric/blob/main/segmentation/pixel_accuracy.py)

### Detection task
- IoU [(code)](https://github.com/Sangh0/pytorch-loss-metric/blob/main/detection/iou.py)
- Generalized IoU [(code)](https://github.com/Sangh0/pytorch-loss-metric/blob/main/detection/giou.py) [(paper)](https://arxiv.org/abs/1911.08287)
- Distance IoU [(code)](https://github.com/Sangh0/pytorch-loss-metric/blob/main/detection/diou.py) [(paper)](https://arxiv.org/abs/1911.08287)
- Complete IoU [(code)](https://github.com/Sangh0/pytorch-loss-metric/blob/main/detection/ciou.py) [(paper)](https://arxiv.org/abs/1911.08287)
- Focal Loss [(code)](https://github.com/Sangh0/pytorch-loss-metric/blob/main/detection/focal_loss.py) [(paper)](https://arxiv.org/abs/1708.02002)
- Quality Focal Loss [(code)](https://github.com/Sangh0/pytorch-loss-metric/blob/main/detection/qfocal.py) [(paper)](https://arxiv.org/abs/2006.04388)
- Distribution Focal Loss [(code)](https://github.com/Sangh0/pytorch-loss-metric/blob/main/detection/dfocal.py) [(paper)](https://arxiv.org/abs/2006.04388)
- Varifocal Loss [(code)](https://github.com/Sangh0/pytorch-loss-metric/blob/main/detection/varifocal_loss.py) [(paper)](https://arxiv.org/abs/2008.13367)


```python
from detection.iou import IoULoss
from detection.giou import GIoULoss
from detection.diou import DIoULoss
from detection.ciou import CIoULoss
from detectionl.focal_loss import FocalLoss
from detection.qfocal_loss import QualityFocalLoss
from detection.dfocal_loss import DistributionFocalLoss
from detection.varifocal_loss import VariFocalLoss

from segmentation.ohem import OhemCELoss
from segmentation.weighted_celoss import WeightedCELoss
from segmentation.focal_loss_seg import FocalLoss
from segmentation.dice import DiceLoss

from segmentation.miou import MeanIOU
from segmentation.pixel_accuracy import PixelAccuracy
```