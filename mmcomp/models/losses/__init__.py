import torch
import torch.nn as nn
from pytorch_msssim import MS_SSIM

from ..builder import LOSSES

__all__ = [
    'L1Loss', 'MSELoss', 'MSSSIMLOSS'
]


@LOSSES.register_module()
class L1Loss(nn.Module):
    """L1Loss.
    """

    def __init__(self):
        super(L1Loss, self).__init__()
        self.loss = torch.nn.L1Loss()

    def forward(self, y, gt):
        return self.loss(y, gt)


@LOSSES.register_module()
class MSELoss(nn.Module):
    """MSELoss.
    """

    def __init__(self):
        super(MSELoss, self).__init__()
        self.loss = torch.nn.MSELoss()

    def forward(self, y, gt):
        return self.loss(y, gt)


# todo add channels
@LOSSES.register_module()
class MSSSIMLOSS(nn.Module):
    """MSSSIMLOSS.
    """

    def __init__(self, channel):
        super(MSSSIMLOSS, self).__init__()
        self.loss = MS_SSIM(data_range=1, channel=channel)

    def forward(self, y, gt):
        return 1. - self.loss(y, gt)
