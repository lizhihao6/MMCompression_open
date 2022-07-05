import torch
import torch.nn as nn

from ..builder import HYPERENCODER
from ..utils import ResBlock, Non_local_Block


@HYPERENCODER.register_module()
class NICEnc_hyper(nn.Module):
    def __init__(self, main_channels=192, hyper_channels=128):
        super(NICEnc_hyper, self).__init__()

        self.trunk6 = nn.Sequential(ResBlock(main_channels, main_channels, 3, 1, 1),
                                    ResBlock(main_channels, main_channels, 3, 1, 1),
                                    nn.Conv2d(main_channels, main_channels, 5, 2, 2))
        self.trunk7 = nn.Sequential(ResBlock(main_channels, main_channels, 3, 1, 1),
                                    ResBlock(main_channels, main_channels, 3, 1, 1),
                                    nn.Conv2d(main_channels, main_channels, 5, 2, 2))

        self.trunk8 = nn.Sequential(ResBlock(main_channels, main_channels, 3, 1, 1),
                                    ResBlock(main_channels, main_channels, 3, 1, 1),
                                    ResBlock(main_channels, main_channels, 3, 1, 1))
        self.mask3 = nn.Sequential(Non_local_Block(main_channels, main_channels // 2),
                                   ResBlock(main_channels, main_channels, 3, 1, 1),
                                   ResBlock(main_channels, main_channels, 3, 1, 1),
                                   ResBlock(main_channels, main_channels, 3, 1, 1),
                                   nn.Conv2d(main_channels, main_channels, 1, 1, 0))
        self.conv2 = nn.Conv2d(main_channels, hyper_channels, 3, 1, 1)

    def forward(self, x6):
        x7 = self.trunk6(x6)
        x8 = self.trunk7(x7)
        x9 = self.trunk8(x8) * torch.sigmoid(self.mask3(x8)) + x8
        x10 = self.conv2(x9)

        return x10
