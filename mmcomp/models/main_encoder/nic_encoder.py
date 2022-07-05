import torch
import torch.nn as nn

from ..builder import MAINENCODER
from ..utils import ResBlock, Non_local_Block


@MAINENCODER.register_module()
class NICEnc(nn.Module):
    def __init__(self, input_channels=3, stem_channels=96, main_channels=192):
        # input_features = 3, N1 = 192, N2 = 128, M = 192, M1 = 96
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, stem_channels, 5, 1, 2)
        self.trunk1 = nn.Sequential(ResBlock(stem_channels, stem_channels, 3, 1, 1),
                                    ResBlock(stem_channels, stem_channels, 3, 1, 1),
                                    nn.Conv2d(stem_channels, 2 * stem_channels, 5, 2, 2))

        self.down1 = nn.Conv2d(2 * stem_channels, main_channels, 5, 2, 2)
        self.trunk2 = nn.Sequential(ResBlock(2 * stem_channels, 2 * stem_channels, 3, 1, 1),
                                    ResBlock(2 * stem_channels, 2 * stem_channels, 3, 1, 1),
                                    ResBlock(2 * stem_channels, 2 * stem_channels, 3, 1, 1))
        self.trunk3 = nn.Sequential(ResBlock(main_channels, main_channels, 3, 1, 1),
                                    ResBlock(main_channels, main_channels, 3, 1, 1),
                                    ResBlock(main_channels, main_channels, 3, 1, 1),
                                    nn.Conv2d(main_channels, main_channels, 5, 2, 2))

        self.trunk4 = nn.Sequential(ResBlock(main_channels, main_channels, 3, 1, 1),
                                    ResBlock(main_channels, main_channels, 3, 1, 1),
                                    ResBlock(main_channels, main_channels, 3, 1, 1),
                                    nn.Conv2d(main_channels, main_channels, 5, 2, 2))

        self.trunk5 = nn.Sequential(ResBlock(main_channels, main_channels, 3, 1, 1),
                                    ResBlock(main_channels, main_channels, 3, 1, 1),
                                    ResBlock(main_channels, main_channels, 3, 1, 1))
        self.mask2 = nn.Sequential(Non_local_Block(main_channels, main_channels // 2),
                                   ResBlock(main_channels, main_channels, 3, 1, 1),
                                   ResBlock(main_channels, main_channels, 3, 1, 1),
                                   ResBlock(main_channels, main_channels, 3, 1, 1),
                                   nn.Conv2d(main_channels, main_channels, 1, 1, 0))

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.trunk1(x1)
        x3 = self.trunk2(x2) + x2
        x3 = self.down1(x3)
        x4 = self.trunk3(x3)
        x5 = self.trunk4(x4)
        x6 = self.trunk5(x5) * torch.sigmoid(self.mask2(x5)) + x5
        return x6
