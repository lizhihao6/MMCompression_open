import torch
import torch.nn as nn

from ..builder import MAINDECODER
from ..utils import ResBlock, Non_local_Block


@MAINDECODER.register_module()
class NICDec(nn.Module):
    def __init__(self, output_channels=3, root_channels=96, main_channels=192):
        super(NICDec, self).__init__()
        self.trunk1 = nn.Sequential(ResBlock(main_channels, main_channels, 3, 1, 1),
                                    ResBlock(main_channels, main_channels, 3, 1, 1),
                                    ResBlock(main_channels, main_channels, 3, 1, 1))
        self.mask1 = nn.Sequential(Non_local_Block(main_channels, main_channels // 2),
                                   ResBlock(main_channels, main_channels, 3, 1, 1),
                                   ResBlock(main_channels, main_channels, 3, 1, 1),
                                   ResBlock(main_channels, main_channels, 3, 1, 1),
                                   nn.Conv2d(main_channels, main_channels, 1, 1, 0))
        self.up1 = nn.ConvTranspose2d(main_channels, main_channels, 5, 2, 2, 1)
        self.trunk2 = nn.Sequential(ResBlock(main_channels, main_channels, 3, 1, 1),
                                    ResBlock(main_channels, main_channels, 3, 1, 1),
                                    ResBlock(main_channels, main_channels, 3, 1, 1),
                                    nn.ConvTranspose2d(main_channels, main_channels, 5, 2, 2, 1))
        self.trunk3 = nn.Sequential(ResBlock(main_channels, main_channels, 3, 1, 1),
                                    ResBlock(main_channels, main_channels, 3, 1, 1),
                                    ResBlock(main_channels, main_channels, 3, 1, 1),
                                    nn.ConvTranspose2d(main_channels, 2 * root_channels, 5, 2, 2, 1))
        self.trunk4 = nn.Sequential(ResBlock(2 * root_channels, 2 * root_channels, 3, 1, 1),
                                    ResBlock(2 * root_channels, 2 * root_channels, 3, 1, 1),
                                    ResBlock(2 * root_channels, 2 * root_channels, 3, 1, 1))
        self.trunk5 = nn.Sequential(nn.ConvTranspose2d(2 * root_channels, root_channels, 5, 2, 2, 1),
                                    ResBlock(root_channels, root_channels, 3, 1, 1),
                                    ResBlock(root_channels, root_channels, 3, 1, 1),
                                    ResBlock(root_channels, root_channels, 3, 1, 1))
        self.conv1 = nn.Conv2d(root_channels, output_channels, 5, 1, 2)

    def forward(self, x):
        x1 = self.trunk1(x) * torch.sigmoid(self.mask1(x)) + x
        x1 = self.up1(x1)
        x2 = self.trunk2(x1)
        x3 = self.trunk3(x2)
        x4 = self.trunk4(x3) + x3
        x5 = self.trunk5(x4)
        output = self.conv1(x5)
        return output
