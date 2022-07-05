import torch
import torch.nn as nn

from ..builder import HYPERDECODER
from ..utils import ResBlock, Non_local_Block


@HYPERDECODER.register_module()
class NICDec_hyper(nn.Module):
    def __init__(self, main_channels=192, hyper_channels=128):
        super(NICDec_hyper, self).__init__()
        self.conv1 = nn.Conv2d(hyper_channels, main_channels, 3, 1, 1)
        self.trunk1 = nn.Sequential(ResBlock(main_channels, main_channels, 3, 1, 1),
                                    ResBlock(main_channels, main_channels, 3, 1, 1),
                                    ResBlock(main_channels, main_channels, 3, 1, 1))
        self.mask1 = nn.Sequential(Non_local_Block(main_channels, main_channels // 2),
                                   ResBlock(main_channels, main_channels, 3, 1, 1),
                                   ResBlock(main_channels, main_channels, 3, 1, 1),
                                   ResBlock(main_channels, main_channels, 3, 1, 1),
                                   nn.Conv2d(main_channels, main_channels, 1, 1, 0))
        self.trunk2 = nn.Sequential(ResBlock(main_channels, main_channels, 3, 1, 1),
                                    ResBlock(main_channels, main_channels, 3, 1, 1),
                                    nn.ConvTranspose2d(main_channels, main_channels, 5, 2, 2, 1))
        self.trunk3 = nn.Sequential(ResBlock(main_channels, main_channels, 3, 1, 1),
                                    ResBlock(main_channels, main_channels, 3, 1, 1),
                                    nn.ConvTranspose2d(main_channels, main_channels, 5, 2, 2, 1))

        self.context_p = nn.Sequential(ResBlock(main_channels, main_channels, 3, 1, 1),
                                       ResBlock(main_channels, main_channels, 3, 1, 1),
                                       ResBlock(main_channels, main_channels, 3, 1, 1),
                                       nn.Conv2d(main_channels, 2 * main_channels, 3, 1, 1)
                                       )

    def forward(self, x_qunat_hyper):
        x1 = self.conv1(x_qunat_hyper)
        x2 = self.trunk1(x1) * torch.sigmoid(self.mask1(x1)) + x1
        x3 = self.trunk2(x2)
        x4 = self.trunk3(x3)
        x_prob = self.context_p(x4)
        return x_prob
