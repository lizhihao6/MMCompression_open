# Copyright (c) NJU Vision Lab. All rights reserved.
import torch
from torch import nn

from ..builder import VAE
from ..utils import ResBlock, Non_local_Block


class Enc(nn.Module):
    """Nonlocal Based Encoder
    Args:
        in_channels (int): The input channels of this Module.
        stem_channels (int): The stem channels of this Module.
        main_channels (int): The main channels of this Module.
    """

    def __init__(self, in_channels=3, stem_channels=96, main_channels=192):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, stem_channels, 5, 1, 2)
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


class Dec(nn.Module):
    """Nonlocal Based Decoder
    Args:
        out_channels (int): The input channels of this Module.
        root_channels (int): The root channels of this Module.
        main_channels (int): The main channels of this Module.
    """

    def __init__(self, out_channels=3, root_channels=96, main_channels=192):
        super().__init__()
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

        hidden_channels = 2 * root_channels
        self.trunk3 = nn.Sequential(ResBlock(main_channels, main_channels, 3, 1, 1),
                                    ResBlock(main_channels, main_channels, 3, 1, 1),
                                    ResBlock(main_channels, main_channels, 3, 1, 1),
                                    nn.ConvTranspose2d(main_channels, hidden_channels, 5, 2, 2, 1))
        self.trunk4 = nn.Sequential(ResBlock(hidden_channels, hidden_channels, 3, 1, 1),
                                    ResBlock(hidden_channels, hidden_channels, 3, 1, 1),
                                    ResBlock(hidden_channels, hidden_channels, 3, 1, 1))
        self.trunk5 = nn.Sequential(nn.ConvTranspose2d(hidden_channels, root_channels, 5, 2, 2, 1),
                                    ResBlock(root_channels, root_channels, 3, 1, 1),
                                    ResBlock(root_channels, root_channels, 3, 1, 1),
                                    ResBlock(root_channels, root_channels, 3, 1, 1))
        self.conv1 = nn.Conv2d(root_channels, out_channels, 5, 1, 2)

    def forward(self, x):
        x1 = self.trunk1(x) * torch.sigmoid(self.mask1(x)) + x
        x1 = self.up1(x1)
        x2 = self.trunk2(x1)
        x3 = self.trunk3(x2)
        x4 = self.trunk4(x3) + x3
        x5 = self.trunk5(x4)
        output = self.conv1(x5)
        return output


class HyperEnc(nn.Module):
    """NonLocal based Hyper-Encoder
    Args:
        main_channels (int): The main channels of this Module.
        hyper_channels (int): The hyper channels of this Module.
    """

    def __init__(self, main_channels=192, hyper_channels=128):
        super().__init__()
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


class HyperDec(nn.Module):
    """NonLocal based Hyper-Decoder
    Args:
        main_channels (int): The main channels of this Module.
        hyper_channels (int): The hyper channels of this Module.
    """

    def __init__(self, main_channels=192, hyper_channels=128):
        super().__init__()
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


@VAE.register_module()
class NonLocalVAE(nn.Module):
    """NonLocal based VAE
    Args:
        in_channels (int): The channels of input images.
        stem_channels (int): The channels of the stem convolution.
        main_channels (int): The main channels of this Module.
        hyper_channels (int): The hyper channels of this Module.
    """

    def __init__(self, in_channels=3, stem_channels=96, main_channels=192, hyper_channels=128):
        super().__init__()
        self.enc = Enc(in_channels=in_channels,
                       stem_channels=stem_channels,
                       main_channels=main_channels)
        self.dec = Dec(out_channels=in_channels,
                       root_channels=stem_channels,
                       main_channels=main_channels)
        self.hyper_enc = HyperEnc(main_channels=main_channels,
                                  hyper_channels=hyper_channels)
        self.hyper_dec = HyperDec(main_channels=main_channels,
                                  hyper_channels=hyper_channels)
