# Copyright (c) NJU Vision Lab. All rights reserved.
import torch

from ..builder import QUANTS

__all__ = ['X3Quant', 'UniverseQuant', ]


@QUANTS.register_module()
class X3Quant(torch.nn.Module):
    """
    x**3 quantization. noise = x-round(x), return round(x) + noise ** 3
    """

    def forward(self, x):
        return torch.round(x) + (x - torch.round(x)) ** 3


@QUANTS.register_module()
class UniverseQuant(torch.nn.Module):
    """
    Universe Quantization. Random add noise to the input.
    """

    def forward(self, x):
        half = float(0.5)
        noise = torch.empty_like(x).uniform_(-half, half)
        x = x + noise
        return x
