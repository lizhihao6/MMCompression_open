import torch

from ..builder import QUANTS

__all__ = [
    'X3Quant', 'UniverseQuant',
]


@QUANTS.register_module()
class X3Quant(torch.nn.Module):
    def __init__(self):
        super(X3Quant, self).__init__()

    def forward(self, x):
        return torch.round(x) + (x - torch.round(x)) ** 3


@QUANTS.register_module()
class UniverseQuant(torch.nn.Module):
    def __init__(self):
        super(UniverseQuant, self).__init__()

    def forward(self, x):
        half = float(0.5)
        noise = torch.empty_like(x).uniform_(-half, half)
        x = x + noise
        return x
