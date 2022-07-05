import torch.nn as nn

from ..builder import MAINENCODER
from ..utils import Glow


@MAINENCODER.register_module()
class NICflowEnc(nn.Module):
    def __init__(self, input_channels=3, hidden_channels=192, non_local=False):
        super().__init__()
        self.flow = Glow(input_channels=input_channels, hidden_channels=hidden_channels, L=3, non_local=non_local)

    def forward(self, x, reverse=False):
        return self.flow(x, reverse=reverse)
