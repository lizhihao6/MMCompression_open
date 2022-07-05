import torch
import torch.nn as nn

from ..builder import MAINENCODER
from ..main_decoder.nvc_decoder import TinyBackbone


@MAINENCODER.register_module()
class NVCEnc(nn.Module):
    def __init__(self, input_channels=3, main_channels=192):
        super().__init__()
        self.encoder = TinyBackbone(input_channels * 2, main_channels)

    def forward(self, cur_image, pre_rec_img):
        return self.encoder(torch.cat([cur_image, pre_rec_img], dim=1))
