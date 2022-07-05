import torch
import torch.nn as nn

from ..builder import HYPERDECODER
from ..utils import RSTB


@HYPERDECODER.register_module()
class TinyLICDec_hyper(nn.Module):
    def __init__(self, main_channels=192, hyper_channels=128):
        super().__init__()
        depths = [2, 4, 6, 2, 2, 2]
        num_heads = [8, 8, 8, 16, 16, 16]
        window_size = 8
        mlp_ratio = 2.
        qkv_bias = True
        qk_scale = None
        drop_rate = 0.
        attn_drop_rate = 0.
        drop_path_rate = 0.1
        norm_layer = nn.LayerNorm
        use_checkpoint = False

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        depths = depths[::-1]
        num_heads = num_heads[::-1]
        self.h_s0 = RSTB(dim=hyper_channels,
                         input_resolution=(4, 4),
                         depth=depths[0],
                         num_heads=num_heads[0],
                         window_size=window_size // 2,
                         mlp_ratio=mlp_ratio,
                         qkv_bias=qkv_bias, qk_scale=qk_scale,
                         drop=drop_rate, attn_drop=attn_drop_rate,
                         drop_path=dpr[sum(depths[:0]):sum(depths[:1])],
                         norm_layer=norm_layer,
                         use_checkpoint=use_checkpoint,
                         )
        self.h_s1 = nn.ConvTranspose2d(hyper_channels, hyper_channels, kernel_size=3, stride=2, output_padding=1,
                                       padding=1)
        self.h_s2 = RSTB(dim=hyper_channels,
                         input_resolution=(8, 8),
                         depth=depths[1],
                         num_heads=num_heads[1],
                         window_size=window_size // 2,
                         mlp_ratio=mlp_ratio,
                         qkv_bias=qkv_bias, qk_scale=qk_scale,
                         drop=drop_rate, attn_drop=attn_drop_rate,
                         drop_path=dpr[sum(depths[:1]):sum(depths[:2])],
                         norm_layer=norm_layer,
                         use_checkpoint=use_checkpoint,
                         )
        self.h_s3 = nn.ConvTranspose2d(hyper_channels, main_channels * 2, kernel_size=3, stride=2, output_padding=1,
                                       padding=1)

    def forward(self, x):
        x_size = (x.shape[2] * 64, x.shape[3] * 64)
        x = self.h_s0(x, (x_size[0] // 64, x_size[1] // 64))
        x = self.h_s1(x)
        x = self.h_s2(x, (x_size[0] // 32, x_size[1] // 32))
        x = self.h_s3(x)
        return x
