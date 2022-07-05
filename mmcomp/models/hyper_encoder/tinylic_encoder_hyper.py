import torch
import torch.nn as nn

from ..builder import HYPERENCODER
from ..utils import RSTB


@HYPERENCODER.register_module()
class TinyLICEnc_hyper(nn.Module):
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

        self.h_a0 = nn.Conv2d(main_channels, hyper_channels, kernel_size=3, stride=2, padding=1)
        self.h_a1 = RSTB(dim=hyper_channels,
                         input_resolution=(8, 8),
                         depth=depths[4],
                         num_heads=num_heads[4],
                         window_size=window_size // 2,
                         mlp_ratio=mlp_ratio,
                         qkv_bias=qkv_bias, qk_scale=qk_scale,
                         drop=drop_rate, attn_drop=attn_drop_rate,
                         drop_path=dpr[sum(depths[:4]):sum(depths[:5])],
                         norm_layer=norm_layer,
                         use_checkpoint=use_checkpoint,
                         )
        self.h_a2 = nn.Conv2d(hyper_channels, hyper_channels, kernel_size=3, stride=2, padding=1)
        self.h_a3 = RSTB(dim=hyper_channels,
                         input_resolution=(4, 4),
                         depth=depths[5],
                         num_heads=num_heads[5],
                         window_size=window_size // 2,
                         mlp_ratio=mlp_ratio,
                         qkv_bias=qkv_bias, qk_scale=qk_scale,
                         drop=drop_rate, attn_drop=attn_drop_rate,
                         drop_path=dpr[sum(depths[:5]):sum(depths[:6])],
                         norm_layer=norm_layer,
                         use_checkpoint=use_checkpoint,
                         )

    def forward(self, x):
        x_size = (x.shape[2] * 16, x.shape[3] * 16)
        x = self.h_a0(x)
        x = self.h_a1(x, (x_size[0] // 32, x_size[1] // 32))
        x = self.h_a2(x)
        x = self.h_a3(x, (x_size[0] // 64, x_size[1] // 64))
        return x
