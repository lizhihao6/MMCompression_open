import torch
import torch.nn as nn

from ..builder import MAINDECODER
from ..utils import RSTB


@MAINDECODER.register_module()
class TinyLICDec(nn.Module):
    def __init__(self, output_channels=3, root_channels=128, main_channels=192):
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
        self.g_s0 = RSTB(dim=main_channels,
                         input_resolution=(16, 16),
                         depth=depths[2],
                         num_heads=num_heads[2],
                         window_size=window_size,
                         mlp_ratio=mlp_ratio,
                         qkv_bias=qkv_bias, qk_scale=qk_scale,
                         drop=drop_rate, attn_drop=attn_drop_rate,
                         drop_path=dpr[sum(depths[:2]):sum(depths[:3])],
                         norm_layer=norm_layer,
                         use_checkpoint=use_checkpoint,
                         )
        self.g_s1 = nn.ConvTranspose2d(main_channels, root_channels, kernel_size=3, stride=2, output_padding=1,
                                       padding=1)
        self.g_s2 = RSTB(dim=root_channels,
                         input_resolution=(32, 32),
                         depth=depths[3],
                         num_heads=num_heads[3],
                         window_size=window_size,
                         mlp_ratio=mlp_ratio,
                         qkv_bias=qkv_bias, qk_scale=qk_scale,
                         drop=drop_rate, attn_drop=attn_drop_rate,
                         drop_path=dpr[sum(depths[:3]):sum(depths[:4])],
                         norm_layer=norm_layer,
                         use_checkpoint=use_checkpoint,
                         )
        self.g_s3 = nn.ConvTranspose2d(root_channels, root_channels, kernel_size=3, stride=2, output_padding=1,
                                       padding=1)
        self.g_s4 = RSTB(dim=root_channels,
                         input_resolution=(64, 64),
                         depth=depths[4],
                         num_heads=num_heads[4],
                         window_size=window_size,
                         mlp_ratio=mlp_ratio,
                         qkv_bias=qkv_bias, qk_scale=qk_scale,
                         drop=drop_rate, attn_drop=attn_drop_rate,
                         drop_path=dpr[sum(depths[:4]):sum(depths[:5])],
                         norm_layer=norm_layer,
                         use_checkpoint=use_checkpoint,
                         )
        self.g_s5 = nn.ConvTranspose2d(root_channels, root_channels, kernel_size=3, stride=2, output_padding=1,
                                       padding=1)
        self.g_s6 = RSTB(dim=root_channels,
                         input_resolution=(128, 128),
                         depth=depths[5],
                         num_heads=num_heads[5],
                         window_size=window_size,
                         mlp_ratio=mlp_ratio,
                         qkv_bias=qkv_bias, qk_scale=qk_scale,
                         drop=drop_rate, attn_drop=attn_drop_rate,
                         drop_path=dpr[sum(depths[:5]):sum(depths[:6])],
                         norm_layer=norm_layer,
                         use_checkpoint=use_checkpoint,
                         )
        self.g_s7 = nn.ConvTranspose2d(root_channels, output_channels, kernel_size=5, stride=2, output_padding=1,
                                       padding=2)

    def forward(self, x):
        x_size = (x.shape[2] * 16, x.shape[3] * 16)
        x = self.g_s0(x, (x_size[0] // 16, x_size[1] // 16))
        x = self.g_s1(x)
        x = self.g_s2(x, (x_size[0] // 8, x_size[1] // 8))
        x = self.g_s3(x)
        x = self.g_s4(x, (x_size[0] // 4, x_size[1] // 4))
        x = self.g_s5(x)
        x = self.g_s6(x, (x_size[0] // 2, x_size[1] // 2))
        x = self.g_s7(x)
        return x
