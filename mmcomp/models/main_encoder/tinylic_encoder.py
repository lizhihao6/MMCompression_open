import torch
import torch.nn as nn

from ..builder import MAINENCODER
from ..utils import RSTB


@MAINENCODER.register_module()
class TinyLICEnc(nn.Module):
    def __init__(self, input_channels=3, stem_channels=128, main_channels=192):
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

        self.g_a0 = nn.Conv2d(input_channels, stem_channels, kernel_size=5, stride=2, padding=2)
        self.g_a1 = RSTB(dim=stem_channels,
                         input_resolution=(128, 128),
                         depth=depths[0],
                         num_heads=num_heads[0],
                         window_size=window_size,
                         mlp_ratio=mlp_ratio,
                         qkv_bias=qkv_bias, qk_scale=qk_scale,
                         drop=drop_rate, attn_drop=attn_drop_rate,
                         drop_path=dpr[sum(depths[:0]):sum(depths[:1])],
                         norm_layer=norm_layer,
                         use_checkpoint=use_checkpoint,
                         )
        self.g_a2 = nn.Conv2d(stem_channels, stem_channels, kernel_size=3, stride=2, padding=1)
        self.g_a3 = RSTB(dim=stem_channels,
                         input_resolution=(64, 64),
                         depth=depths[1],
                         num_heads=num_heads[1],
                         window_size=window_size,
                         mlp_ratio=mlp_ratio,
                         qkv_bias=qkv_bias, qk_scale=qk_scale,
                         drop=drop_rate, attn_drop=attn_drop_rate,
                         drop_path=dpr[sum(depths[:1]):sum(depths[:2])],
                         norm_layer=norm_layer,
                         use_checkpoint=use_checkpoint,
                         )
        self.g_a4 = nn.Conv2d(stem_channels, stem_channels, kernel_size=3, stride=2, padding=1)
        self.g_a5 = RSTB(dim=stem_channels,
                         input_resolution=(32, 32),
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
        self.g_a6 = nn.Conv2d(stem_channels, main_channels, kernel_size=3, stride=2, padding=1)
        self.g_a7 = RSTB(dim=main_channels,
                         input_resolution=(16, 16),
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

    def forward(self, x):
        x_size = x.shape[2:4]
        x = self.g_a0(x)
        x = self.g_a1(x, (x_size[0] // 2, x_size[1] // 2))
        x = self.g_a2(x)
        x = self.g_a3(x, (x_size[0] // 4, x_size[1] // 4))
        x = self.g_a4(x)
        x = self.g_a5(x, (x_size[0] // 8, x_size[1] // 8))
        x = self.g_a6(x)
        x = self.g_a7(x, (x_size[0] // 16, x_size[1] // 16))
        return x
