# Copyright (c) NJU Vision Lab. All rights reserved.
import torch
from torch import nn

from ..builder import VAE
from ..utils import RSTB

depths = [2, 4, 6, 2, 2, 2]
num_heads = [8, 8, 8, 16, 16, 16]
window_size = 8
mlp_ratio = 2.0
qkv_bias = True
qk_scale = None
drop_rate = 0.0
attn_drop_rate = 0.0
drop_path_rate = 0.1
norm_layer = nn.LayerNorm
use_checkpoint = False


class Enc(nn.Module):
    """Swin based Encoder
    Args:
        in_channels (int): The input channels of this Module.
        stem_channels (int): The stem channels of this Module.
        main_channels (int): The main channels of this Module.
    """

    def __init__(self, in_channels=3, stem_channels=96, main_channels=192):
        super().__init__()
        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        self.g_a0 = nn.Conv2d(
            in_channels, stem_channels, kernel_size=5, stride=2, padding=2
        )
        self.g_a1 = RSTB(
            dim=stem_channels,
            input_resolution=(128, 128),
            depth=depths[0],
            num_heads=num_heads[0],
            window_size=window_size,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop=drop_rate,
            attn_drop=attn_drop_rate,
            drop_path=dpr[sum(depths[:0]): sum(depths[:1])],
            norm_layer=norm_layer,
            use_checkpoint=use_checkpoint,
        )
        self.g_a2 = nn.Conv2d(
            stem_channels, stem_channels, kernel_size=3, stride=2, padding=1
        )
        self.g_a3 = RSTB(
            dim=stem_channels,
            input_resolution=(64, 64),
            depth=depths[1],
            num_heads=num_heads[1],
            window_size=window_size,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop=drop_rate,
            attn_drop=attn_drop_rate,
            drop_path=dpr[sum(depths[:1]): sum(depths[:2])],
            norm_layer=norm_layer,
            use_checkpoint=use_checkpoint,
        )
        self.g_a4 = nn.Conv2d(
            stem_channels, stem_channels, kernel_size=3, stride=2, padding=1
        )
        self.g_a5 = RSTB(
            dim=stem_channels,
            input_resolution=(32, 32),
            depth=depths[2],
            num_heads=num_heads[2],
            window_size=window_size,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop=drop_rate,
            attn_drop=attn_drop_rate,
            drop_path=dpr[sum(depths[:2]): sum(depths[:3])],
            norm_layer=norm_layer,
            use_checkpoint=use_checkpoint,
        )
        self.g_a6 = nn.Conv2d(
            stem_channels, main_channels, kernel_size=3, stride=2, padding=1
        )
        self.g_a7 = RSTB(
            dim=main_channels,
            input_resolution=(16, 16),
            depth=depths[3],
            num_heads=num_heads[3],
            window_size=window_size,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop=drop_rate,
            attn_drop=attn_drop_rate,
            drop_path=dpr[sum(depths[:3]): sum(depths[:4])],
            norm_layer=norm_layer,
            use_checkpoint=use_checkpoint,
        )

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor.
        Returns:
            torch.Tensor: Output tensor.
        """
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


class Dec(nn.Module):
    """Swin based Decoder
    Args:
        out_channels (int): The input channels of this Module.
        root_channels (int): The root channels of this Module.
        main_channels (int): The main channels of this Module.
    """

    def __init__(self, out_channels=3, root_channels=96, main_channels=192):
        super().__init__()
        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        _depths = depths[::-1]
        _num_heads = num_heads[::-1]
        self.g_s0 = RSTB(
            dim=main_channels,
            input_resolution=(16, 16),
            depth=_depths[2],
            num_heads=_num_heads[2],
            window_size=window_size,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop=drop_rate,
            attn_drop=attn_drop_rate,
            drop_path=dpr[sum(_depths[:2]): sum(_depths[:3])],
            norm_layer=norm_layer,
            use_checkpoint=use_checkpoint,
        )
        self.g_s1 = nn.ConvTranspose2d(
            main_channels,
            root_channels,
            kernel_size=3,
            stride=2,
            output_padding=1,
            padding=1,
        )
        self.g_s2 = RSTB(
            dim=root_channels,
            input_resolution=(32, 32),
            depth=_depths[3],
            num_heads=_num_heads[3],
            window_size=window_size,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop=drop_rate,
            attn_drop=attn_drop_rate,
            drop_path=dpr[sum(_depths[:3]): sum(_depths[:4])],
            norm_layer=norm_layer,
            use_checkpoint=use_checkpoint,
        )
        self.g_s3 = nn.ConvTranspose2d(
            root_channels,
            root_channels,
            kernel_size=3,
            stride=2,
            output_padding=1,
            padding=1,
        )
        self.g_s4 = RSTB(
            dim=root_channels,
            input_resolution=(64, 64),
            depth=_depths[4],
            num_heads=_num_heads[4],
            window_size=window_size,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop=drop_rate,
            attn_drop=attn_drop_rate,
            drop_path=dpr[sum(_depths[:4]): sum(_depths[:5])],
            norm_layer=norm_layer,
            use_checkpoint=use_checkpoint,
        )
        self.g_s5 = nn.ConvTranspose2d(
            root_channels,
            root_channels,
            kernel_size=3,
            stride=2,
            output_padding=1,
            padding=1,
        )
        self.g_s6 = RSTB(
            dim=root_channels,
            input_resolution=(128, 128),
            depth=_depths[5],
            num_heads=_num_heads[5],
            window_size=window_size,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop=drop_rate,
            attn_drop=attn_drop_rate,
            drop_path=dpr[sum(_depths[:5]): sum(_depths[:6])],
            norm_layer=norm_layer,
            use_checkpoint=use_checkpoint,
        )
        self.g_s7 = nn.ConvTranspose2d(
            root_channels,
            out_channels,
            kernel_size=5,
            stride=2,
            output_padding=1,
            padding=2,
        )

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


class HyperEnc(nn.Module):
    """Swin based Hyper-Encoder
    Args:
        main_channels (int): The main channels of this Module.
        hyper_channels (int): The hyper channels of this Module.
    """

    def __init__(self, main_channels=192, hyper_channels=128):
        super().__init__()
        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        self.h_a0 = nn.Conv2d(
            main_channels, hyper_channels, kernel_size=3, stride=2, padding=1
        )
        self.h_a1 = RSTB(
            dim=hyper_channels,
            input_resolution=(8, 8),
            depth=depths[4],
            num_heads=num_heads[4],
            window_size=window_size // 2,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop=drop_rate,
            attn_drop=attn_drop_rate,
            drop_path=dpr[sum(depths[:4]): sum(depths[:5])],
            norm_layer=norm_layer,
            use_checkpoint=use_checkpoint,
        )
        self.h_a2 = nn.Conv2d(
            hyper_channels, hyper_channels, kernel_size=3, stride=2, padding=1
        )
        self.h_a3 = RSTB(
            dim=hyper_channels,
            input_resolution=(4, 4),
            depth=depths[5],
            num_heads=num_heads[5],
            window_size=window_size // 2,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop=drop_rate,
            attn_drop=attn_drop_rate,
            drop_path=dpr[sum(depths[:5]): sum(depths[:6])],
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


class HyperDec(nn.Module):
    """Swin based Hyper-Decoder
    Args:
        main_channels (int): The main channels of this Module.
        hyper_channels (int): The hyper channels of this Module.
    """

    def __init__(self, main_channels=192, hyper_channels=128):
        super().__init__()
        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        _depths = depths[::-1]
        _num_heads = num_heads[::-1]
        self.h_s0 = RSTB(
            dim=hyper_channels,
            input_resolution=(4, 4),
            depth=_depths[0],
            num_heads=_num_heads[0],
            window_size=window_size // 2,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop=drop_rate,
            attn_drop=attn_drop_rate,
            drop_path=dpr[sum(_depths[:0]): sum(_depths[:1])],
            norm_layer=norm_layer,
            use_checkpoint=use_checkpoint,
        )
        self.h_s1 = nn.ConvTranspose2d(
            hyper_channels,
            hyper_channels,
            kernel_size=3,
            stride=2,
            output_padding=1,
            padding=1,
        )
        self.h_s2 = RSTB(
            dim=hyper_channels,
            input_resolution=(8, 8),
            depth=_depths[1],
            num_heads=_num_heads[1],
            window_size=window_size // 2,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop=drop_rate,
            attn_drop=attn_drop_rate,
            drop_path=dpr[sum(_depths[:1]): sum(_depths[:2])],
            norm_layer=norm_layer,
            use_checkpoint=use_checkpoint,
        )
        self.h_s3 = nn.ConvTranspose2d(
            hyper_channels,
            main_channels * 2,
            kernel_size=3,
            stride=2,
            output_padding=1,
            padding=1,
        )

    def forward(self, x):
        x_size = (x.shape[2] * 64, x.shape[3] * 64)
        x = self.h_s0(x, (x_size[0] // 64, x_size[1] // 64))
        x = self.h_s1(x)
        x = self.h_s2(x, (x_size[0] // 32, x_size[1] // 32))
        x = self.h_s3(x)
        return x


@VAE.register_module()
class SwinVAE(nn.Module):
    """Swin based VAE
    Args:
        in_channels (int): The channels of input images.
        stem_channels (int): The channels of the stem convolution.
        main_channels (int): The main channels of this Module.
        hyper_channels (int): The hyper channels of this Module.
    """

    def __init__(
            self, in_channels=3, stem_channels=96, main_channels=192, hyper_channels=128
    ):
        super().__init__()
        self.enc = Enc(
            in_channels=in_channels,
            stem_channels=stem_channels,
            main_channels=main_channels,
        )
        self.dec = Dec(
            out_channels=in_channels,
            root_channels=stem_channels,
            main_channels=main_channels,
        )
        self.hyper_enc = HyperEnc(
            main_channels=main_channels, hyper_channels=hyper_channels
        )
        self.hyper_dec = HyperDec(
            main_channels=main_channels, hyper_channels=hyper_channels
        )
