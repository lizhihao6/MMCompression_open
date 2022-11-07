# Copyright (c) NJU Vision Lab. All rights reserved.
"""
Neighborhood Attention PyTorch Module (Based on existing torch modules)
This version does not require the torch extension and is implemented using unfold + pad.
This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""
"""
Neighborhood Attention PyTorch Module (CUDA only)
This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""
import os
import warnings
from typing import Optional

import torch
from mmcv.cnn.bricks.transformer import DropPath
from torch import nn
from torch.autograd import Function
from torch.cuda.amp import custom_fwd, custom_bwd
from torch.nn.functional import pad
from torch.utils.cpp_extension import load, is_ninja_available

## move NATTENAVFunction and NATTENQKRPBFunction into mmcv

if is_ninja_available():
    this_dir = os.path.dirname(os.path.realpath(__file__))
    nattenav_cuda = load(
        'nattenav_cuda', [f'{this_dir}/src/nattenav_cuda.cpp', f'{this_dir}/src/nattenav_cuda_kernel.cu'],
        verbose=False)
    nattenqkrpb_cuda = load(
        'nattenqkrpb_cuda', [f'{this_dir}/src/nattenqkrpb_cuda.cpp', f'{this_dir}/src/nattenqkrpb_cuda_kernel.cu'],
        verbose=False)
else:
    warnings.warn("Ninja is not installed, looking up extensions manually.")
    try:
        import nattenav_cuda
        import nattenqkrpb_cuda
    except:
        raise RuntimeError("Could not load NATTEN CUDA extension. " +
                           "Please make sure your device has CUDA, the CUDA toolkit for PyTorch is installed, and that you've compiled NATTEN correctly.")


class NATTENAVFunction(Function):
    """
    AV autograd function
    Computes neighborhood attention outputs given attention weights, and values.
    This calls the `AV` kernel.
    """

    @staticmethod
    @custom_fwd(cast_inputs=torch.float16)
    def forward(ctx, attn, value):
        attn = attn.contiguous()
        value = value.contiguous()
        out = nattenav_cuda.forward(
            attn,
            value)
        ctx.save_for_backward(attn, value)
        return out

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_out):
        outputs = nattenav_cuda.backward(
            grad_out.contiguous(), ctx.saved_variables[0], ctx.saved_variables[1])
        d_attn, d_value = outputs
        return d_attn, d_value, None


class NATTENQKRPBFunction(Function):
    """
    QK+RPB autograd function
    Computes neighborhood attention weights given queries and keys,
    and adds relative positional biases.
    This calls the `QKRPB` kernel.
    """

    @staticmethod
    @custom_fwd(cast_inputs=torch.float16)
    def forward(ctx, query, key, rpb):
        query = query.contiguous()
        key = key.contiguous()
        attn = nattenqkrpb_cuda.forward(
            query,
            key,
            rpb)
        ctx.save_for_backward(query, key)
        return attn

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_out):
        outputs = nattenqkrpb_cuda.backward(
            grad_out.contiguous(), ctx.saved_variables[0], ctx.saved_variables[1])
        d_query, d_key, d_rpb = outputs
        return d_query, d_key, d_rpb, None


## end of NATTENAVFunction and NATTENQKRPBFunction

class NeighborhoodAttention(nn.Module):
    """
    Neighborhood Attention Module
    """

    def __init__(self, dim: int, kernel_size: int, num_heads: int,
                 qkv_bias: bool = True, qk_scale: Optional = None, attn_drop: float = 0., proj_drop: float = 0.):
        """
        Args:
            dim (int): Number of input channels.
            kernel_size (int): Size of the convolutional kernel.
            num_heads (int): Number of attention heads.
            qkv_bias (bool):  If True, add a learnable bias to q, k, v. Default: True
            qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
            attn_drop (float): Dropout ratio of attention weight. Default: 0.0
            proj_drop (float): Dropout ratio of output. Default: 0.0
        """
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // self.num_heads
        self.scale = qk_scale or self.head_dim ** -0.5
        assert kernel_size > 1 and kernel_size % 2 == 1, \
            f"Kernel size must be an odd number greater than 1, got {kernel_size}."
        assert kernel_size in [3, 5, 7, 9, 11, 13], \
            f"CUDA kernel only supports kernel sizes 3, 5, 7, 9, 11, and 13; got {kernel_size}."
        self.kernel_size = kernel_size

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.rpb = nn.Parameter(torch.zeros(num_heads, (2 * kernel_size - 1), (2 * kernel_size - 1)))
        # todo : how could I init a specific parameter?
        # trunc_normal_(self.rpb, std=.02, mean=0., a=-2., b=2.)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor):
        """
        Args:
            x (torch.Tensor): Input feature map with shape of (B, H_p, W_p, C).
        """
        B, Hp, Wp, C = x.shape
        H, W = Hp, Wp
        pad_l = pad_t = pad_r = pad_b = 0
        if H < self.kernel_size or W < self.kernel_size:
            pad_l = pad_t = 0
            pad_r = max(0, self.window_size - W)
            pad_b = max(0, self.window_size - H)
            x = pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))
            _, H, W, _ = x.shape
        qkv = self.qkv(x).reshape(B, H, W, 3, self.num_heads, self.head_dim).permute(3, 0, 4, 1, 2, 5)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = q * self.scale
        attn = NATTENQKRPBFunction.apply(q, k, self.rpb)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = NATTENAVFunction.apply(attn, v)
        x = x.permute(0, 2, 3, 1, 4).reshape(B, H, W, C)
        if pad_r or pad_b:
            x = x[:, :Hp, :Wp, :]

        return self.proj_drop(self.proj(x))


class Mlp(nn.Module):
    """
    MLP module
    """

    def __init__(self, in_features: int, hidden_features: Optional = None,
                 out_features: Optional = None,
                 act_layer: nn.Module = nn.GELU, drop: float = 0.):
        """
        Args:
            in_features (int): Number of input channels.
            hidden_features (int | None, optional): Number of hidden channels. Default: in_features
            out_features (int | None, optional): Number of output channels. Default: in_features
            act_layer (nn.Module): Activation layer. Default: nn.GELU
            drop (float): Dropout rate. Default: 0.0
        """
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor):
        """
        Args:
            x (torch.Tensor): Input feature map with shape of (B, H, W, C).
        """
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class NSABlock(nn.Module):
    """
    Neighborhood Self-Attention Block
    """

    def __init__(self, dim: int, num_heads: int, kernel_size: int = 7,
                 mlp_ratio: int = 4, qkv_bias: bool = True, qk_scale: Optional = None,
                 drop: float = 0., attn_drop: float = 0., drop_path: float = 0.,
                 act_layer: nn.Module = nn.GELU, norm_layer: nn.Module = nn.LayerNorm):
        """
        Args:
            dim (int): Number of input channels.
            num_heads (int): Number of attention heads.
            kernel_size (int): Size of the convolutional kernel.
            mlp_ratio (int): Ratio of mlp hidden dim to embedding dim.
            qkv_bias (bool): If True, add a learnable bias to q, k, v. Default: True
            qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
            drop (float): Dropout rate.
            attn_drop (float): Attention dropout rate.
            drop_path (float): Stochastic depth rate.
            act_layer (nn.Module): Activation layer.
            norm_layer (nn.Module): Normalization layer.
        """
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio

        self.norm1 = norm_layer(dim)
        self.attn = NeighborhoodAttention(
            dim, kernel_size=kernel_size, num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, drop=drop)

    def forward(self, x: torch.Tensor):
        """
        Args:
            x (torch.Tensor): Input feature map with shape of (B, H, W, C).
        """
        shortcut = x
        x = self.norm1(x)
        x = self.attn(x)
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class BasicViTLayer(nn.Module):
    """
    Basic ViT Layer
    """

    def __init__(self, dim: int, depth: int, num_heads: int, kernel_size: int, mlp_ratio: int = 4,
                 qkv_bias: bool = True, qk_scale: Optional = None, drop: float = 0., attn_drop: float = 0.,
                 drop_path: Optional = 0., norm_layer: nn.Module = nn.LayerNorm):
        """
        Args:
            dim (int): Number of input channels.
            depth (int): Number of blocks.
            num_heads (int): Number of attention heads.
            kernel_size (int): Size of the convolutional kernel.
            mlp_ratio (int): Ratio of mlp hidden dim to embedding dim.
            qkv_bias (bool): If True, add a learnable bias to q, k, v. Default: True
            qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
            drop (float): Dropout rate.
            attn_drop (float | list, optional): Attention dropout rate.
            drop_path (float): Stochastic depth rate.
            norm_layer (nn.Module): Normalization layer.
        """
        super().__init__()
        self.dim = dim
        self.depth = depth

        self.blocks = nn.ModuleList([
            NSABlock(dim=dim,
                     num_heads=num_heads, kernel_size=kernel_size,
                     mlp_ratio=mlp_ratio,
                     qkv_bias=qkv_bias, qk_scale=qk_scale,
                     drop=drop, attn_drop=attn_drop,
                     drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                     norm_layer=norm_layer)
            for i in range(depth)])

    def forward(self, x: torch.Tensor):
        """
        Args:
            x (torch.Tensor): Input feature map with shape of (B, H, W, C).
        """
        for blk in self.blocks:
            x = blk(x)
        return x


class ResViTBlock(nn.Module):
    def __init__(self, dim: int, depth: int, num_heads: int, kernel_size: int = 7, mlp_ratio: int = 4,
                 qkv_bias: bool = True, qk_scale: Optional = None, drop_rate: float = 0., attn_drop_rate: float = 0.,
                 drop_path_rate: Optional = 0.2, norm_layer: nn.Module = nn.LayerNorm):
        super(ResViTBlock, self).__init__()
        """
        Args:
            dim (int): Number of input channels.
            depth (int): Number of blocks.
            num_heads (int): Number of attention heads.
            kernel_size (int): Size of the convolutional kernel.
            mlp_ratio (int): Ratio of mlp hidden dim to embedding dim.
            qkv_bias (bool): If True, add a learnable bias to q, k, v. Default: True
            qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
            drop_rate (float): Dropout rate.
            attn_drop_rate (float): Attention dropout rate.
            drop_path_rate (float | list, Optional): Stochastic depth rate.
            norm_layer (nn.Module): Normalization layer.
        """
        self.dim = dim

        self.residual_group = BasicViTLayer(dim=dim, depth=depth, num_heads=num_heads, kernel_size=kernel_size,
                                            mlp_ratio=mlp_ratio,
                                            qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate,
                                            attn_drop=attn_drop_rate,
                                            drop_path=drop_path_rate, norm_layer=norm_layer)

    def forward(self, x: torch.Tensor):
        """
        Args:
            x (torch.Tensor): Input feature map with shape of (B, C, H, W).
        """
        return self.residual_group(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2) + x
