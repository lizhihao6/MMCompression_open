# Copyright (c) NJU Vision Lab. All rights reserved.
from typing import Any

import torch
from torch import Tensor
from torch import nn

from .weighted_gaussian import GaussianEntropy
from ..builder import CONTEXT


class MultistageMaskedConv2d(nn.Conv2d):
    """
    Masked Convolution 2D.
    Args:
        mask_type (str): Mask type ['A', 'B', 'C'].
    """

    def __init__(self, *args: Any, mask_type: str = "A", **kwargs: Any):
        super().__init__(*args, **kwargs)

        self.register_buffer("mask", torch.zeros_like(self.weight.data))

        if mask_type == "A":
            self.mask[:, :, 0::2, 0::2] = 1
        elif mask_type == "B":
            self.mask[:, :, 0::2, 1::2] = 1
            self.mask[:, :, 1::2, 0::2] = 1
        elif mask_type == "C":
            self.mask[:, :, :, :] = 1
            self.mask[:, :, 1:2, 1:2] = 0
        else:
            raise ValueError(f'Invalid "mask_type" value "{mask_type}"')

    def forward(self, input: Tensor) -> Tensor:
        """
        Masked convolution 2D using masked weights.
        Args:
            input (torch.Tensor): the input tensor.
        Returns:
            torch.Tensor: the output tensor.
        """
        self.weight.data *= self.mask
        return super().forward(input)


@CONTEXT.register_module()
class MCM(nn.Module):
    """
    Checkerboard context model.
    Args:
        main_channels (int): Number of channels in the main branch.
    """

    def __init__(self, main_channels):
        super().__init__()
        self.gaussian_conditional = GaussianEntropy()

        self.context_prediction_1 = MultistageMaskedConv2d(
            main_channels,
            main_channels * 2,
            kernel_size=3,
            padding=1,
            stride=1,
            mask_type="A",
        )
        self.context_prediction_2 = MultistageMaskedConv2d(
            main_channels,
            main_channels * 2,
            kernel_size=3,
            padding=1,
            stride=1,
            mask_type="B",
        )
        self.context_prediction_3 = MultistageMaskedConv2d(
            main_channels,
            main_channels * 2,
            kernel_size=3,
            padding=1,
            stride=1,
            mask_type="C",
        )

        self.entropy_parameters = nn.Sequential(
            nn.Conv2d(main_channels * 24 // 3, main_channels * 18 // 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(main_channels * 18 // 3, main_channels * 12 // 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(main_channels * 12 // 3, main_channels * 6 // 3, 1, 1),
        )

    def forward(self, x, x_prob):
        """
        Args:
            x (torch.Tensor): the input tensor.
            x_prob (torch.Tensor): the input hyper prior tensor.
        Returns:
            likelihood (torch.Tensor): the estimated entropy.
        """
        predicted = torch.zeros_like(x)
        predicted[:, :, 0::2, 0::2] = x[:, :, 0::2, 0::2]
        ctx_params_1 = self.context_prediction_1(predicted.detach())
        ctx_params_1[:, :, 0::2, :] = 0
        ctx_params_1[:, :, 1::2, 0::2] = 0

        predicted[:, :, 1::2, 1::2] = x[:, :, 1::2, 1::2]
        ctx_params_2 = self.context_prediction_2(predicted.detach())
        ctx_params_2[:, :, 0::2, 0::2] = 0
        ctx_params_2[:, :, 1::2, :] = 0

        predicted[:, :, 0::2, 1::2] = x[:, :, 0::2, 1::2]
        ctx_params_3 = self.context_prediction_3(predicted.detach())
        ctx_params_3[:, :, 0::2, :] = 0
        ctx_params_3[:, :, 1::2, 1::2] = 0

        prob = self.entropy_parameters(
            torch.cat((x_prob, ctx_params_1, ctx_params_2, ctx_params_3), dim=1)
        )
        likelihoods = self.gaussian_conditional(x, prob)
        return likelihoods
