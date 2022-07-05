from typing import Any

import torch
import torch.nn as nn
from torch import Tensor

from .weighted_gaussian import GaussianEntropy
from ..builder import CONTEXT


class MultistageMaskedConv2d(nn.Conv2d):
    def __init__(self, *args: Any, mask_type: str = "A", **kwargs: Any):
        super().__init__(*args, **kwargs)

        self.register_buffer("mask", torch.zeros_like(self.weight.data))

        if mask_type == 'A':
            self.mask[:, :, 0::2, 0::2] = 1
        elif mask_type == 'B':
            self.mask[:, :, 0::2, 1::2] = 1
            self.mask[:, :, 1::2, 0::2] = 1
        elif mask_type == 'C':
            self.mask[:, :, :, :] = 1
            self.mask[:, :, 1:2, 1:2] = 0
        else:
            raise ValueError(f'Invalid "mask_type" value "{mask_type}"')

    def forward(self, x: Tensor) -> Tensor:
        # TODO: weight assigment is not supported by torchscript
        self.weight.data *= self.mask
        return super().forward(x)


@CONTEXT.register_module()
class MCM(nn.Module):
    def __init__(self, main_channels):
        super().__init__()
        self.gaussian_conditional = GaussianEntropy()

        self.context_prediction_1 = MultistageMaskedConv2d(
            main_channels, main_channels * 2, kernel_size=3, padding=1, stride=1, mask_type='A'
        )
        self.context_prediction_2 = MultistageMaskedConv2d(
            main_channels, main_channels * 2, kernel_size=3, padding=1, stride=1, mask_type='B'
        )
        self.context_prediction_3 = MultistageMaskedConv2d(
            main_channels, main_channels * 2, kernel_size=3, padding=1, stride=1, mask_type='C'
        )

        self.entropy_parameters = nn.Sequential(
            nn.Conv2d(main_channels * 24 // 3, main_channels * 18 // 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(main_channels * 18 // 3, main_channels * 12 // 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(main_channels * 12 // 3, main_channels * 6 // 3, 1, 1),
        )

    def forward(self, x, x_prob):
        x_1 = x.clone()
        x_1[:, :, 0::2, 1::2] = 0
        x_1[:, :, 1::2, :] = 0
        ctx_params_1 = self.context_prediction_1(x_1)
        ctx_params_1[:, :, 0::2, :] = 0
        ctx_params_1[:, :, 1::2, 0::2] = 0

        x_2 = x.clone()
        x_2[:, :, 0::2, 1::2] = 0
        x_2[:, :, 1::2, 0::2] = 0
        ctx_params_2 = self.context_prediction_2(x_2)
        ctx_params_2[:, :, 0::2, 0::2] = 0
        ctx_params_2[:, :, 1::2, :] = 0

        x_3 = x.clone()
        x_3[:, :, 1::2, 0::2] = 0
        ctx_params_3 = self.context_prediction_3(x_3)
        ctx_params_3[:, :, 0::2, :] = 0
        ctx_params_3[:, :, 1::2, 1::2] = 0

        prob = self.entropy_parameters(
            torch.cat((x_prob, ctx_params_1, ctx_params_2, ctx_params_3), dim=1)
        )
        likelihoods = self.gaussian_conditional(x, prob)
        return likelihoods
