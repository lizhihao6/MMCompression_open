# Copyright (c) NJU Vision Lab. All rights reserved.
import torch
import torch.nn.functional as f
from torch import nn

from ..builder import CONTEXT
from ..utils import Low_bound, MaskConv3d


def _process_scale(scale):
    scale = torch.clamp(scale, min=1e-6)
    return scale


class GaussianEntropy(nn.Module):
    """
    Gaussian Entropy model.
    Estimate the entropy of the Gaussian distribution.
    """

    def forward(self, x, prob):
        """
        Args:
             x (torch.Tensor): To be coded tensor.
            prob (torch.Tensor): Probability of the Gaussian distribution.
        Returns:
            torch.Tensor: Entropy of the Gaussian distribution.
        """
        c = prob.shape[1]
        mean = prob[:, : c // 2, :, :]
        scale = prob[:, c // 2:, :, :]
        # to make the scale always positive
        scale = _process_scale(scale)
        m1 = torch.distributions.normal.Normal(mean, scale)

        lower = m1.cdf(x - 0.5)
        upper = m1.cdf(x + 0.5)
        likelihood = torch.abs(upper - lower)
        likelihood = Low_bound.apply(likelihood)
        return likelihood


class MultiGaussianEntropy(nn.Module):
    """
    Mixed (Three) Gaussian Entropy model.
    Estimate the entropy of the Mixed Gaussian distribution.
    """

    def forward(self, x, prob):
        """
        Args:
             x (torch.Tensor): To be coded tensor.
            prob (torch.Tensor): Probability of the Gaussian distribution.
        Returns:
            torch.Tensor: Entropy of the Gaussian distribution.
        """
        # you can use use 3 gaussian
        prob0, mean0, scale0, prob1, mean1, scale1, prob2, mean2, scale2 = torch.chunk(
            prob, 9, dim=1
        )
        # keep the weight  summation of prob == 1
        probs = torch.stack((prob0, prob1, prob2), dim=-1).squeeze(1)
        probs = f.softmax(probs, dim=-1)
        # process the scale value to non-zero
        scale0 = _process_scale(scale0).squeeze(1)
        scale1 = _process_scale(scale1).squeeze(1)
        scale2 = _process_scale(scale2).squeeze(1)
        # 3 gaussian distribution
        m0 = torch.distributions.normal.Normal(mean0.squeeze(1), scale0)
        m1 = torch.distributions.normal.Normal(mean1.squeeze(1), scale1)
        m2 = torch.distributions.normal.Normal(mean2.squeeze(1), scale2)

        likelihood0 = torch.abs(m0.cdf(x + 0.5) - m0.cdf(x - 0.5))
        likelihood1 = torch.abs(m1.cdf(x + 0.5) - m1.cdf(x - 0.5))
        likelihood2 = torch.abs(m2.cdf(x + 0.5) - m2.cdf(x - 0.5))

        likelihoods = Low_bound.apply(
            probs[:, :, :, :, 0] * likelihood0
            + probs[:, :, :, :, 1] * likelihood1
            + probs[:, :, :, :, 2] * likelihood2
        )

        return likelihoods


@CONTEXT.register_module()
class WeightedGaussian(nn.Module):
    """
    Context Model:
    Model the mean and variance of the Gaussian distribution using the x and hyper-prior (x_prob).
    Args:
        main_channels (int): The number of channels of the input tensor x.
    """

    def __init__(self, main_channels=192):
        super().__init__()
        self.conv1 = MaskConv3d("A", 1, 24, 11, 1, 5)
        self.conv2 = nn.Sequential(
            nn.Conv3d(25, 48, 1, 1, 0),
            nn.ReLU(inplace=False),
            nn.Conv3d(48, 96, 1, 1, 0),
            nn.ReLU(inplace=False),
            nn.Conv3d(96, 9, 1, 1, 0),
        )
        self.conv3 = nn.Conv2d(main_channels * 2, main_channels, 3, 1, 1)

        self.gaussin_entropy_func = MultiGaussianEntropy()

    def forward(self, x, x_prob):
        x1 = self.conv1(torch.unsqueeze(x, dim=1))
        hyper = torch.unsqueeze(self.conv3(x_prob), dim=1)
        prob = self.conv2(torch.cat((x1, hyper), dim=1))
        likelihoods = self.gaussin_entropy_func(x, prob)
        return likelihoods
