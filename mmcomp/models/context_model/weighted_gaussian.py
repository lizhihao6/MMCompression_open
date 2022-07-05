import torch
import torch.nn as nn
import torch.nn.functional as f

from ..builder import CONTEXT
from ..utils import Low_bound


class MaskConv3d(nn.Conv3d):
    def __init__(self, mask_type, in_ch, out_ch, kernel_size, stride, padding):
        super(MaskConv3d, self).__init__(in_ch, out_ch,
                                         kernel_size, stride, padding, bias=True)

        self.mask_type = mask_type
        ch_out, ch_in, k, k, k = self.weight.size()
        mask = torch.zeros(ch_out, ch_in, k, k, k)
        central_id = k * k * k // 2 + 1
        current_id = 1
        if mask_type == 'A':
            for i in range(k):
                for j in range(k):
                    for t in range(k):
                        if current_id < central_id:
                            mask[:, :, i, j, t] = 1
                        else:
                            mask[:, :, i, j, t] = 0
                        current_id = current_id + 1
        if mask_type == 'B':
            for i in range(k):
                for j in range(k):
                    for t in range(k):
                        if current_id <= central_id:
                            mask[:, :, i, j, t] = 1
                        else:
                            mask[:, :, i, j, t] = 0
                        current_id = current_id + 1
        self.register_buffer('mask', mask)

    def forward(self, x):
        self.weight.data *= self.mask
        return super(MaskConv3d, self).forward(x)


class GaussianEntropy(nn.Module):
    def __init__(self):
        super().__init__()

    def _process_scale(self, scale):
        scale = torch.clamp(torch.abs(scale), min=1e-6)
        return scale

    def forward(self, x, prob):
        c = prob.shape[1]
        mean = prob[:, :c // 2, :, :]
        scale = prob[:, c // 2:, :, :]
        # to make the scale always positive
        scale = self._process_scale(scale)
        m1 = torch.distributions.normal.Normal(mean, scale)

        lower = m1.cdf(x - 0.5)
        upper = m1.cdf(x + 0.5)

        # sign = -torch.sign(torch.add(lower, upper))
        # sign = sign.detach()
        # likelihood = torch.abs(f.sigmoid(sign * upper) - f.sigmoid(sign * lower))
        likelihood = torch.abs(upper - lower)

        likelihood = Low_bound.apply(likelihood)
        return likelihood


class MultiGaussianEntropy(nn.Module):
    def __init__(self):
        super().__init__()

    def _process_scale(self, scale):
        scale = torch.clamp(torch.abs(scale), min=1e-6)
        return scale.squeeze(1)

    def forward(self, x, prob):
        # you can use use 3 gaussian
        prob0, mean0, scale0, prob1, mean1, scale1, prob2, mean2, scale2 = torch.chunk(prob, 9, dim=1)
        # keep the weight  summation of prob == 1
        probs = torch.stack((prob0, prob1, prob2), dim=-1).squeeze(1)
        probs = f.softmax(probs, dim=-1)
        # process the scale value to non-zero
        scale0 = self._process_scale(scale0)
        scale1 = self._process_scale(scale1)
        scale2 = self._process_scale(scale2)
        # 3 gaussian distribution
        m0 = torch.distributions.normal.Normal(mean0.squeeze(1), scale0)
        m1 = torch.distributions.normal.Normal(mean1.squeeze(1), scale1)
        m2 = torch.distributions.normal.Normal(mean2.squeeze(1), scale2)

        likelihood0 = torch.abs(m0.cdf(x + 0.5) - m0.cdf(x - 0.5))
        likelihood1 = torch.abs(m1.cdf(x + 0.5) - m1.cdf(x - 0.5))
        likelihood2 = torch.abs(m2.cdf(x + 0.5) - m2.cdf(x - 0.5))

        likelihoods = Low_bound.apply(
            probs[:, :, :, :, 0] * likelihood0 + probs[:, :, :, :, 1] * likelihood1 + probs[:, :, :, :,
                                                                                      2] * likelihood2)

        return likelihoods


class ScaleMultiGaussianEntropy(nn.Module):
    def __init__(self):
        super().__init__()

    def _process_scale(self, scale):
        scale = torch.clamp(torch.abs(scale), min=1e-6)
        return scale.squeeze(1)

    def forward(self, x, prob, factor):
        assert factor.shape == x.shape
        # you can use use 3 gaussian
        prob0, mean0, scale0, prob1, mean1, scale1, prob2, mean2, scale2 = torch.chunk(prob, 9, dim=1)
        # keep the weight  summation of prob == 1
        probs = torch.stack((prob0, prob1, prob2), dim=-1).squeeze(1)
        probs = f.softmax(probs, dim=-1)
        # process the scale value to non-zero
        scale0 = self._process_scale(scale0)
        scale1 = self._process_scale(scale1)
        scale2 = self._process_scale(scale2)
        # 3 gaussian distribution
        m0 = torch.distributions.normal.Normal(mean0.squeeze(1), scale0)
        m1 = torch.distributions.normal.Normal(mean1.squeeze(1), scale1)
        m2 = torch.distributions.normal.Normal(mean2.squeeze(1), scale2)

        likelihood0 = torch.abs(m0.cdf(x + 0.5 / factor) - m0.cdf(x - 0.5 / factor))
        likelihood1 = torch.abs(m1.cdf(x + 0.5 / factor) - m1.cdf(x - 0.5 / factor))
        likelihood2 = torch.abs(m2.cdf(x + 0.5 / factor) - m2.cdf(x - 0.5 / factor))

        likelihoods = Low_bound.apply(
            probs[:, :, :, :, 0] * likelihood0 + probs[:, :, :, :, 1] * likelihood1 + probs[:, :, :, :,
                                                                                      2] * likelihood2)

        return likelihoods


@CONTEXT.register_module()
class Weighted_Gaussian(nn.Module):
    def __init__(self, main_channels=192):
        super().__init__()
        self.conv1 = MaskConv3d('A', 1, 24, 11, 1, 5)
        self.conv2 = nn.Sequential(nn.Conv3d(25, 48, 1, 1, 0), nn.ReLU(inplace=False), nn.Conv3d(48, 96, 1, 1, 0),
                                   nn.ReLU(inplace=False),
                                   nn.Conv3d(96, 9, 1, 1, 0))
        self.conv3 = nn.Conv2d(main_channels * 2, main_channels, 3, 1, 1)

        self.gaussin_entropy_func = MultiGaussianEntropy()

    def forward(self, x, x_prob):
        x1 = self.conv1(torch.unsqueeze(x, dim=1))
        hyper = torch.unsqueeze(self.conv3(x_prob), dim=1)
        prob = self.conv2(torch.cat((x1, hyper), dim=1))
        likelihoods = self.gaussin_entropy_func(x, prob)
        return likelihoods


@CONTEXT.register_module()
class Scale_Weighted_Gaussian(Weighted_Gaussian):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.gaussin_entropy_func = ScaleMultiGaussianEntropy()

    def forward(self, x, x_prob, factor):
        x1 = self.conv1(torch.unsqueeze(x, dim=1))
        hyper = torch.unsqueeze(self.conv3(x_prob), dim=1)
        prob = self.conv2(torch.cat((x1, hyper), dim=1))
        likelihoods = self.gaussin_entropy_func(x, prob, factor)
        return likelihoods
