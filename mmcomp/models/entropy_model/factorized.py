import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as f
from torch.nn.parameter import Parameter

from ..builder import ENTROPY
from ..utils import Low_bound


@ENTROPY.register_module()
class Factorized_Entropy(nn.Module):
    def __init__(self, hyper_channels, init_scale=10, filters=(3, 3, 3), likelihood_bound=1e-6,
                 tail_mass=1e-9, optimize_integer_offset=True):
        super(Factorized_Entropy, self).__init__()

        self.filters = tuple(int(t) for t in filters)
        self.init_scale = float(init_scale)
        self.likelihood_bound = float(likelihood_bound)
        self.tail_mass = float(tail_mass)

        self.optimize_integer_offset = bool(optimize_integer_offset)

        if not 0 < self.tail_mass < 1:
            raise ValueError(
                "`tail_mass` must be between 0 and 1")
        filters = (1,) + self.filters + (1,)
        scale = self.init_scale ** (1.0 / (len(self.filters) + 1))
        for i in range(len(self.filters) + 1):

            init = np.log(np.expm1(1.0 / scale / filters[i + 1]))
            matrix = Parameter(torch.FloatTensor(hyper_channels, filters[i + 1], filters[i]), requires_grad=True)
            matrix.data.fill_(init)
            self.register_parameter("matrix_{}".format(i), matrix)

            bias = Parameter(torch.FloatTensor(hyper_channels, filters[i + 1], 1), requires_grad=True)
            noise = np.random.uniform(-0.5, 0.5, bias.size())
            noise = torch.FloatTensor(noise)
            bias.data.copy_(noise)
            self.register_parameter("bias_{}".format(i), bias)

            if i < len(self.filters):
                factor = Parameter(torch.FloatTensor(hyper_channels, filters[i + 1], 1), requires_grad=True)
                factor.data.fill_(0.)
                self.register_parameter("factor_{}".format(i), factor)

    def _logits_cumulative(self, logits):

        for i in range(len(self.filters) + 1):
            matrix = f.softplus(self.__getattr__("matrix_{}".format(i)))
            logits = torch.matmul(matrix, logits)

            bias = self.__getattr__("bias_{}".format(i))
            logits += bias

            if i < len(self.filters):
                factor = torch.tanh(self.__getattr__("factor_{}".format(i)))
                logits += factor * torch.tanh(logits)
        return logits

    def forward(self, x_quant):
        shape = x_quant.shape
        _x_quant = x_quant.permute(1, 0, 2, 3).reshape([shape[1], 1, -1])

        lower = self._logits_cumulative(_x_quant - 0.5)
        upper = self._logits_cumulative(_x_quant + 0.5)

        sign = -torch.sign(torch.add(lower, upper))
        sign = sign.detach()
        likelihood = torch.abs(torch.sigmoid(sign * upper) - torch.sigmoid(sign * lower))

        if self.likelihood_bound > 0:
            likelihood = Low_bound.apply(likelihood)

        likelihood = likelihood.reshape([shape[1], shape[0], shape[2], shape[3]]).permute(1, 0, 2, 3)
        return likelihood
