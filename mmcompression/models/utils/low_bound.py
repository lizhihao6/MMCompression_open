# Copyright (c) NJU Vision Lab. All rights reserved.
import numpy as np
import torch


class Low_bound(torch.autograd.Function):
    """
    This class is used to avoid the overflow of the log function.
    """

    @staticmethod
    def forward(ctx, x):
        """
        Args:
            ctx (torch.autograd.Function): the context.
            x (torch.Tensor): the input tensor.
        Returns:
            torch.Tensor: the output tensor.
        """
        ctx.save_for_backward(x)
        x = torch.clamp(x, min=1e-6)
        return x

    @staticmethod
    def backward(ctx, g):
        """
        Args:
            ctx (torch.autograd.Function): the context.
            g (torch.Tensor): the gradient tensor.
        Returns:
            torch.Tensor: the gradient tensor.
        """
        x, = ctx.saved_tensors
        grad1 = g.clone()
        grad1[x < 1e-6] = 0
        pass_through_if = np.logical_or(
            x.cpu().numpy() >= 1e-6, g.cpu().numpy() < 0.0)
        t = torch.Tensor(pass_through_if + 0.0).cuda()
        return grad1 * t
