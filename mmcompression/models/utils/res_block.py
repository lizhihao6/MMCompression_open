# Copyright (c) NJU Vision Lab. All rights reserved.
from torch import nn


class ResBlock(nn.Module):
    """
    Basic ResBlock for VAE.
    Args:
        in_channel (int): the number of input channels.
        out_channel (int): the number of output channels.
        kernel_size (int): the size of the kernel.
        stride (int): the stride of the convolution.
        padding (int): the padding of the convolution.
    """

    def __init__(self, in_channel, out_channel, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.in_ch = int(in_channel)
        self.out_ch = int(out_channel)
        self.k = int(kernel_size)
        self.stride = int(stride)
        self.padding = int(padding)
        self.relu = nn.ReLU(inplace=False)

        self.conv1 = nn.Conv2d(self.in_ch, self.in_ch,
                               self.k, self.stride, self.padding)
        self.conv2 = nn.Conv2d(self.in_ch, self.out_ch,
                               self.k, self.stride, self.padding)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): the input tensor.
        Returns:
            torch.Tensor: the output tensor.
        """
        x1 = self.conv2(self.relu(self.conv1(x)))
        out = x + x1
        return out
