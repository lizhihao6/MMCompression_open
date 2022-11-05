import torch
from torch import nn


class MaskConv3d(nn.Conv3d):
    """
    Mask 3d-convolutional layer.
    Args:
        mask_type (str): 'A' or 'B'
        in_ch (int): input channel
        out_ch (int): output channel
        kernel_size (int): kernel size
        stride (int): stride
        padding (int): padding
    """

    def __init__(self, mask_type, in_ch, out_ch, kernel_size, stride, padding):
        super().__init__(in_ch, out_ch, kernel_size, stride, padding, bias=True)

        self.mask_type = mask_type
        ch_out, ch_in, k1, k2, k3 = self.weight.size()
        assert k1 == k2 == k3, 'kernel size must be equal'
        k = k1
        mask = torch.zeros(ch_out, ch_in, k, k, k)
        central_id = k * k * k // 2 + 1
        current_id = 1
        if mask_type == "A":
            for i in range(k):
                for j in range(k):
                    for t in range(k):
                        if current_id < central_id:
                            mask[:, :, i, j, t] = 1
                        else:
                            mask[:, :, i, j, t] = 0
                        current_id = current_id + 1
        if mask_type == "B":
            for i in range(k):
                for j in range(k):
                    for t in range(k):
                        if current_id <= central_id:
                            mask[:, :, i, j, t] = 1
                        else:
                            mask[:, :, i, j, t] = 0
                        current_id = current_id + 1
        self.register_buffer("mask", mask)

    def forward(self, input):
        """
        Args:
            input (torch.Tensor): the input tensor.
        Returns:
            torch.Tensor: the output tensor.
        """
        self.weight.data *= self.mask
        return super().forward(input)
