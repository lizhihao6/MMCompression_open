import numpy as np
import torch


def tensor_to_image(tensor:torch.FloatTensor):
    """
    Convert a tensor to an image
    Args:
        tensor (Tensor): Tensor image of size (C, H, W)
    Returns:
        numpy.ndarray: Image of shape (H, W, C)
    """
    assert len(tensor.shape) == 3 and tensor.shape[0] == 3
    im = tensor.detach().cpu().numpy().transpose([1, 2, 0])
    im = np.clip(im, 0, 1) * 255.
    return im.astype(np.uint8)


def numpy_to_img(mat):
    """
    Convert a numpy array to an image
    Args:
        mat (numpy.ndarray): numpy array of shape (H, W, C)
    Returns:
        numpy.ndarray: Image of shape (H, W, C)
    """
    assert mat.shape[2] == 3
    return np.round(mat * 255).astype(np.uint8)
