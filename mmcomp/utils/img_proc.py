import numpy as np


def tensor_to_image(tensor, **kwargs):
    assert len(tensor.shape) == 4 and tensor.shape[1] in [1, 3, 4]
    im = tensor[0].detach().cpu().numpy().transpose([1, 2, 0])
    if im.shape[-1] in [1, 3]:
        im = np.clip(im, 0, 1) * 255.
    else:
        im = inverse_rearrange(im, **kwargs)
        im = gamma(im) * 255.
    return im.astype(np.uint8)


def tensor_to_raw(tensor, blc, saturate, bayer_pattern='rggb'):
    assert len(tensor.shape) == 4 and tensor.shape[1] == 4
    im = tensor[0].detach().cpu().numpy().transpose([1, 2, 0])
    im = inverse_rearrange(im, bayer_pattern)
    im = np.clip(im, 0, 1)
    im = im * (saturate - blc) + blc
    return im.astype(np.uint16)


def rearrange(raw, bayer_pattern='rggb'):
    assert isinstance(raw, np.ndarray) and len(raw.shape) == 2
    h, w = raw.shape
    if bayer_pattern == 'rggb':
        raw = raw
    else:
        raise NotImplementedError
    rearrange_raw = np.zeros([h // 2, w // 2, 4], dtype=raw.dtype)
    rearrange_raw[..., 0] = raw[0::2, 0::2]
    rearrange_raw[..., 1] = raw[0::2, 1::2]
    rearrange_raw[..., 2] = raw[1::2, 0::2]
    rearrange_raw[..., 3] = raw[1::2, 1::2]
    return rearrange_raw


def inverse_rearrange(rearrange_raw, bayer_pattern='rggb'):
    assert isinstance(rearrange_raw, np.ndarray) and len(rearrange_raw.shape) == 3
    h, w, c = rearrange_raw.shape
    assert c == 4
    if bayer_pattern == 'rggb':
        rearrange_raw = rearrange_raw
    else:
        raise NotImplementedError
    raw = np.zeros([int(h * 2), int(w * 2)], dtype=rearrange_raw.dtype)
    raw[0::2, 0::2] = rearrange_raw[..., 0]
    raw[0::2, 1::2] = rearrange_raw[..., 1]
    raw[1::2, 0::2] = rearrange_raw[..., 2]
    raw[1::2, 1::2] = rearrange_raw[..., 3]
    return raw


def gamma(linear):
    linear = np.clip(linear, 0, 1)
    return np.where(linear <= 0.0031308, 12.92 * linear, 1.055 * np.power(linear, 1 / 2.4) - 0.055)


def inv_gamma(srgb):
    srgb = np.clip(srgb, 0, 1)
    return np.where(srgb > 0.04045, ((srgb + 0.055) / 1.055) ** 2.4, srgb / 12.92)


def numpy_to_img(mat):
    if mat.shape[2] == 3:
        return np.round(mat * 255).astype(np.uint8)
    elif mat.shape[2] == 4:
        img = np.where(mat <= 0.0031308, 12.92 * mat, 1.055 * np.power(mat, 1 / 2.4) - 0.055)
        img = np.round(img * 255).astype(np.uint8)
        h, w = mat.shape[:2]
        raw = np.zeros([h * 2, w * 2], dtype=np.float32)
        raw[0::2, 0::2] = img[..., 0]
        raw[0::2, 1::2] = img[..., 1]
        raw[1::2, 0::2] = img[..., 2]
        raw[1::2, 1::2] = img[..., 3]
        return raw[..., None]
    else:
        raise NotImplementedError
