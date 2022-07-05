import io
import json

import cv2
import numpy as np
import rawpy
from imageio import imread


def load_raw(filename):
    if filename.endswith('TIF') or filename.endswith('png'):
        return imread(filename)
    else:
        with rawpy.imread(filename) as f:
            raw = f.raw_image_visible.copy()
        return raw


def load_metainfo(meta_path):
    with open(meta_path, 'r') as f:
        meta = json.load(f)
    return meta


def rearrange(raw, bayer='rggb'):
    assert len(raw.shape) == 2
    h, w = raw.shape
    rearrange_raw = np.zeros([h // 2, w // 2, 4], dtype=raw.dtype)
    if bayer == 'rggb':
        raw = raw
    else:
        return NotImplementedError
    rearrange_raw[..., 0] = raw[0::2, 0::2]
    rearrange_raw[..., 1] = raw[0::2, 1::2]
    rearrange_raw[..., 2] = raw[1::2, 0::2]
    rearrange_raw[..., 3] = raw[1::2, 1::2]
    return rearrange_raw


def inverse_rearrange(rearrange_raw, bayer='rggb'):
    assert rearrange_raw.shape[2] == 4
    h, w = rearrange_raw.shape[:2]
    raw = np.zeros([h * 2, w * 2], dtype=rearrange_raw.dtype)
    if bayer == 'rggb':
        rearrange_raw = rearrange_raw
    else:
        return NotImplementedError
    raw[0::2, 0::2] = rearrange_raw[..., 0]
    raw[0::2, 1::2] = rearrange_raw[..., 1]
    raw[1::2, 0::2] = rearrange_raw[..., 2]
    raw[1::2, 1::2] = rearrange_raw[..., 3]
    return raw


def linearization(raw, black_level, white_level):
    raw = (raw.astype(np.float32) - black_level) / white_level
    return np.clip(raw, 0, 1)


def gamma(linear):
    assert linear.dtype == np.float32
    assert linear.max() <= 1 and linear.min() >= 0, f'max: {linear.max()}, min: {linear.min()}'
    # srgb = np.where(linear <= 0.0031308, 12.92 * linear, 1.055 * np.power(linear, 1 / 2.4) - 0.055)
    # return srgb
    return linear


def inv_gamma(srgb):
    assert srgb.dtype == np.float32
    assert srgb.max() <= 1 and srgb.min() >= 0, f'max: {srgb.max()}, min: {srgb.min()}'
    # linear = np.where(srgb > 0.04045, ((srgb + 0.055) / 1.055) ** 2.4, srgb / 12.92)
    # return linear
    return srgb


def _rgb_to_yuv(rgb):
    '''
    convert gamma rgb to yuv according to b.t.709
    :param rgb [0, 1] (transition should be placed after gamma correction)
    :return: yuv [0, 1] (b.t.709)
    '''
    assert rgb.shape[2] == 3 and rgb.dtype == np.float32
    assert rgb.max() <= 1 and rgb.min() >= 0
    rgb_to_yuv = np.array([
        [0.2126, 0.7125, 0.0722],
        [-0.09991, -0.33609, 0.436],
        [0.615, -0.55861, -0.05639]
    ], dtype=np.float32)
    yuv = np.dot(rgb, rgb_to_yuv.T)
    return np.clip(yuv, 0, 1)


def _yuv_to_rgb(yuv):
    '''
    convert yuv to gamma rgb according to b.t.709
    :param yuv [0, 1] (b.t.709)
    :return: rgb [0, 1] (gamma corrected)
    '''
    assert yuv.shape[2] == 3 and yuv.dtype == np.float32
    assert yuv.max() <= 1 and yuv.min() >= 0
    yuv_to_rgb = np.array([
        [1, 0, 1.28033],
        [1, -0.21482, -0.38509],
        [1, 2.12798, 0]
    ], dtype=np.float32)
    rgb = np.dot(yuv, yuv_to_rgb.T)
    return np.clip(rgb, 0, 1)


def rgb_to_yuv420(rgb):
    assert rgb.dtype == np.float32
    assert rgb.shape[0] % 2 == 0 and rgb.shape[1] % 2 == 0, 'yuv.shape should be even'
    yuv = _rgb_to_yuv(rgb)
    y, u, v = cv2.split(yuv)
    # borrow from
    # https://stackoverflow.com/questions/60729170/python-opencv-converting-planar-yuv-420-image-to-rgb-yuv-array-format
    # may be use other down sample method
    u = cv2.resize(u, (u.shape[1] // 2, u.shape[0] // 2))
    v = cv2.resize(v, (v.shape[1] // 2, v.shape[0] // 2))
    return (y, u, v)


def yuv420_to_rgb(yuv):
    assert yuv[0].dtype == yuv[1].dtype == yuv[2].dtype == np.float32
    y, u, v = yuv
    assert u.shape[0] * 2 == v.shape[0] * 2 == y.shape[0] and u.shape[1] * 2 == v.shape[1] * 2 == y.shape[1]
    u = cv2.resize(u, (u.shape[1] * 2, u.shape[0] * 2))
    v = cv2.resize(v, (v.shape[1] * 2, v.shape[0] * 2))
    yuv = np.concatenate([y[..., None], u[..., None], v[..., None]], axis=-1)
    return _yuv_to_rgb(yuv)


def save_yuv420(yuv, path):
    y, u, v = yuv
    assert y.dtype == u.dtype == v.dtype and y.dtype in [np.uint8, np.uint16]
    assert u.shape[0] * 2 == v.shape[0] * 2 == y.shape[0] and u.shape[1] * 2 == v.shape[1] * 2 == y.shape[1]
    bitstream = io.BytesIO()
    bitstream.write(y.tobytes())
    bitstream.write(u.tobytes())
    bitstream.write(v.tobytes())
    with open(path, 'wb+') as f:
        f.write(bitstream.getbuffer())


def load_yuv420(height, width, depth, path):
    assert depth in [8, 16]
    y_size = height * width * depth // 8
    with open(path, 'rb') as f:
        y = f.read(y_size)
        u = f.read(y_size // 4)
        v = f.read(y_size // 4)
    dtype = np.uint8 if depth == 8 else np.uint16
    y = np.frombuffer(y, dtype).reshape(height, width)
    u = np.frombuffer(u, dtype).reshape(height // 2, width // 2)
    v = np.frombuffer(v, dtype).reshape(height // 2, width // 2)
    return (y, u, v)


def save_yuv400(yuv, path):
    y = yuv[0]
    with open(path, 'wb+') as f:
        f.write(y.tobytes())


def load_yuv400(height, width, depth, path):
    assert depth in [8, 16]
    y_size = height * width * depth // 2
    with open(path, 'rb') as f:
        y = f.read(y_size)
    dtype = np.uint8 if depth == 8 else np.uint16
    y = np.frombuffer(y, dtype).reshape(height, width)
    return (y, None, None)
