import argparse

import numpy as np
import torch
import torchac
from imageio import imread

parser = argparse.ArgumentParser(description='Lossless encoding useing arithmetic encoding')
parser.add_argument('-i', '--img', type=str, required=True, help='near-lossless compressed image')
parser.add_argument('-g', '--gt', type=str, required=True, help='gt image')
args = parser.parse_args()

if __name__ == '__main__':
    frequency_table = {}
    im, gt = imread(args.img).astype(np.int16), imread(args.gt).astype(np.int16)
    h, w = im.shape[:2]
    residual = (im - gt)
    if len(residual.shape) == 2:
        residual = residual[..., None]

    bpp = 0.
    for c in range(residual.shape[-1]):
        message = residual[..., c]
        h, w = message.shape

        message -= message.min()
        max_v = message.max()
        hist, _ = np.histogram(message, bins=[i for i in range(-1, max_v + 2)])
        prob = hist.astype(np.float32) / hist.sum()
        cdf = np.cumsum(prob)

        message = torch.from_numpy(message)[None, None]  # [1, 1, h, w]
        cdf = torch.from_numpy(cdf)[None, None, None, None].repeat(1, 1, h, w, 1)  # [1, 1, h, w, max_v+1]

        byte_stream = torchac.encode_float_cdf(cdf, message, check_input_bounds=True)
        real_bits = len(byte_stream) * 8
        bpp += real_bits / h / w

    print('bpp = {}'.format(bpp))
