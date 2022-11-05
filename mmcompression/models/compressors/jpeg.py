# Copyright (c) NJU Vision Lab. All rights reserved.
import time

import cv2
import numpy as np
import torch

from .base import BaseCompressor
from ..builder import COMPRESSOR


@COMPRESSOR.register_module()
class JPEGCompressor(BaseCompressor):
    """JPEG Compression"""

    def __init__(self, qp: int):
        """
        Args:
            qp (int): quality parameter, range [1, 100], 100 is the best quality.
        """
        super().__init__()
        self.qp = qp

    def compress(self, img: np.ndarray):
        """
        Args:
            img (np.ndarray): The original images of shape (C, H, W).
                Typically, these should be scaled into zero to one within RGB format.
        Returns:
            bitstreams (dict):  The compressed bitstreams.
        """
        # convert RGB to BGR
        img = img[..., ::-1]
        img = np.round(img * 255.).astype(np.float32)
        params = [cv2.IMWRITE_JPEG_QUALITY, self.qp]
        msg = cv2.imencode(".jpg", img, params)[1]
        msg = (np.array(msg)).tobytes()
        return dict(msg=msg)

    def decompress(self, bitstreams: dict):
        """
        Args:
            bitstreams (dict): The compressed bitstreams.
        Returns:
            img (np.ndarray): The recovered image of shape (C, H, W).
        """
        msg = bitstreams["msg"]
        img = cv2.imdecode(np.frombuffer(msg, np.uint8), cv2.IMREAD_COLOR)
        # convert BGR to RGB
        img = img[..., ::-1]
        img = img.astype(np.float32) / 255.
        return img

    def forward_train(self, img: torch.FloatTensor, img_metas: list, **kwargs):
        """
        Args:
            img (Tensor): The original images of shape (N, C, H, W).
                Typically, these should be scaled into zero to one within RGB format.
            img_metas (list[dict]): List of image info dict where each dict
                has: 'filename' and may also contain other keys.
                For details on the values of these keys, see
                :class:`mmcompress.datasets.pipelines.Collect`.
            kwargs (keyword arguments): Specific to concrete implementation.
        """
        raise NotImplementedError("JPEG is a handcrafted compression method, it is not suitable for training.")

    def forward_test(self, img: torch.FloatTensor, img_metas: list, return_image: bool = False, **kwargs):
        """
        Args:
            img (Tensor): The original images of shape (N, C, H, W).
                Typically, these should be scaled into zero to one within RGB format.
            img_metas (list[dict]): List of image info dict where each dict
                has: 'filename' and may also contain other keys.
                For details on the values of these keys, see
                :class:`mmcompress.datasets.pipelines.Collect`.
            return_image (bool): whether to return the image after compress-decompress.
            kwargs (keyword arguments): Specific to concrete implementation.
        """
        results = []
        for img, img_meta in zip(img, img_metas):
            # compress
            _img = img.transpose([1, 2, 0]).detach().cpu().numpy()
            start = time.time()
            bitstreams = self.compress(_img)
            end = time.time()
            compress_time = end - start

            # decompress
            start = time.time()
            _rec_img = self.decompress(bitstreams)
            end = time.time()
            decompress_time = end - start
            rec_img = torch.from_numpy(_rec_img.transpose([2, 0, 1])).to(img.device)

            # calculate bpp
            bits_num = self.calculate_bits_num(bitstreams)
            bpp = bits_num / img.shape[1] / img.shape[2]

            # calculate psnr
            psnr = self.calculate_psnr(img, rec_img)

            result = dict(
                rec_img=None if not return_image else self.save_img(rec_img),
                bpp=bpp,
                psnr=psnr,
                bits_num=bits_num,
                compress_time=compress_time,
                decompress_time=decompress_time)
            results.append(result)
        return results
