# Copyright (c) NJU Vision Lab. All rights reserved.
import os

import cv2
import numpy as np

from mmcomp.utils import tensor_to_image, numpy_to_img
from .base import BaseCompressor
from ..builder import COMPRESSOR


@COMPRESSOR.register_module()
class JPEGCompressor(BaseCompressor):
    """JPEG Compression"""

    def __init__(self, qp, **kwargs):
        super().__init__()
        self.qp = qp

    def _compression(self, rgb, quality):
        """
        use jpeg to compression rgb
        Args:
            rgb: numpy.ndarray, shape (H, W, 3)
            quality: int [1, 100]
        Returns:
            numpy.ndarray, shape (H, W, 3), compressed image
            float, bits
        """
        assert (
                rgb.max() <= 1 and rgb.min() >= 0
        ), f"rgb max: {rgb.max()}, rgb min: {rgb.min}"
        # compression
        assert (
                rgb.shape[2] == 3
        ), f"rgb should have 3 channels, instead of {rgb.shape[2]}"
        rgb = (rgb.copy() * 255).astype(np.uint8)[..., ::-1]
        jpg_file = self._get_tmp_file("jpg")
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        cv2.imwrite(jpg_file, rgb, params=encode_param)
        re_rgb = cv2.imread(jpg_file).astype(np.float32)[..., ::-1] / 255
        bits = float(os.path.getsize(jpg_file)) * 8
        return re_rgb, bits

    def forward_train(self, img, img_metas, **kwargs):
        raise NotImplementedError

    def forward_test(self, img, img_metas, return_image, **kwargs):
        """Simple test with single image.
        Args:
            img (tensor): a tensor of shape (1, C, H, W)
            img_metas (dict): a dict contains image meta info.
            return_image (bool): whether to return image.
            kwargs (dict): other arguments.
        Returns:
            results (dict): a dict containing compression results.
        """
        assert img.shape[0] == 1
        ori_img = tensor_to_image(img[0]).astype(np.float32) / 255
        results = {}
        rec_img, bits = self._compression(ori_img, self.qp)
        psnr = 10 * np.log10(1.0 / np.power(ori_img - rec_img, 2).mean())
        bpp = bits / img.shape[0] / img.shape[1]
        results["psnr"] = float(psnr)
        results["bpp"] = float(bpp)
        if return_image:
            ori_img = numpy_to_img(ori_img)[..., ::-1]
            rec_img = numpy_to_img(rec_img)[..., ::-1]
            ori_img_path = self._get_tmp_file("png")
            rec_img_path = self._get_tmp_file("png")
            cv2.imwrite(ori_img_path, ori_img)
            cv2.imwrite(rec_img_path, rec_img)
            results["ori_img"] = ori_img_path
            results["rec_img"] = rec_img_path
        return results
