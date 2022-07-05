import faulthandler;
import os
import shutil
import tempfile

from pidng.core import RAW2DNG, DNGTags, Tag
from pidng.defs import *

faulthandler.enable()

import cv2
import numpy as np

from mmcomp.utils import tensor_to_image, tensor_to_raw, inverse_rearrange
from .base import BaseCompressor
from ..builder import COMPRESSOR


@COMPRESSOR.register_module()
class DNGCompressor(BaseCompressor):
    """Our Neural Image Compression
    """

    def __init__(self,
                 depth=None,
                 test_cfg=None,
                 **kwargs):
        super().__init__()
        self.depth = depth
        self.cache_files = []
        self.converter = RAW2DNG()

    def _get_tags(self, height, width):
        # uncalibrated color matrix, just for demo.
        ccm1 = [[19549, 10000], [-7877, 10000], [-2582, 10000],
                [-5724, 10000], [10121, 10000], [1917, 10000],
                [-1267, 10000], [-110, 10000], [6621, 10000]]

        # set DNG tags.
        t = DNGTags()
        t.set(Tag.ImageWidth, width)
        t.set(Tag.ImageLength, height)
        t.set(Tag.TileWidth, width)
        t.set(Tag.TileLength, height)
        t.set(Tag.Orientation, Orientation.Horizontal)
        t.set(Tag.PhotometricInterpretation, PhotometricInterpretation.Color_Filter_Array)
        t.set(Tag.SamplesPerPixel, 1)
        t.set(Tag.BitsPerSample, self.depth)
        t.set(Tag.CFARepeatPatternDim, [2, 2])
        t.set(Tag.CFAPattern, CFAPattern.GBRG)
        t.set(Tag.BlackLevel, (4096 >> (16 - self.depth)))
        t.set(Tag.WhiteLevel, ((1 << self.depth) - 1))
        t.set(Tag.ColorMatrix1, ccm1)
        t.set(Tag.CalibrationIlluminant1, CalibrationIlluminant.D65)
        t.set(Tag.AsShotNeutral, [[1, 1], [1, 1], [1, 1]])
        t.set(Tag.BaselineExposure, [[-150, 100]])
        t.set(Tag.Make, "Camera Brand")
        t.set(Tag.Model, "Camera Model")
        t.set(Tag.DNGVersion, DNGVersion.V1_4)
        t.set(Tag.DNGBackwardVersion, DNGVersion.V1_2)
        t.set(Tag.PreviewColorSpace, PreviewColorSpace.sRGB)
        return t

    def _get_tmp_file(self, suffix=None):
        tmp_file = os.path.join(tempfile.gettempdir(), next(tempfile._get_candidate_names()))
        tmp_file = tmp_file + f'_{os.getpid()}' + f'.{suffix}'
        self.cache_files.append(tmp_file)
        return tmp_file

    def __del__(self):
        for p in self.cache_files:
            try:
                os.remove(p)
            except:
                continue
        self.cache_files.clear()

    def _quant(self, img):
        assert img.max() <= 1 and img.min() >= 0, f'img max: {img.max()}, img min: {img.min()}'
        img = img.copy()
        img *= (2 ** self.depth - 1)
        dtype = np.uint8 if self.depth == 8 else np.uint16
        img = np.round(img).astype(dtype)
        return img

    def _comp(self, img):
        """
        use flif to compression img
        :param img: np.array [0, 1]
        :return: reconstruct img, bits
        """
        assert img.max() <= 1 and img.min() >= 0, f'img max: {img.max()}, img min: {img.min}'
        ori_img = img.copy()
        # compression
        assert img.shape[2] == 4, 'DNG compressor only support RAW comp'  # only support raw
        img = inverse_rearrange(img)
        img = self._quant(img)
        img = img >> (16 - self.depth)

        tags = self._get_tags(height=img.shape[0], width=img.shape[1])
        self.converter.options(tags, path="", compress=True)
        dng = self._get_tmp_file('')
        self.converter.convert(img, filename=dng)
        bits = float(os.path.getsize(dng + '.dng')) * 8
        self.cache_files.remove(dng)
        self.cache_files.append(dng + '.dng')
        return ori_img, bits

    def forward_train(self, img, **kwargs):
        raise NotImplementedError

    def forward_test(self, img, return_image, **kwargs):
        """Simple test with single image."""
        assert img.shape[0] == 1
        _img = img.detach().clone()
        img = img[0].permute([1, 2, 0]).detach().cpu().numpy()
        results = {}
        re_img, bits = self._comp(img)
        psnr = np.nan
        bpp = bits / img.shape[0] / img.shape[1]
        if img.shape[-1] == 4:
            bpp /= 4
        results[f'psnr'] = float(psnr)
        results[f'bpp'] = float(bpp)
        if return_image:
            if _img.shape[1] == 3:
                ori_img = tensor_to_image(_img)[..., ::-1]
                rec_img = np.round(re_img * 255).astype(np.uint8)[..., ::-1]
            else:
                img_metas = kwargs['img_metas']
                blc, saturate = img_metas[0]['black_level'], img_metas[0]['white_level']
                ori_img = tensor_to_raw(_img, blc, saturate)
                rec_img = inverse_rearrange(re_img)
                rec_img = np.clip(rec_img, 0, 1)
                rec_img = rec_img * (saturate - blc) + blc
                rec_img = rec_img.astype(np.uint16)
            ori_img_path = self._get_tmp_file('png')
            rec_img_path = self._get_tmp_file('png')
            cv2.imwrite(ori_img_path, ori_img)
            cv2.imwrite(rec_img_path, rec_img)
            results['ori_img'] = ori_img_path
            results['rec_img'] = rec_img_path
        del _img, img
        return results

    def show_result(self, data, result, show=False, out_file=None):
        # diffierent from nn based visual, it is designed for save 16bits raw
        if show:
            raise NotImplementedError

        if out_file is None:
            pass
        else:
            suffix = out_file.split('.')[-1]
            assert suffix == 'png'
            shutil.move(result['ori_img'], out_file[:-3] + 'gt.png')
            shutil.move(result['rec_img'], out_file)
