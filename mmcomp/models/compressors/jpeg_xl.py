import os
import shutil
import tempfile

import cv2
import numpy as np

from mmcomp.utils import tensor_to_image, tensor_to_raw, inverse_rearrange
from .base import BaseCompressor
from ..builder import COMPRESSOR


@COMPRESSOR.register_module()
class JPEGXLCompressor(BaseCompressor):
    """Our Neural Image Compression
    """

    def __init__(self,
                 jpeg_xl=None,
                 depth=None,
                 test_cfg=None,
                 **kwargs):
        super().__init__()
        self.jpeg_xl = jpeg_xl
        self.depth = depth
        self.cache_files = []

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
        use jpeg_xl to compression img
        :param img: np.array [0, 1]
        :return: reconstruct img, bits
        """
        assert img.max() <= 1 and img.min() >= 0, f'img max: {img.max()}, img min: {img.min}'
        ori_img = img.copy()
        # compression
        png_file = self._get_tmp_file('png')
        jpeg_xl_file = self._get_tmp_file('jxl')
        if img.shape[2] == 4:  # raw
            rgb = img[..., [0, 1, 3]]
            residual = img[..., 2] - img[..., 1] + 0.5
            if residual.min() < 0 or residual.max() > 0:
                residual = img[..., 2]
            rgb, residual = self._quant(rgb), self._quant(residual)
            png_file = self._get_tmp_file('png')
            cv2.imwrite(png_file, rgb)
            os.system(f'{self.jpeg_xl} -q 100 {png_file} {jpeg_xl_file} > /dev/null 2>&1')
            bits = float(os.path.getsize(jpeg_xl_file)) * 8
            cv2.imwrite(png_file, residual)
            os.system(f'{self.jpeg_xl} -q 100 {png_file} {jpeg_xl_file} > /dev/null 2>&1')
            bits += float(os.path.getsize(jpeg_xl_file)) * 8
        else:
            img = self._quant(img)
            cv2.imwrite(png_file, img)
            os.system(f'{self.jpeg_xl} -q 100 {png_file} {jpeg_xl_file} > /dev/null 2>&1')
            bits = float(os.path.getsize(jpeg_xl_file)) * 8
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
