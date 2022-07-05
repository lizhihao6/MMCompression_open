import os
import shutil
import tempfile

import cv2
import numpy as np

from mmcomp.utils import tensor_to_image, tensor_to_raw, inverse_rearrange
from .base import BaseCompressor
from ..builder import COMPRESSOR


@COMPRESSOR.register_module()
class JPEGCompressor(BaseCompressor):
    """Our Neural Image Compression
    """

    def __init__(self,
                 test_cfg=None,
                 **kwargs):
        super().__init__()
        self.qp = test_cfg['qp']
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

    def _comp_rgb(self, rgb, quality):
        """
        use jpeg to compression rgb
        :param rgb: np.array [0, 1]
        :param quality: int [1, 100]
        :return: reconstruct rgb, bits
        """
        assert rgb.max() <= 1 and rgb.min() >= 0, f'rgb max: {rgb.max()}, rgb min: {rgb.min}'
        # compression
        assert rgb.shape[2] == 3, f'rgb should have 3 channels, instead of {rgb.shape[2]}'
        rgb = (rgb.copy() * 255).astype(np.uint8)[..., ::-1]
        jpg_file = self._get_tmp_file('jpg')
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        cv2.imwrite(jpg_file, rgb, params=encode_param)
        re_rgb = cv2.imread(jpg_file).astype(np.float32)[..., ::-1] / 255
        bits = float(os.path.getsize(jpg_file)) * 8
        return re_rgb, bits

    def _comp_raw(self, raw, quality):
        """
        use jpeg to compression r, gb, b, and use other jpeg to compression gr
        :param raw: np.array [0, 1] (shape [h, w, 4])
        :param quality: int [1, 51]
        :return: reconstruct rgb, bits
        """
        # assert self.mode == 444, 'raw compression only support 444'
        assert raw.max() <= 1 and raw.min() >= 0, f'raw max: {raw.max()}, raw min: {raw.min}'
        raw = np.where(raw <= 0.0031308, 12.92 * raw, 1.055 * np.power(raw, 1 / 2.4) - 0.055)
        r, gb, gr, b = np.split(raw, 4, axis=-1)
        # use jpeg to comp rgb
        rgb = np.concatenate([r, gb, b], axis=-1)
        re_rgb, bits = self._comp_rgb(rgb, quality)
        # use jpeg comp g
        gr -= gb
        g_min = gr.min()
        gr -= g_min
        gr = np.clip(gr, 0, 1) * 255
        gr = gr.astype(np.uint8)
        jpg_file = self._get_tmp_file('jpg')
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        cv2.imwrite(jpg_file, gr, params=encode_param)
        re_g = cv2.imread(jpg_file, cv2.IMREAD_UNCHANGED).astype(np.float32) / 255
        bits += float(os.path.getsize(jpg_file)) * 8
        re_g += g_min
        re_g += re_rgb[..., 1]
        re_raw = np.zeros_like(raw)
        re_raw[..., (0, 1, 3)] = re_rgb
        re_raw[..., 2] = re_g
        re_raw = np.clip(re_raw, 0, 1)
        re_raw = np.where(re_raw > 0.04045, ((re_raw + 0.055) / 1.055) ** 2.4, re_raw / 12.92)
        return re_raw, bits

    def forward_train(self, img, **kwargs):
        raise NotImplementedError

    def forward_test(self, img, return_image, **kwargs):
        """Simple test with single image."""
        assert img.shape[0] == 1
        _img = img.detach().clone()
        img = img[0].permute([1, 2, 0]).detach().cpu().numpy()
        results = {}
        comp_fn = self._comp_rgb if img.shape[2] == 3 else self._comp_raw
        re_img, bits = comp_fn(img, self.qp)
        psnr = 10 * np.log10(1. / np.power(img - re_img, 2).mean())
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
