import os
import shutil
import tempfile

import cv2
import numpy as np

from mmcomp.utils import tensor_to_image, tensor_to_raw, inverse_rearrange
from .base import BaseCompressor
from ..builder import COMPRESSOR


@COMPRESSOR.register_module()
class MPEGCompressor(BaseCompressor):
    """Our Neural Image Compression
    """

    def __init__(self,
                 mpeg=None,
                 mpeg_cfg=None,
                 depth=None,
                 mode=420,
                 test_cfg=None,
                 **kwargs):
        assert mode in [420, 444], 'only support 420, 444 now'
        super().__init__()
        self.depth = depth
        self.mode = mode
        self.qp = test_cfg['qp']
        # todo check output bit depth
        self.command = f'{mpeg} -c {mpeg_cfg} '
        self.command += f'--InputBitDepth={depth} ' \
                        f'--InternalBitDepth={depth} ' \
                        f'--OutputBitDepth={depth} ' \
                        f'--FrameRate=50 ' \
                        f'--FrameSkip=0 ' \
                        f'--FramesToBeEncoded=1 ' \
                        f'--Level=2.1 '
        self.command += '--SourceWidth={w} ' \
                        '--SourceHeight={h} ' \
                        '--InputChromaFormat={mode} ' \
                        '--QP={q} ' \
                        '-i {input} -b {bin} -o {output}'
        self.debug = False
        self.ffmpeg = '/usr/bin/ffmpeg'
        self.cache_files = []

    def _get_tmp_file(self, suffix=None):
        tmp_file = os.path.join(tempfile.gettempdir(), next(tempfile._get_candidate_names()))
        tmp_file = tmp_file + f'_{os.getpid()}' + f'.{suffix}'
        self.cache_files.append(tmp_file)
        return tmp_file

    def __del__(self):
        if self.debug:
            pass
        else:
            for p in self.cache_files:
                try:
                    os.remove(p)
                except:
                    continue
            self.cache_files.clear()

    def _system(self, command):
        if self.debug:
            print('-' * 20)
            print(command)
            print('-' * 20)
        else:
            command += ' > /dev/null 2>&1'
        os.system(command)

    def _quant(self, img, depth):
        assert img.max() <= 1 and img.min() >= 0, f'img max: {img.max()}, img min: {img.min()}'
        img = img.copy()
        img *= (2 ** depth - 1)
        dtype = np.uint8 if self.depth == 8 else np.uint16
        img = np.round(img).astype(dtype)
        return img

    def _inv_quant(self, img, depth):
        assert img.dtype is not np.float32
        img = img.astype(np.float32)
        img /= (2 ** depth - 1)
        return img

    def _rgb_to_yuv_file(self, rgb):
        assert rgb.max() <= 1 and rgb.min() >= 0, f'rgb max: {rgb.max()}, rgb min: {rgb.min()}'
        # gray
        tmp_yuv_file = self._get_tmp_file('yuv')
        if len(rgb.shape) == 2 or rgb.shape[2] == 1:
            y = self._quant(rgb, self.depth)
            with open(tmp_yuv_file, 'wb+') as f:
                f.write(y.tobytes())
        # rgb
        else:
            tmp_png_file = self._get_tmp_file('png')
            depth = 8 if self.depth == 8 else 16
            rgb = self._quant(rgb, depth)[..., ::-1]
            cv2.imwrite(tmp_png_file, rgb)
            pix_fmt = f'yuv{self.mode}p' if self.depth == 8 else f'yuv{self.mode}p{self.depth}le'
            command = f'{self.ffmpeg} -y -i {tmp_png_file} -pix_fmt {pix_fmt} {tmp_yuv_file}'
            self._system(command)
        return tmp_yuv_file

    def _yuv_file_to_rgb(self, yuv_file, h, w, mode):
        tmp_png_file = self._get_tmp_file('png')
        # gray
        if mode == 400:
            dtype = np.uint8 if self.depth == 8 else np.uint16
            y = np.fromfile(yuv_file, dtype).reshape([h, w])
            rgb = self._inv_quant(y, self.depth)
        # rgb
        else:
            pix_fmt = f'yuv{mode}p' if self.depth == 8 else f'yuv{mode}p{self.depth}le'
            command = f'{self.ffmpeg} -s {w}x{h} -y -pix_fmt {pix_fmt} -i {yuv_file} {tmp_png_file}'
            self._system(command)
            bgr = cv2.imread(tmp_png_file, cv2.IMREAD_UNCHANGED)
            if bgr is None:  # got unexpected error 'ffmpeg invalid data found when processing input'
                with open(yuv_file, 'rb') as f:
                    dtype = np.uint8 if self.depth == 8 else np.uint16
                    _size = w * h * 3 // 2 if self.depth == 8 else w * h * 3
                    yuv = np.frombuffer(f.read(_size), dtype=dtype).reshape(h * 3 // 2, w)
                    yuv = self._inv_quant(yuv, self.depth)
                    yuv = (yuv * 255).astype(np.uint8)
                    bgr = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR_I420)
                    if self.depth != 8:
                        bgr = bgr.astype(np.float32) / 255 * (2 ** 16 - 1)
                        bgr = bgr.astype(np.uint16)
            depth = 8 if self.depth == 8 else 16
            bgr = self._inv_quant(bgr, depth)
            rgb = bgr[..., ::-1]
        return rgb

    def _comp_rgb(self, rgb, quality):
        """
        use yuv 444 or yuv 420 to compression rgb
        :param rgb: np.array [0, 1]
        :param quality: int [1, 51]
        :return: reconstruct rgb, bits
        """
        assert rgb.max() <= 1 and rgb.min() >= 0, f'rgb max: {rgb.max()}, rgb min: {rgb.min}'
        # compression
        h, w = rgb.shape[:2]
        yuv_file = self._rgb_to_yuv_file(rgb)
        bin_file = self._get_tmp_file('bin')
        rec_file = self._get_tmp_file('yuv')
        command = self.command.format(h=h, w=w, mode=self.mode, q=quality,
                                      input=yuv_file, bin=bin_file, output=rec_file)
        self._system(command)
        re_rgb = self._yuv_file_to_rgb(rec_file, h, w, self.mode)
        bits = float(os.path.getsize(bin_file)) * 8
        return re_rgb, bits

    def _comp_raw(self, raw, quality):
        """
        use yuv 444 to compression r, gb, b, and use yuv 400 to compression gr
        :param raw: np.array [0, 1] (shape [h, w, 4])
        :param quality: int [1, 51]
        :return: reconstruct rgb, bits
        """
        assert self.mode == 444, 'raw compression only support 444'
        assert raw.max() <= 1 and raw.min() >= 0, f'raw max: {raw.max()}, raw min: {raw.min}'
        raw = np.where(raw <= 0.0031308, 12.92 * raw, 1.055 * np.power(raw, 1 / 2.4) - 0.055)
        r, gb, gr, b = np.split(raw, 4, axis=-1)
        # use 444 to comp rgb
        rgb = np.concatenate([r, gb, b], axis=-1)
        re_rgb, bits = self._comp_rgb(rgb, quality)
        # use 400 comp g
        gr -= gb
        g_min = gr.min()
        gr -= g_min
        gr = np.clip(gr, 0, 1)
        h, w = gr.shape[:2]
        yuv_file = self._rgb_to_yuv_file(gr)
        bin_file = self._get_tmp_file('bin')
        rec_file = self._get_tmp_file('yuv')
        command = self.command.format(h=h, w=w, mode=400, q=quality,
                                      input=yuv_file, bin=bin_file, output=rec_file)
        self._system(command)
        re_g = self._yuv_file_to_rgb(rec_file, h, w, 400)
        bits += float(os.path.getsize(bin_file)) * 8
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
