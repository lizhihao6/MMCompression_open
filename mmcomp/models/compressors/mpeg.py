import os

import cv2
import numpy as np

from mmcomp.utils import numpy_to_img
from .base import BaseCompressor
from ..builder import COMPRESSOR


@COMPRESSOR.register_module()
class MPEGCompressor(BaseCompressor):
    """Our Neural Image Compression
    Args:
        mpeg (str): mpeg encoder path
        mpeg_cfg (str): mpeg encoder config path
        depth (int): bit depth
        mode (str): 420 or 444
        qp (int): quality
        kwargs: other args
    """

    def __init__(
            self, mpeg=None, mpeg_cfg=None, depth=None, mode=420, qp=51, **kwargs
    ):
        super().__init__()
        assert mode in [420, 444], "only support 420, 444 now"
        self.depth = depth
        self.mode = mode
        self.qp = qp
        self.command = f"{mpeg} -c {mpeg_cfg} "
        self.command += (
            f"--InputBitDepth={depth} "
            f"--InternalBitDepth={depth} "
            f"--OutputBitDepth={depth} "
            f"--FrameRate=50 "
            f"--FrameSkip=0 "
            f"--FramesToBeEncoded=1 "
            f"--Level=2.1 "
        )
        self.command += (
            "--SourceWidth={w} "
            "--SourceHeight={h} "
            "--InputChromaFormat={mode} "
            "--QP={q} "
            "-i {input} -b {bin} -o {output}"
        )
        self.ffmpeg = "/usr/bin/ffmpeg"

    @staticmethod
    def _system(command):
        """
        Run system command.
        Args:
            command (str): system command
        """
        command += " > /dev/null 2>&1"
        os.system(command)

    def _quant(self, img, depth):
        """
        Quantize image according to depth.
        Args:
            img (np.ndarray): image
            depth (int): bit depth
        Returns:
            quantized image (np.ndarray): quantized image
        """
        assert (
                img.max() <= 1 and img.min() >= 0
        ), f"img max: {img.max()}, img min: {img.min()}"
        img = img.copy()
        img *= 2 ** depth - 1
        dtype = np.uint8 if self.depth == 8 else np.uint16
        img = np.round(img).astype(dtype)
        return img

    @staticmethod
    def _inv_quant(img, depth):
        """
        Inverse quantize image according to depth.
        Args:
            img (np.ndarray): image
            depth (int): bit depth
        Returns:
            inverse quantized image (np.ndarray): inverse quantized image
        """
        assert img.dtype is not np.float32
        img = img.astype(np.float32)
        img /= 2 ** depth - 1
        return img

    def _rgb_to_yuv_file(self, rgb):
        """
        Convert rgb image to yuv file.
        Args:
            rgb (np.ndarray): rgb image
        Returns:
            yuv_file (str): yuv file path
        """
        assert (
                rgb.max() <= 1 and rgb.min() >= 0
        ), f"rgb max: {rgb.max()}, rgb min: {rgb.min()}"
        # gray
        tmp_yuv_file = self._get_tmp_file("yuv")
        if len(rgb.shape) == 2 or rgb.shape[2] == 1:
            y = self._quant(rgb, self.depth)
            with open(tmp_yuv_file, "wb+") as f:
                f.write(y.tobytes())
        # rgb
        else:
            tmp_png_file = self._get_tmp_file("png")
            depth = 8 if self.depth == 8 else 16
            rgb = self._quant(rgb, depth)[..., ::-1]
            cv2.imwrite(tmp_png_file, rgb)
            pix_fmt = (
                f"yuv{self.mode}p"
                if self.depth == 8
                else f"yuv{self.mode}p{self.depth}le"
            )
            command = (
                f"{self.ffmpeg} -y -i {tmp_png_file} -pix_fmt {pix_fmt} {tmp_yuv_file}"
            )
            self._system(command)
        return tmp_yuv_file

    def _yuv_file_to_rgb(self, yuv_file, h, w, mode):
        """
        Convert yuv file to rgb image.
        Args:
            yuv_file (str): yuv file path
            h (int): height
            w (int): width
            mode (str): 420 or 444
        Returns:
            rgb (np.ndarray): rgb image
        """
        tmp_png_file = self._get_tmp_file("png")
        # gray
        if mode == 400:
            dtype = np.uint8 if self.depth == 8 else np.uint16
            y = np.fromfile(yuv_file, dtype).reshape([h, w])
            rgb = self._inv_quant(y, self.depth)
        # rgb
        else:
            pix_fmt = f"yuv{mode}p" if self.depth == 8 else f"yuv{mode}p{self.depth}le"
            command = f"{self.ffmpeg} -s {w}x{h} -y -pix_fmt {pix_fmt} -i {yuv_file} {tmp_png_file}"
            self._system(command)
            bgr = cv2.imread(tmp_png_file, cv2.IMREAD_UNCHANGED)
            if (
                    bgr is None
            ):  # got unexpected error 'ffmpeg invalid data found when processing input'
                with open(yuv_file, "rb") as f:
                    dtype = np.uint8 if self.depth == 8 else np.uint16
                    _size = w * h * 3 // 2 if self.depth == 8 else w * h * 3
                    yuv = np.frombuffer(f.read(_size), dtype=dtype).reshape(
                        h * 3 // 2, w
                    )
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

    def _compression(self, rgb, quality):
        """
        Use yuv 444 or yuv 420 to compression rgb.
        Args:
            rgb (np.ndarray): rgb image
            quality (int): quality
        Returns:
            np.ndarray: compressed rgb image
        """
        assert (
                rgb.max() <= 1 and rgb.min() >= 0
        ), f"rgb max: {rgb.max()}, rgb min: {rgb.min}"
        # compression
        h, w = rgb.shape[:2]
        yuv_file = self._rgb_to_yuv_file(rgb)
        bin_file = self._get_tmp_file("bin")
        rec_file = self._get_tmp_file("yuv")
        command = self.command.format(
            h=h,
            w=w,
            mode=self.mode,
            q=quality,
            input=yuv_file,
            bin=bin_file,
            output=rec_file,
        )
        self._system(command)
        re_rgb = self._yuv_file_to_rgb(rec_file, h, w, self.mode)
        bits = float(os.path.getsize(bin_file)) * 8
        return re_rgb, bits

    def forward_train(self, img, img_metas, **kwargs):
        raise NotImplementedError

    def forward_test(self, img, img_metas, return_image, **kwargs):
        """Simple test with single image."""
        assert img.shape[0] == 1
        ori_img = img[0].permute([1, 2, 0]).detach().cpu().numpy()
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
