import cv2
import torch

from mmcompression.utils import tensor_to_image
from .base import BaseCompressor
from .. import builder
from ..builder import COMPRESSOR


@COMPRESSOR.register_module()
class NICCompressor(BaseCompressor):
    """Our Neural Image Compression"""

    def __init__(
            self,
            vae,
            entropy_model,
            context_model,
            quant,
            rec_loss,
            train_cfg=None,
            test_cfg=None,
            pretrained=None,
    ):
        super().__init__()
        self.vae = builder.build_vae(vae)
        self.entropy_model = builder.build_entropy_model(entropy_model)
        self.context_model = builder.build_context_model(context_model)
        self.quant_model = builder.build_quant(quant)
        self.rec_loss_fn = builder.build_loss(rec_loss)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.init_weights(pretrained=pretrained)

    def _quant(self, x):
        return self.quant_model(x) if self.training else torch.round(x)

    def _compression(self, img):
        features = self.vae.enc(img)
        features_quant = self._quant(features)
        rec_img = self.vae.dec(features_quant)

        hyper = self.vae.hyper_enc(features)
        hyper_quant = self._quant(hyper)
        features_prob = self.vae.hyper_dec(hyper_quant)

        main_prob = self.context_model(features_quant, features_prob)
        hyper_prob = self.entropy_model(hyper_quant)

        num_pixels = img.shape[0] * img.shape[2] * img.shape[3]
        k = -1.0 / torch.log(torch.FloatTensor([2])) / num_pixels
        main_bpp = torch.sum(torch.log(main_prob)) * k.to(main_prob.device)
        hyper_bpp = torch.sum(torch.log(hyper_prob)) * k.to(hyper_prob.device)
        return rec_img, main_bpp, hyper_bpp

    def forward_train(self, img, img_metas, **kwargs):
        """Forward function for training.
        Args:
            img (Tensor): Input images.
            img_metas (list[dict]): Meta information of input images.
        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        rec_img, main_bpp, hyper_bpp = self._compression(img)
        losses = {"rec_loss": self.train_cfg.lambda_rd * self.rec_loss_fn(rec_img, img)}
        bpp_scale = self.train_cfg.get("lambda_bpp", 0.01)
        losses["main_bpp_loss"] = bpp_scale * main_bpp
        losses["hyper_bpp_loss"] = bpp_scale * hyper_bpp
        losses["psnr"] = self._calculate_psnr(rec_img, img)
        losses["bpp"] = main_bpp + hyper_bpp
        return losses

    def slide_inference(self, img):
        """Inference by sliding-window with overlap.

        If h_crop > h_img or w_crop > w_img, the small patch will be used to
        decode with padding.
        """
        h_stride, w_stride = self.test_cfg.stride
        h_crop, w_crop = h_stride, w_stride
        h_mod, w_mod = self.test_cfg.mod_size

        batch_size, c_img, h_img, w_img = img.size()
        h_grids = max(h_img - h_crop + h_stride - 1, 0) // h_stride + 1
        w_grids = max(w_img - w_crop + w_stride - 1, 0) // w_stride + 1
        rec_img = img.new_zeros((batch_size, c_img, h_img, w_img))
        count_mat = img.new_zeros((batch_size, 1, h_img, w_img))
        total_bits = 0.0
        for h_idx in range(h_grids):
            for w_idx in range(w_grids):
                y1 = h_idx * h_stride
                x1 = w_idx * w_stride
                y2 = min(y1 + h_crop, h_img)
                x2 = min(x1 + w_crop, w_img)

                crop_img = img[:, :, y1:y2, x1:x2]
                h, w = crop_img.shape[2:]
                pad_h, pad_w, target_h, target_w = 0, 0, h, w
                if h % h_mod != 0:
                    target_h = (h // h_mod + 1) * h_mod
                    pad_h = (target_h - h) // 2
                if w % w_mod != 0:
                    target_w = (w // w_mod + 1) * w_mod
                    pad_w = (target_w - w) // 2

                crop_img = torch.nn.functional.pad(
                    crop_img,
                    [pad_w, target_w - pad_w - w, pad_h, target_h - pad_h - h],
                )
                with torch.no_grad():
                    crop_rec_img, main_bpp, hyper_bpp = self._compression(crop_img)
                rec_img[:, :, y1:y2, x1:x2] = crop_rec_img[
                                              :, :, pad_h: pad_h + h, pad_w: pad_w + w
                                              ]

                bpp = main_bpp + hyper_bpp
                pixels = batch_size * target_h * target_w
                total_bits += float(bpp) * pixels
                count_mat[:, :, y1:y2, x1:x2] += 1
        assert (count_mat == 0).sum() == 0 and (
                count_mat == 1
        ).sum() == count_mat.numel()
        return rec_img, total_bits / count_mat.numel()

    def whole_inference(self, img):
        """Inference with full image."""
        h_mod, w_mod = self.test_cfg.mod_size

        batch_size, _, h_img, w_img = img.size()
        pad_h, pad_w, target_h, target_w = 0, 0, h_img, w_img
        if h_img % h_mod != 0:
            target_h = (h_img // h_mod + 1) * h_mod
            pad_h = (target_h - h_img) // 2
        if w_img % w_mod != 0:
            target_w = (w_img // w_mod + 1) * w_mod
            pad_w = (target_w - w_img) // 2
        _img = torch.nn.functional.pad(
            img, [pad_w, target_w - pad_w - w_img, pad_h, target_h - pad_h - h_img]
        )
        with torch.no_grad():
            rec_img, main_bpp, hyper_bpp = self._compression(_img)
        rec_img = rec_img[:, :, pad_h: pad_h + h_img, pad_w: pad_w + w_img]
        bpp = main_bpp + hyper_bpp
        ori_num_pixels = batch_size * h_img * w_img
        tar_num_pixels = batch_size * target_h * target_w
        bpp = bpp * tar_num_pixels / ori_num_pixels
        return rec_img, bpp

    def inference(self, img):
        """Inference with slide/whole style.

        Args:
            img (Tensor): The input image of shape (N, 3, H, W).

        Returns:
            Tensor: The output segmentation map.
        """
        assert self.test_cfg.mode in ["slide", "whole"]
        if self.test_cfg.mode == "slide":
            rec_img, bpp = self.slide_inference(img)
        else:
            rec_img, bpp = self.whole_inference(img)
        return rec_img, bpp

    def forward_test(self, img, img_metas, return_image, **kwargs):
        """Simple test with single image."""
        self.eval()
        with torch.no_grad():
            rec_img, bpp = self.inference(img)
        rec_loss = self.rec_loss_fn(rec_img, img)
        psnr = self._calculate_psnr(rec_img, img)
        results = dict(bpp=float(bpp), rec_loss=float(rec_loss), psnr=float(psnr))
        if return_image:
            ori_img_path = self._get_tmp_file("png")
            rec_img_path = self._get_tmp_file("png")
            cv2.imwrite(ori_img_path, tensor_to_image(img[0]))
            cv2.imwrite(rec_img_path, tensor_to_image(rec_img[0]))
            results["ori_img"] = ori_img_path
            results["rec_img"] = rec_img_path
        self.train()
        return results
