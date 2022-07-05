import cv2
import numpy as np
import torch
import torch.nn.functional as F

from mmcomp.utils import tensor_to_image
from .base import BaseCompressor
from .. import builder
from ..builder import COMPRESSOR


@COMPRESSOR.register_module()
class NICCompressor(BaseCompressor):
    """Our Neural Image Compression
    """

    def __init__(self,
                 main_encoder,
                 main_decoder,
                 hyper_encoder,
                 hyper_decoder,
                 entropy_model,
                 context_model,
                 quant,
                 rec_loss,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super(NICCompressor, self).__init__()
        self.main_encoder = builder.build_main_encoder(main_encoder)
        if main_decoder is not None:
            # use vae
            self.main_decoder = builder.build_main_decoder(main_decoder)
        self.hyper_encoder = builder.build_hyper_encoder(hyper_encoder)
        self.hyper_decoder = builder.build_hyper_decoder(hyper_decoder)
        self.entropy_model = builder.build_entropy_model(entropy_model)
        self.context_model = builder.build_context_model(context_model)
        self.quant_model = builder.build_quant(quant)
        self.rec_loss_fn = builder.build_loss(rec_loss)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        # import pickle
        # with open('tmp.pkl', 'wb+') as f:
        #     pickle.dump(list(self.state_dict().keys()), f)
        self.init_weights(pretrained=pretrained)

    def _compression(self, img, **kwargs):
        features = self.main_encoder(img)
        hyper = self.hyper_encoder(features)
        if self.training:
            features_quant = self.quant_model(features)
            hyper_quant = self.quant_model(hyper)
        else:
            features_quant = torch.round(features)
            hyper_quant = torch.round(hyper)
        features_prob = self.hyper_decoder(hyper_quant)
        if hasattr(self, 'main_decoder'):
            rec_img = self.main_decoder(features_quant)
        else:
            if self.training:
                _features_quant = features_quant
            else:
                # todo: fix the quantization error when use flow at high rates
                _features_quant = self.quant_model(features)
            rec_img = self.main_encoder(_features_quant, reverse=True)
        main_prob = self.context_model(features_quant, features_prob)
        hyper_prob = self.entropy_model(hyper_quant)
        num_pixels = img.shape[0] * img.shape[2] * img.shape[3]
        k = -1. / torch.log(torch.FloatTensor([2])) / num_pixels
        main_bpp = torch.sum(torch.log(main_prob)) * k.to(main_prob.device)
        hyper_bpp = torch.sum(torch.log(hyper_prob)) * k.to(hyper_prob.device)
        return rec_img, main_bpp, hyper_bpp

    def forward_train(self, img, **kwargs):
        """Forward function for training.

        Args:
            img (Tensor): Input images.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        rec_img, main_bpp, hyper_bpp = self._compression(img, **kwargs)
        losses = dict()
        losses["rec_loss"] = self.train_cfg.lambda_rd * self.rec_loss_fn(rec_img, img)
        bpp_scale = self.train_cfg.get('lambda_bpp_scale', 0.01)
        losses["main_bpp_loss"] = bpp_scale * main_bpp
        losses["hyper_bpp_loss"] = bpp_scale * hyper_bpp
        losses["psnr"] = self._calculate_psnr(rec_img, img)
        losses["bpp"] = main_bpp + hyper_bpp
        return losses

    # TODO refactor
    def slide_inference(self, img, **kwargs):
        """Inference by sliding-window with overlap.

        If h_crop > h_img or w_crop > w_img, the small patch will be used to
        decode without padding.
        """
        h_stride, w_stride = self.test_cfg.stride
        h_crop, w_crop = self.test_cfg.crop_size
        batch_size, c_img, h_img, w_img = img.size()
        h_grids = max(h_img - h_crop + h_stride - 1, 0) // h_stride + 1
        w_grids = max(w_img - w_crop + w_stride - 1, 0) // w_stride + 1
        rec_img = img.new_zeros((batch_size, c_img, h_img, w_img))
        count_mat = img.new_zeros((batch_size, 1, h_img, w_img))
        total_bits, pixel_counter = 0., 0
        for h_idx in range(h_grids):
            for w_idx in range(w_grids):
                y1 = h_idx * h_stride
                x1 = w_idx * w_stride
                y2 = min(y1 + h_crop, h_img)
                x2 = min(x1 + w_crop, w_img)
                # y1 = max(y2 - h_crop, 0)
                # x1 = max(x2 - w_crop, 0)
                if (y2 - y1) % 64 != 0:
                    y1 = max(y2 - (y2 - y1) // 64 * 64 - 64, 0)
                if (x2 - x1) % 64 != 0:
                    x1 = max(x2 - (x2 - x1) // 64 * 64 - 64, 0)
                crop_img = img[:, :, y1:y2, x1:x2]
                with torch.no_grad():
                    crop_rec_img, main_bpp, hyper_bpp = self._compression(crop_img, **kwargs)
                rec_img += F.pad(crop_rec_img,
                                 (int(x1), int(rec_img.shape[3] - x2), int(y1),
                                  int(rec_img.shape[2] - y2)))
                bpp = main_bpp + hyper_bpp
                pixels = (y2 - y1) * (x2 - x1)
                total_bits += float(bpp) * pixels
                pixel_counter += pixels
                count_mat[:, :, y1:y2, x1:x2] += 1
        assert (count_mat == 0).sum() == 0
        if torch.onnx.is_in_onnx_export():
            # cast count_mat to constant while exporting to ONNX
            count_mat = torch.from_numpy(
                count_mat.cpu().detach().numpy()).to(device=img.device)
        rec_img = rec_img / count_mat
        return rec_img, total_bits / pixel_counter

    def whole_inference(self, img, **kwargs):
        """Inference with full image."""
        # print(img.size())
        batch_size, c_img, h_img, w_img = img.size()
        pad_h, pad_w, target_h, target_w = 0, 0, h_img, w_img
        if h_img % 256 != 0:
            target_h = (h_img // 256 + 1) * 256
            pad_h = (target_h - h_img) // 2
        if w_img % 256 != 0:
            target_w = (w_img // 256 + 1) * 256
            pad_w = (target_w - w_img) // 2
        _img = torch.nn.functional.pad(img, [pad_w, target_w - pad_w - w_img, pad_h, target_h - pad_h - h_img])
        with torch.no_grad():
            rec_img, main_bpp, hyper_bpp = self._compression(_img, **kwargs)
        rec_img = rec_img[:, :, pad_h:pad_h + h_img, pad_w:pad_w + w_img]
        bpp = main_bpp + hyper_bpp
        ori_num_pixels = batch_size * h_img * w_img
        tar_num_pixels = batch_size * target_h * target_w
        bpp = bpp * tar_num_pixels / ori_num_pixels
        return rec_img, bpp

    def inference(self, img, **kwargs):
        """Inference with slide/whole style.

        Args:
            img (Tensor): The input image of shape (N, 3, H, W).

        Returns:
            Tensor: The output segmentation map.
        """
        torch.cuda.empty_cache()
        assert self.test_cfg.mode in ['slide', 'whole']
        if self.test_cfg.mode == 'slide':
            rec_img, bpp = self.slide_inference(img, **kwargs)
        else:
            rec_img, bpp = self.whole_inference(img, **kwargs)
        torch.cuda.empty_cache()
        return rec_img, bpp

    def forward_test(self, img, return_image, **kwargs):
        """Simple test with single image."""
        self.eval()
        rec_img, bpp = self.inference(img, **kwargs)
        rec_loss = self.rec_loss_fn(rec_img, img)
        psnr = self._calculate_psnr(rec_img, img)
        results = dict(
            bpp=float(bpp),
            rec_loss=float(rec_loss),
            psnr=float(psnr)
        )
        if return_image:
            results['ori_img'] = img.detach().clone()
            results['rec_img'] = rec_img.detach().clone()
        del img, rec_img, bpp, rec_loss, psnr
        self.train()
        return results

    def show_result(self, data, result, show=False, out_file=None):
        ori_img = tensor_to_image(result['ori_img'])[..., ::-1]
        rec_img = tensor_to_image(result['rec_img'])[..., ::-1]
        if show:
            caption = "compression result" if out_file is None else out_file
            cv2.imshow(caption, np.hstack([ori_img, rec_img]))
            cv2.waitKey(0)

        if out_file is None:
            pass
        else:
            suffix = out_file.split('.')[-1]
            assert suffix == 'png'
            cv2.imwrite(out_file[:-3] + 'gt.png', ori_img)
            cv2.imwrite(out_file, rec_img)
