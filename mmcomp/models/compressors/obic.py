import cv2
import torch
from torch import nn

from mmcomp.utils import tensor_to_image
from .nic import NICCompressor
from ..builder import COMPRESSOR


@COMPRESSOR.register_module()
class OBICCompressor(NICCompressor):
    """Object Based Compressor. Do not require finetuning, just set object and bkg lambda.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        for param in self.parameters():
            param.requires_grad = False
        self.factor_obj, self.factor_bkg = self.test_cfg['factor_obj'], self.test_cfg['factor_bkg']
        assert self.factor_obj > self.factor_bkg

    def _calculate_factor(self, features, mask):
        factor = torch.ones_like(features) * self.factor_bkg
        target_shape = features.shape
        mask = torch.nn.functional.interpolate(mask, size=target_shape[2:], mode='nearest').expand(target_shape)
        mask = mask * self.factor_obj
        factor = torch.maximum(factor, mask)
        return factor

    def _compression(self, img, **kwargs):
        features = self.main_encoder(img)
        hyper = self.hyper_encoder(features)

        # quant
        factor = self._calculate_factor(features, kwargs['mask'])
        features = features * factor
        if self.training:
            features_quant = self.quant_model(features)
            hyper_quant = self.quant_model(hyper)
        else:
            features_quant = torch.round(features)
            hyper_quant = torch.round(hyper)
        features_quant = features_quant / factor

        features_prob = self.hyper_decoder(hyper_quant)
        if hasattr(self, 'main_decoder'):
            rec_img = self.main_decoder(features_quant)
        else:
            rec_img = self.main_encoder(features_quant, reverse=True)

        main_prob = self.context_model(features_quant, features_prob, factor)
        hyper_prob = self.entropy_model(hyper_quant)
        num_pixels = img.shape[0] * img.shape[2] * img.shape[3]
        k = -1. / torch.log(torch.FloatTensor([2])) / num_pixels
        main_bpp = torch.sum(torch.log(main_prob)) * k.to(main_prob.device)
        hyper_bpp = torch.sum(torch.log(hyper_prob)) * k.to(hyper_prob.device)
        return rec_img, main_bpp, hyper_bpp

    def slide_inference(self, img, **kwargs):
        h_stride, w_stride = self.test_cfg.stride
        h_crop, w_crop = self.test_cfg.crop_size
        batch_size, c_img, h_img, w_img = img.size()
        h_grids = max(h_img - h_crop + h_stride - 1, 0) // h_stride + 1
        w_grids = max(w_img - w_crop + w_stride - 1, 0) // w_stride + 1
        rec_img = img.new_zeros((batch_size, c_img, h_img, w_img))
        count_mat = img.new_zeros((batch_size, 1, h_img, w_img))
        total_bits, pixel_counter = 0., 0
        mask = kwargs['mask'].clone()
        for h_idx in range(h_grids):
            for w_idx in range(w_grids):
                y1 = h_idx * h_stride
                x1 = w_idx * w_stride
                y2 = min(y1 + h_crop, h_img)
                x2 = min(x1 + w_crop, w_img)
                y1 = max(y2 - h_crop, 0)
                x1 = max(x2 - w_crop, 0)
                crop_img = img[:, :, y1:y2, x1:x2]
                kwargs['mask'] = mask[:, :, y1:y2, x1:x2]
                crop_rec_img, main_bpp, hyper_bpp = self._compression(crop_img, **kwargs)
                rec_img += nn.functional.pad(crop_rec_img,
                                             (int(x1), int(rec_img.shape[3] - x2), int(y1), int(rec_img.shape[2] - y2)))
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
        kwargs['mask'] = mask
        return rec_img, total_bits / pixel_counter

    def forward_train(self, img, **kwargs):
        raise NotImplementedError

    def forward_test(self, img, return_image, **kwargs):
        """Simple test with single image."""
        results = super().forward_test(img, return_image, **kwargs)
        if return_image:
            results['mask'] = kwargs['mask'].detach().clone()
        return results

    def show_result(self, data, result, show=False, out_file=None):
        super(OBICCompressor, self).show_result(data, result, show, out_file)
        mask = tensor_to_image(result['mask'])
        if out_file is None:
            pass
        else:
            suffix = out_file.split('.')[-1]
            assert suffix == 'png'
            cv2.imwrite(out_file[:-3] + 'mask.png', mask)
