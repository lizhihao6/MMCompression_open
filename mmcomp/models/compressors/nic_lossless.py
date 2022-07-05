import torch

from .nic import NICCompressor
from .. import builder
from ..builder import COMPRESSOR


@COMPRESSOR.register_module()
class NICLosslessCompressor(NICCompressor):
    """Our Neural Image Compression
    """

    def __init__(self, **kwargs):
        residual_entropy_model_cfg = kwargs.pop("residual_entropy_model")
        self.scale_factor = kwargs.pop("scale_factor")
        super(NICLosslessCompressor, self).__init__(**kwargs)
        self.residual_entropy_model = builder.build_entropy_model(residual_entropy_model_cfg)

    def _compression(self, img, **kwargs):
        rec_img, main_bpp, hyper_bpp = super()._compression(img, **kwargs)
        rec_img = torch.clip(rec_img, 0, 1)
        rec_img = rec_img * self.scale_factor
        if self.training:
            rec_img = self.quant_model(rec_img)
        else:
            rec_img = torch.round(rec_img)
        img = torch.round(img * self.scale_factor)
        residual_quant = img - rec_img
        residual_prob = self.residual_entropy_model(residual_quant)
        rec_img = (rec_img + residual_quant) / self.scale_factor
        num_pixels = img.shape[0] * img.shape[2] * img.shape[3]
        k = -1. / torch.log(torch.FloatTensor([2])) / num_pixels
        main_bpp = main_bpp + torch.sum(torch.log(residual_prob)) * k.to(residual_prob.device)
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
        losses["main_bpp_loss"] = main_bpp
        losses["hyper_bpp_loss"] = hyper_bpp
        losses["psnr"] = self._calculate_psnr(rec_img, img)
        losses["bpp"] = main_bpp + hyper_bpp
        return losses
