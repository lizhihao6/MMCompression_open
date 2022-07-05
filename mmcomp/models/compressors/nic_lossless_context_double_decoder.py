import torch

from .nic import NICCompressor
from .. import builder
from ..builder import COMPRESSOR


@COMPRESSOR.register_module()
class NICLosslessContextDoubleDecoderCompressor(NICCompressor):
    """Our Neural Image Compression
    """

    def __init__(self, **kwargs):
        residual_decoder_cfg = kwargs.pop("residual_decoder")
        residual_context_model_cfg = kwargs.pop("residual_context_model")
        self.scale_factor = kwargs.pop("scale_factor")
        super().__init__(**kwargs)
        self.residual_decoder = builder.build_main_decoder(residual_decoder_cfg)
        self.residual_context_model = builder.build_context_model(residual_context_model_cfg)

    def _quant(self, x):
        return self.quant_model(x) if self.training else torch.round(x)

    def _compression(self, img, **kwargs):
        features = self.main_encoder(img)
        hyper = self.hyper_encoder(features)
        features_quant = self._quant(features)
        hyper_quant = self._quant(hyper)

        rec_img = self.main_decoder(features_quant)
        rec_img = torch.clip(rec_img, 0, 1)

        _rec_img = self._quant(rec_img * self.scale_factor)
        _img = torch.round(img * self.scale_factor)
        residual_quant = _img - _rec_img
        residual_prior = self.residual_decoder(features_quant)

        features_prob = self.hyper_decoder(hyper_quant)

        main_prob = self.context_model(features_quant, features_prob)
        hyper_prob = self.entropy_model(hyper_quant)
        residual_prob = self.residual_context_model(residual_quant, residual_prior)
        num_pixels = img.shape[0] * img.shape[2] * img.shape[3]
        k = -1. / torch.log(torch.FloatTensor([2])) / num_pixels
        main_bpp = torch.sum(torch.log(main_prob)) * k.to(main_prob.device)
        hyper_bpp = torch.sum(torch.log(hyper_prob)) * k.to(hyper_prob.device)
        residual_bpp = torch.sum(torch.log(residual_prob)) * k.to(residual_prob.device)

        main_bpp = main_bpp + residual_bpp

        rec_img = (_rec_img + residual_quant) / self.scale_factor
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
