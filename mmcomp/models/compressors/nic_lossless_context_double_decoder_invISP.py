import torch

from .nic_lossless_context_double_decoder import NICLosslessContextDoubleDecoderCompressor
from .. import builder
from ..builder import COMPRESSOR
from mmgen.models import build_module
from collections import OrderedDict
import torch.distributions as D

def mosaic(x):
    b, c, h, w = x.shape
    _x = torch.zeros([b, 4, h // 2, w // 2], device=x.device)
    _x[:, 0, :, :] = x[:, 0, 0::2, 0::2]
    _x[:, 1, :, :] = x[:, 1, 0::2, 1::2]
    _x[:, 2, :, :] = x[:, 1, 1::2, 0::2]
    _x[:, 3, :, :] = x[:, 2, 1::2, 1::2]
    return _x

@COMPRESSOR.register_module()
class NICLosslessContextDoubleDecoderCompressorinvISP(NICLosslessContextDoubleDecoderCompressor):
    """Our Neural Image Compression
    """

    def __init__(self, **kwargs):
        invISPtype = kwargs.pop('inv_ISP_type', 'inverseISP')
        self.ganISP_pretrained = kwargs.pop('ganISP_pretrained')

        super().__init__(**kwargs)
        cfg = dict(type=invISPtype)
        self.ganISP = build_module(cfg)
        self.loaded = False

    def _get_condition(self, img):
        # condition = torch.randn([img.shape[0], self.c_condition]).to(img.device).detach()
        mean_var = self.ganISP.color_condition_gen(img)
        m = D.Normal(mean_var[:, 0], torch.clamp_min(torch.abs(mean_var[:, 1]), 1e-6))
        color_condition = m.sample()
        mean_var = self.ganISP.bright_condition_gen(img)
        m = D.Normal(mean_var[:, 0], torch.clamp_min(torch.abs(mean_var[:, 1]), 1e-6))
        bright_condition = m.sample()
        condition = torch.cat([color_condition[:, None], bright_condition[:, None]], 1)
        return condition

    def _compression(self, img, **kwargs):
        assert img.shape[1] == 4
        rec_img, main_bpp, hyper_bpp = super()._compression(img, **kwargs)
        main_bpp, hyper_bpp = main_bpp / 4, hyper_bpp / 4
        return rec_img, main_bpp, hyper_bpp

    def forward(self, img, return_loss=True, return_image=False, **kwargs):
        """Calls either :func:`forward_train` or :func:`forward_test` depending
        on whether ``return_loss`` is ``True``.

        Note this setting will change the expected inputs. When
        ``return_loss=True``, img and img_meta are single-nested (i.e. Tensor
        and List[dict]), and when ``resturn_loss=False``, img and img_meta
        should be double nested (i.e.  List[Tensor], List[List[dict]]), with
        the outer list indicating test time augmentations.
        """
        if return_loss:
            assert img.shape[1] == 3, 'Expected 3 channels, got {}'.format(x.shape[1])
            if not self.loaded:
                self.load_ganISP_checkpoint(self.ganISP_pretrained)
                self.loaded = True
            with torch.no_grad():
                condition = self._get_condition(img)
                img = self.ganISP(img, condition)
                img = torch.clamp(img, 0, 1)
                img = mosaic(img)
            return self.forward_train(img, **kwargs)
        else:
            return self.forward_test(img, return_image, **kwargs)

    def load_ganISP_checkpoint(self, checkpoint_path):
        """Load ganISP checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        state_dict = OrderedDict()
        for k, v in checkpoint['state_dict'].items():
            if k.startswith('generator.'):
                state_dict[k[len('generator.'):]] = v
        self.ganISP.load_state_dict(state_dict, strict=False)
        self.ganISP.eval()