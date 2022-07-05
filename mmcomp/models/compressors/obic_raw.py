from .obic import OBICCompressor
from ..builder import COMPRESSOR


@COMPRESSOR.register_module()
class OBICRAWCompressor(OBICCompressor):
    """Our RAW Image Compression
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _compression(self, img, **kwargs):
        assert img.shape[1] == 4
        rec_img, main_bpp, hyper_bpp = super()._compression(img, **kwargs)
        main_bpp, hyper_bpp = main_bpp / 4, hyper_bpp / 4
        return rec_img, main_bpp, hyper_bpp
