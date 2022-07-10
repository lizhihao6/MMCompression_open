import numpy as np
import torch
from mmcv import ConfigDict
from torch import nn

from mmcomp.models import CONTEXT, ENTROPY, LOSSES, QUANTS, VAE, build_compressor


def _demo_mm_inputs(input_shape=(1, 3, 8, 16)):
    """Create a superset of inputs needed to run test or train batches.

    Args:
        input_shape (tuple):
            input batch dimensions
    """
    (N, C, H, W) = input_shape

    rng = np.random.RandomState(0)

    imgs = rng.rand(*input_shape)

    img_metas = [{
        'img_shape': (H, W, C),
        'ori_shape': (H, W, C),
        'pad_shape': (H, W, C),
        'filename': '<demo>.png',
        'scale_factor': 1.0,
        'flip': False,
        'flip_direction': 'horizontal'
    } for _ in range(N)]

    mm_inputs = {
        'imgs': torch.FloatTensor(imgs),
        'img_metas': img_metas
    }
    return mm_inputs


@VAE.register_module()
class ExampleVAE(nn.Module):

    def __init__(self):
        super().__init__()
        self.enc = nn.Conv2d(3, 6, 1, 1)
        self.dec = nn.Conv2d(6, 3, 1, 1)
        self.hyper_enc = nn.Conv2d(6, 12, 1, 1)
        self.hyper_dec = nn.Conv2d(12, 6 * 2, 1, 1)


@ENTROPY.register_module()
class ExampleEntropy(nn.Module):

    def __init__(self):
        super(ExampleEntropy, self).__init__()
        self.conv = nn.Conv2d(12, 24, 1, 1)

    def forward(self, x):
        return self.conv(x)


@CONTEXT.register_module()
class ExampleContext(nn.Module):

    def __init__(self):
        super(ExampleContext, self).__init__()
        self.conv = nn.Conv2d(12, 12, 1, 1)

    def forward(self, x, x_prob):
        assert x.shape[1] * 2 == x_prob.shape[1]
        return self.conv(x_prob)


@QUANTS.register_module()
class ExampleQuant(nn.Module):

    def __init__(self):
        super(ExampleQuant, self).__init__()

    def forward(self, x):
        return torch.round(x) + (x - torch.round(x)) ** 3


@LOSSES.register_module()
class ExampleLoss(nn.Module):

    def __init__(self):
        super(ExampleLoss, self).__init__()
        self.mse = torch.nn.MSELoss()

    def forward(self, y, gt):
        return self.mse(y, gt)


def _compressor_forward_train_test(compressor):
    # batch_size=2 for BatchNorm
    mm_inputs = _demo_mm_inputs()

    imgs = mm_inputs.pop('imgs')
    img_metas = mm_inputs.pop('img_metas')

    # convert to cuda Tensor if applicable
    if torch.cuda.is_available():
        compressor = compressor.cuda()
        imgs = imgs.cuda()

    # Test forward train
    losses = compressor.forward(imgs, img_metas, return_loss=True)
    assert isinstance(losses, dict)

    # Test forward simple test
    with torch.no_grad():
        compressor.eval()
        compressor.forward(imgs, img_metas, return_loss=False)


def test_nic():
    # test nic

    cfg = ConfigDict(
        type='NICCompressor',
        pretrained=None,
        vae=dict(type='ExampleVAE'),
        entropy_model=dict(type="ExampleEntropy"),
        context_model=dict(type="ExampleContext"),
        quant=dict(type="ExampleQuant"),
        rec_loss=dict(type="ExampleLoss"),
        # model training and testing settings
        train_cfg=dict(lambda_rd=0.1, lambda_bpp=0.1),
        test_cfg=dict(mode='whole', mod_size=(64, 64)))

    compressor = build_compressor(cfg)
    _compressor_forward_train_test(compressor)

    # test slide mode
    cfg.test_cfg = ConfigDict(mode='slide', crop_size=(3, 3), stride=(2, 2), mod_size=(3, 3))
    compressor = build_compressor(cfg)
    _compressor_forward_train_test(compressor)
