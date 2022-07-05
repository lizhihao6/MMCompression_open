import numpy as np
import torch
from mmcv import ConfigDict
from torch import nn

from mmcomp.models import MAINENCODER, MAINDECODER, HYPERENCODER, HYPERDECODER, CONTEXT, ENTROPY, LOSSES, QUANTS, \
    build_compressor


def _demo_mm_inputs(input_shape=(1, 3, 8, 16), mask=False):
    """Create a superset of inputs needed to run test or train batches.

    Args:
        input_shape (tuple):
            input batch dimensions

        mask (int):
            mask
    """
    (N, C, H, W) = input_shape

    rng = np.random.RandomState(0)

    imgs = rng.rand(*input_shape)
    if mask:
        _mask = rng.randint(
            low=0, high=1, size=(N, 1, H, W)).astype(np.uint8)

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
    if mask:
        mm_inputs['mask'] = torch.FloatTensor(_mask)
    return mm_inputs


@MAINENCODER.register_module()
class ExampleMainEncoder(nn.Module):

    def __init__(self):
        super(ExampleMainEncoder, self).__init__()
        self.conv = nn.Conv2d(3, 6, 1, 1)

    def init_weights(self, pretrained=None):
        pass

    def forward(self, x):
        return self.conv(x)


@MAINDECODER.register_module()
class ExampleMainDecoder(nn.Module):

    def __init__(self):
        super(ExampleMainDecoder, self).__init__()
        self.conv = nn.Conv2d(6, 3, 1, 1)

    def forward(self, x):
        return self.conv(x)


@HYPERENCODER.register_module()
class ExampleHyperEncoder(nn.Module):

    def __init__(self):
        super(ExampleHyperEncoder, self).__init__()
        self.conv = nn.Conv2d(6, 12, 1, 1)

    def init_weights(self, pretrained=None):
        pass

    def forward(self, x):
        return self.conv(x)


@HYPERDECODER.register_module()
class ExampleHyperDecoder(nn.Module):

    def __init__(self):
        super(ExampleHyperDecoder, self).__init__()
        self.conv = nn.Conv2d(12, 12, 1, 1)

    def forward(self, x):
        return self.conv(x)


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


def _compressor_forward_train_test(compressor, mask=False):
    # batch_size=2 for BatchNorm
    mm_inputs = _demo_mm_inputs(mask=mask)

    imgs = mm_inputs.pop('imgs')
    img_metas = mm_inputs.pop('img_metas')
    if mask:
        _mask = mm_inputs['mask']

    # convert to cuda Tensor if applicable
    if torch.cuda.is_available():
        compressor = compressor.cuda()
        imgs = imgs.cuda()
        if mask:
            _mask = _mask.cuda()

    # Test forward train
    if mask:
        losses = compressor.forward(imgs, img_metas, mask=_mask, return_loss=True)
    else:
        losses = compressor.forward(imgs, img_metas, return_loss=True)
    assert isinstance(losses, dict)

    # Test forward simple test
    with torch.no_grad():
        compressor.eval()
        # pack into lists
        img_list = [img[None, :] for img in imgs]
        img_meta_list = [[img_meta] for img_meta in img_metas]
        if mask:
            compressor.forward(img_list, img_meta_list, mask=_mask, return_loss=False)
        else:
            compressor.forward(img_list, img_meta_list, return_loss=False)


def test_nic():
    # test nic

    cfg = ConfigDict(
        type='NICCompressor',
        main_encoder=dict(type="ExampleMainEncoder"),
        main_decoder=dict(type="ExampleMainDecoder"),
        hyper_encoder=dict(type="ExampleHyperEncoder"),
        hyper_decoder=dict(type="ExampleHyperDecoder"),
        entropy_model=dict(type="ExampleEntropy"),
        context_model=dict(type="ExampleContext"),
        quant=dict(type="ExampleQuant"),
        rec_loss=dict(type="ExampleLoss"),
        train_cfg=dict(lambda_rd=0.1),
        test_cfg=dict(mode='whole'))
    compressor = build_compressor(cfg)
    _compressor_forward_train_test(compressor)

    # test slide mode
    cfg.test_cfg = ConfigDict(mode='slide', crop_size=(3, 3), stride=(2, 2))
    compressor = build_compressor(cfg)
    _compressor_forward_train_test(compressor)


def test_obic():
    # test obic

    cfg = ConfigDict(
        type='OBICCompressor',
        main_encoder_obj=dict(type="ExampleMainEncoder"),
        main_encoder_bkg=dict(type="ExampleMainEncoder"),
        main_decoder=dict(type="ExampleMainDecoder"),
        hyper_encoder_obj=dict(type="ExampleHyperEncoder"),
        hyper_encoder_bkg=dict(type="ExampleHyperEncoder"),
        hyper_decoder_obj=dict(type="ExampleHyperDecoder"),
        hyper_decoder_bkg=dict(type="ExampleHyperDecoder"),
        entropy_model_obj=dict(type="ExampleEntropy"),
        entropy_model_bkg=dict(type="ExampleEntropy"),
        context_model_obj=dict(type="ExampleContext"),
        context_model_bkg=dict(type="ExampleContext"),
        quant=dict(type="ExampleQuant"),
        rec_loss=dict(type="ExampleLoss"),
        train_cfg=dict(lambda_rd=0.1, lambda_bkg=0.1),
        test_cfg=dict(mode='whole'))
    compressor = build_compressor(cfg)
    _compressor_forward_train_test(compressor, mask=True)

    # test slide mode
    cfg.test_cfg = ConfigDict(mode='slide', crop_size=(3, 3), stride=(2, 2))
    compressor = build_compressor(cfg)
    _compressor_forward_train_test(compressor, mask=True)
