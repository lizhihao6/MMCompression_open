"""pytest tests/test_forward.py."""
import copy
from os.path import dirname, exists, join
from unittest.mock import patch

import numpy as np
import torch
from mmcv.utils.parrots_wrapper import SyncBatchNorm, _BatchNorm


def _demo_mm_inputs(input_shape=(2, 3, 8, 16)):
    """Create a superset of inputs needed to run test or train batches.

    Args:
        input_shape (tuple):
            input batch dimensions

        num_classes (int):
            number of semantic classes
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
        'img_metas': img_metas,
    }
    return mm_inputs


def _get_config_directory():
    """Find the predefined segmentor config directory."""
    try:
        # Assume we are running in the source mmsegmentation repo
        repo_dpath = dirname(dirname(dirname(__file__)))
    except NameError:
        # For IPython development when this __file__ is not defined
        import mmseg
        repo_dpath = dirname(dirname(dirname(mmseg.__file__)))
    config_dpath = join(repo_dpath, 'configs')
    if not exists(config_dpath):
        raise Exception('Cannot find config path')
    return config_dpath


def _get_config_module(fname):
    """Load a configuration as a python module."""
    from mmcv import Config
    config_dpath = _get_config_directory()
    config_fpath = join(config_dpath, fname)
    config_mod = Config.fromfile(config_fpath)
    return config_mod


def _get_compressor_cfg(fname):
    """Grab configs necessary to create a segmentor.

    These are deep copied to allow for safe modification of parameters without
    influencing other tests.
    """
    config = _get_config_module(fname)
    model = copy.deepcopy(config.model)
    return model


def test_nlaic_forward():
    _test_compressor_forward(
        'nlaic/nlaic_flicker2w_100k_qp_1.py', test_train=True)


def test_tinylic_forward():
    _test_compressor_forward(
        'tinylic/tinylic_flicker2w_200epochs_qp_1.py', test_train=True)


def test_jpeg_forward():
    _test_compressor_forward(
        'jpeg/jpeg_IEEE1857_qp_92.py', test_train=False)


# def test_hm_forward():
#     _test_compressor_forward(
#         'mpeg/hm_IEEE1857_qp_42.py', test_train=False)

# def test_vvc_forward():
#     _test_compressor_forward(
#         'mpeg/vvc_IEEE1857_qp_42.py', test_train=False)


def get_world_size(process_group):
    return 1


def _check_input_dim(self, inputs):
    pass


def _convert_batchnorm(module):
    module_output = module
    if isinstance(module, SyncBatchNorm):
        # to be consistent with SyncBN, we hack dim check function in BN
        module_output = _BatchNorm(module.num_features, module.eps,
                                   module.momentum, module.affine,
                                   module.track_running_stats)
        if module.affine:
            module_output.weight.data = module.weight.data.clone().detach()
            module_output.bias.data = module.bias.data.clone().detach()
            # keep requires_grad unchanged
            module_output.weight.requires_grad = module.weight.requires_grad
            module_output.bias.requires_grad = module.bias.requires_grad
        module_output.running_mean = module.running_mean
        module_output.running_var = module.running_var
        module_output.num_batches_tracked = module.num_batches_tracked
    for name, child in module.named_children():
        module_output.add_module(name, _convert_batchnorm(child))
    del module
    return module_output


@patch('torch.nn.modules.batchnorm._BatchNorm._check_input_dim',
       _check_input_dim)
@patch('torch.distributed.get_world_size', get_world_size)
def _test_compressor_forward(cfg_file, test_train=True):
    model = _get_compressor_cfg(cfg_file)
    model['pretrained'] = None
    model['test_cfg']['mode'] = 'whole'

    from mmcompression.models import build_compressor
    compressor = build_compressor(model)

    # batch_size=2 for BatchNorm
    input_shape = (2, 3, 256, 256)
    mm_inputs = _demo_mm_inputs(input_shape)

    imgs = mm_inputs.pop('imgs')
    img_metas = mm_inputs.pop('img_metas')

    # convert to cuda Tensor if applicable
    if torch.cuda.is_available():
        compressor = compressor.cuda()
        imgs = imgs.cuda()
    else:
        compressor = _convert_batchnorm(compressor)

    # Test forward train
    if test_train:
        losses = compressor.forward(imgs, img_metas, return_loss=True)
        assert isinstance(losses, dict)

    # Test forward simple test
    imgs = imgs[0].unsqueeze(0)
    img_metas = [img_metas[0]]

    with torch.no_grad():
        compressor.eval()
        compressor.forward(imgs, img_metas, return_loss=False)
