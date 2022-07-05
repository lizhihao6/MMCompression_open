import warnings

from mmcv.utils import Registry, build_from_cfg
from torch import nn

CONTEXT = Registry('context_model')
ENTROPY = Registry('entropy_model')
HYPERENCODER = Registry('hyper_encoder')
HYPERDECODER = Registry('hyper_decoder')
MAINENCODER = Registry('main_encoder')
MAINDECODER = Registry('main_decoder')
LOSSES = Registry('loss')
QUANTS = Registry('quant')
COMPRESSOR = Registry('compressor')


def build(cfg, registry, default_args=None):
    """Build a module.

    Args:
        cfg (dict, list[dict]): The config of modules, is is either a dict
            or a list of configs.
        registry (:obj:`Registry`): A registry the module belongs to.
        default_args (dict, optional): Default arguments to build the module.
            Defaults to None.

    Returns:
        nn.Module: A built nn module.
    """

    if isinstance(cfg, list):
        modules = [
            build_from_cfg(cfg_, registry, default_args) for cfg_ in cfg
        ]
        return nn.Sequential(*modules)
    else:
        return build_from_cfg(cfg, registry, default_args)


def build_context_model(cfg):
    """Build context_model."""
    return build(cfg, CONTEXT)


def build_entropy_model(cfg):
    """Build entropy model."""
    return build(cfg, ENTROPY)


def build_hyper_encoder(cfg):
    """Build hyper encoder."""
    return build(cfg, HYPERENCODER)


def build_hyper_decoder(cfg):
    """Build hyper decoder."""
    return build(cfg, HYPERDECODER)


def build_main_encoder(cfg):
    """Build main encoder."""
    return build(cfg, MAINENCODER)


def build_main_decoder(cfg):
    """Build main decoder."""
    return build(cfg, MAINDECODER)


def build_loss(cfg):
    """Build loss."""
    return build(cfg, LOSSES)


def build_quant(cfg):
    """Build quant."""
    return build(cfg, QUANTS)


def build_compressor(cfg, train_cfg=None, test_cfg=None):
    """Build segmentor."""
    if train_cfg is not None or test_cfg is not None:
        warnings.warn(
            'train_cfg and test_cfg is deprecated, '
            'please specify them in model', UserWarning)
    assert cfg.get('train_cfg') is None or train_cfg is None, \
        'train_cfg specified in both outer field and model field '
    assert cfg.get('test_cfg') is None or test_cfg is None, \
        'test_cfg specified in both outer field and model field '
    return build(cfg, COMPRESSOR, dict(train_cfg=train_cfg, test_cfg=test_cfg))
