# Copyright (c) OpenMMLab. All rights reserved.
from mmcv import ConfigDict

from mmcompression.models import build_compressor
from .utils import _compressor_forward_train_test


def test_jpeg():
    # test 1 decode head, w.o. aux head

    cfg = ConfigDict(
        type='JPEGCompressor',
        qp=10,
        train_cfg=None,
        test_cfg=None)
    compressor = build_compressor(cfg)
    _compressor_forward_train_test(compressor)
