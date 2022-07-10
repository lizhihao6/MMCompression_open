import glob
import os
from os.path import dirname, exists, isdir, join, relpath

from mmcv import Config
from torch import nn

from mmcomp.models import build_compressor


def _get_config_directory():
    """Find the predefined compressor config directory."""
    try:
        # Assume we are running in the source mmcompressor repo
        repo_dpath = dirname(dirname(__file__))
    except NameError:
        # For IPython development when this __file__ is not defined
        import mmcomp
        repo_dpath = dirname(dirname(mmcomp.__file__))
    config_dpath = join(repo_dpath, 'configs')
    if not exists(config_dpath):
        raise Exception('Cannot find config path')
    return config_dpath


def test_config_build_compressor():
    """Test that all compressor models defined in the configs can be
    initialized."""
    config_dpath = _get_config_directory()
    print('Found config_dpath = {!r}'.format(config_dpath))

    config_fpaths = []
    # one config each sub folder
    for sub_folder in os.listdir(config_dpath):
        if isdir(sub_folder):
            config_fpaths.append(
                list(glob.glob(join(config_dpath, sub_folder, '*.py')))[0])
    config_fpaths = [p for p in config_fpaths if p.find('_base_') == -1]
    config_names = [relpath(p, config_dpath) for p in config_fpaths]

    print('Using {} config files'.format(len(config_names)))

    for config_fname in config_names:
        config_fpath = join(config_dpath, config_fname)
        config_mod = Config.fromfile(config_fpath)

        config_mod.model
        print('Building compressor, config_fpath = {!r}'.format(config_fpath))

        # Remove pretrained keys to allow for testing in an offline environment
        if 'pretrained' in config_mod.model:
            config_mod.model['pretrained'] = None

        print('building {}'.format(config_fname))
        compressor = build_compressor(config_mod.model)
        assert compressor is not None


def test_config_data_pipeline():
    """Test whether the data pipeline is valid and can process corner cases.

    CommandLine:
        xdoctest -m tests/test_config.py test_config_build_data_pipeline
    """
    from mmcv import Config
    from mmcomp.datasets.pipelines import Compose
    import numpy as np

    config_dpath = _get_config_directory()
    print('Found config_dpath = {!r}'.format(config_dpath))

    import glob
    config_fpaths = list(glob.glob(join(config_dpath, '**', '*.py')))
    config_fpaths = [p for p in config_fpaths if p.find('_base_') == -1]
    config_names = [relpath(p, config_dpath) for p in config_fpaths]

    print('Using {} config files'.format(len(config_names)))

    for config_fname in config_names:
        config_fpath = join(config_dpath, config_fname)
        print(
            'Building data pipeline, config_fpath = {!r}'.format(config_fpath))
        config_mod = Config.fromfile(config_fpath)

        # remove loading pipeline
        load_img_pipeline = config_mod.train_pipeline.pop(0)
        to_float32 = load_img_pipeline.get('to_float32', False)
        config_mod.train_pipeline.pop(0)
        config_mod.test_pipeline.pop(0)
        config_mod.test_pipeline.pop(0)

        train_pipeline = Compose(config_mod.train_pipeline)
        test_pipeline = Compose(config_mod.test_pipeline)

        img = np.random.randint(0, 255, size=(1024, 2048, 3), dtype=np.uint8)
        if to_float32:
            img = img.astype(np.float32)

        results = dict(
            filename='test_img.png',
            ori_filename='test_img.png',
            img=img,
            img_shape=img.shape,
            ori_shape=img.shape)

        print('Test training data pipeline: \n{!r}'.format(train_pipeline))
        output_results = train_pipeline(results)
        assert output_results is not None

        results = dict(
            filename='test_img.png',
            ori_filename='test_img.png',
            img=img,
            img_shape=img.shape,
            ori_shape=img.shape)

        print('Test testing data pipeline: \n{!r}'.format(test_pipeline))
        output_results = test_pipeline(results)
        assert output_results is not None
