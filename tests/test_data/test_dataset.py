import os.path as osp
from unittest.mock import MagicMock, patch

from mmcompression.datasets import CustomDataset, ConcatDataset, RepeatDataset


@patch('mmcompression.datasets.CustomDataset.load_annotations', MagicMock)
@patch('mmcompression.datasets.CustomDataset.__getitem__',
       MagicMock(side_effect=lambda idx: idx))
def test_dataset_wrapper():
    # CustomDataset.load_annotations = MagicMock()
    # CustomDataset.__getitem__ = MagicMock(side_effect=lambda idx: idx)
    dataset_a = CustomDataset(img_dir=MagicMock(), pipeline=[])
    len_a = 10
    dataset_a.img_infos = MagicMock()
    dataset_a.img_infos.__len__.return_value = len_a
    dataset_b = CustomDataset(img_dir=MagicMock(), pipeline=[])
    len_b = 20
    dataset_b.img_infos = MagicMock()
    dataset_b.img_infos.__len__.return_value = len_b

    concat_dataset = ConcatDataset([dataset_a, dataset_b])
    assert concat_dataset[5] == 5
    assert concat_dataset[25] == 15
    assert len(concat_dataset) == len(dataset_a) + len(dataset_b)

    repeat_dataset = RepeatDataset(dataset_a, 10)
    assert repeat_dataset[5] == 5
    assert repeat_dataset[15] == 5
    assert repeat_dataset[27] == 7
    assert len(repeat_dataset) == 10 * len(dataset_a)


def test_custom_dataset():
    img_norm_cfg = dict(
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_rgb=True)
    crop_size = (512, 1024)
    train_pipeline = [
        dict(type='LoadImageFromFile'),
        dict(type='Resize', img_scale=(128, 256), ratio_range=(0.5, 2.0)),
        dict(type='RandomCrop', crop_size=crop_size),
        dict(type='RandomFlip', flip_ratio=0.5),
        dict(type='Normalize', **img_norm_cfg),
        dict(type='Pad', size=crop_size, pad_val=0),
        dict(type='DefaultFormatBundle'),
        dict(type='Collect', keys=['img']),
    ]
    test_pipeline = [
        dict(type='LoadImageFromFile'),
        dict(type='Normalize', **img_norm_cfg),
        dict(type='ImageToTensor', keys=['img']),
        dict(type='Collect', keys=['img']),
    ]

    # with img_dir
    train_dataset = CustomDataset(
        train_pipeline,
        data_root=osp.join(osp.dirname(__file__), '../data/pseudo_dataset'),
        img_dir='imgs/',
        img_suffix='img.jpg')
    assert len(train_dataset) == 5

    # with img_dir, split
    train_dataset = CustomDataset(
        train_pipeline,
        data_root=osp.join(osp.dirname(__file__), '../data/pseudo_dataset'),
        img_dir='imgs/',
        img_suffix='img.jpg',
        split='splits/train.txt')
    assert len(train_dataset) == 4

    # no data_root
    train_dataset = CustomDataset(
        train_pipeline,
        img_dir=osp.join(osp.dirname(__file__), '../data/pseudo_dataset/imgs'),
        img_suffix='img.jpg')
    assert len(train_dataset) == 5

    # with data_root but img_dir are abs path
    train_dataset = CustomDataset(
        train_pipeline,
        data_root=osp.join(osp.dirname(__file__), '../data/pseudo_dataset'),
        img_dir=osp.abspath(
            osp.join(osp.dirname(__file__), '../data/pseudo_dataset/imgs')),
        img_suffix='img.jpg')
    assert len(train_dataset) == 5

    test_dataset = CustomDataset(
        test_pipeline,
        img_dir=osp.join(osp.dirname(__file__), '../data/pseudo_dataset/imgs'),
        img_suffix='img.jpg')
    assert len(test_dataset) == 5

    # training data get
    train_data = train_dataset[0]
    assert isinstance(train_data, dict)

    # test data get
    test_data = test_dataset[0]
    assert isinstance(test_data, dict)

    # evaluation
    pseudo_results = [{'bpp': 0.2, 'psnr': 25}, {'bpp': 0.1, 'psnr': 30}]
    eval_results = train_dataset.evaluate(pseudo_results)
    assert isinstance(eval_results, dict)


@patch('mmcompression.datasets.CustomDataset.load_annotations', MagicMock)
@patch('mmcompression.datasets.CustomDataset.__getitem__',
       MagicMock(side_effect=lambda idx: idx))
def test_custom_dataset_random_palette_is_generated():
    dataset = CustomDataset(
        pipeline=[],
        img_dir=MagicMock(),
        split=MagicMock())


@patch('mmcompression.datasets.CustomDataset.load_annotations', MagicMock)
@patch('mmcompression.datasets.CustomDataset.__getitem__',
       MagicMock(side_effect=lambda idx: idx))
def test_custom_dataset_custom_palette():
    dataset = CustomDataset(
        pipeline=[],
        img_dir=MagicMock(),
        split=MagicMock())
