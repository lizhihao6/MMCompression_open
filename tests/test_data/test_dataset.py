import os.path as osp
from unittest.mock import MagicMock, patch

import pytest
import torch

from mmcomp.datasets import DATASETS, CustomDataset, ConcatDataset, RepeatDataset


@patch('mmcomp.datasets.CustomDataset.load_annotations', MagicMock)
@patch('mmcomp.datasets.CustomDataset.__getitem__',
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
        dict(type='LoadAnnotations'),
        dict(type='Resize', img_scale=(128, 256), ratio_range=(0.5, 2.0)),
        dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
        dict(type='RandomFlip', prob=0.5),
        dict(type='PhotoMetricDistortion'),
        dict(type='Normalize', **img_norm_cfg),
        dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=255),
        dict(type='DefaultFormatBundle'),
        dict(type='Collect', keys=['img', 'gt_semantic_seg']),
    ]
    test_pipeline = [
        dict(type='LoadImageFromFile'),
        dict(
            type='TestTimePipeline',
            img_scale=(128, 256),
            # img_ratios=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75],
            flip=False,
            transforms=[
                dict(type='Resize', keep_ratio=True),
                dict(type='Normalize', **img_norm_cfg),
                dict(type='ImageToTensor', keys=['img']),
                dict(type='Collect', keys=['img']),
            ])
    ]

    # with img_dir and ann_dir
    train_dataset = CustomDataset(
        train_pipeline,
        data_root=osp.join(osp.dirname(__file__), '../data/pseudo_dataset'),
        img_dir='imgs/',
        ann_dir='gts/',
        img_suffix='img.jpg',
        seg_map_suffix='gt.png')
    assert len(train_dataset) == 5

    # with img_dir, ann_dir, split
    train_dataset = CustomDataset(
        train_pipeline,
        data_root=osp.join(osp.dirname(__file__), '../data/pseudo_dataset'),
        img_dir='imgs/',
        ann_dir='gts/',
        img_suffix='img.jpg',
        seg_map_suffix='gt.png',
        split='splits/train.txt')
    assert len(train_dataset) == 4

    # no data_root
    train_dataset = CustomDataset(
        train_pipeline,
        img_dir=osp.join(osp.dirname(__file__), '../data/pseudo_dataset/imgs'),
        ann_dir=osp.join(osp.dirname(__file__), '../data/pseudo_dataset/gts'),
        img_suffix='img.jpg',
        seg_map_suffix='gt.png')
    assert len(train_dataset) == 5

    # with data_root but img_dir/ann_dir are abs path
    train_dataset = CustomDataset(
        train_pipeline,
        data_root=osp.join(osp.dirname(__file__), '../data/pseudo_dataset'),
        img_dir=osp.abspath(
            osp.join(osp.dirname(__file__), '../data/pseudo_dataset/imgs')),
        ann_dir=osp.abspath(
            osp.join(osp.dirname(__file__), '../data/pseudo_dataset/gts')),
        img_suffix='img.jpg',
        seg_map_suffix='gt.png')
    assert len(train_dataset) == 5

    # test_mode=True
    test_dataset = CustomDataset(
        test_pipeline,
        img_dir=osp.join(osp.dirname(__file__), '../data/pseudo_dataset/imgs'),
        img_suffix='img.jpg',
        test_mode=True)
    assert len(test_dataset) == 5

    # training data get
    train_data = train_dataset[0]
    assert isinstance(train_data, dict)

    # test data get
    test_data = test_dataset[0]
    assert isinstance(test_data, dict)

    # get gt comp map
    gt_seg_maps = train_dataset.get_gt_seg_maps()
    assert len(gt_seg_maps) == 5

    # evaluation
    pseudo_results = [torch.randn([1]), torch.randn([1]), torch.randn([1]), torch.randn([1])]
    eval_results = train_dataset.evaluate(pseudo_results)
    assert isinstance(eval_results, dict)


@patch('mmcomp.datasets.CustomDataset.load_annotations', MagicMock)
@patch('mmcomp.datasets.CustomDataset.__getitem__',
       MagicMock(side_effect=lambda idx: idx))
@pytest.mark.parametrize('dataset', [
    ('IphonescapesDataset'),
])
def test_custom_classes_override_default(dataset):
    dataset_class = DATASETS.get(dataset)

    # Test setting classes as a tuple
    custom_dataset = dataset_class(
        pipeline=[],
        img_dir=MagicMock(),
        split=MagicMock(),
        test_mode=True)

    # Test setting classes as a list
    custom_dataset = dataset_class(
        pipeline=[],
        img_dir=MagicMock(),
        split=MagicMock(),
        test_mode=True)

    # Test overriding not a subset
    custom_dataset = dataset_class(
        pipeline=[],
        img_dir=MagicMock(),
        split=MagicMock(),
        test_mode=True)

    # Test default behavior
    custom_dataset = dataset_class(
        pipeline=[],
        img_dir=MagicMock(),
        split=MagicMock(),
        test_mode=True)


@patch('mmcomp.datasets.CustomDataset.load_annotations', MagicMock)
@patch('mmcomp.datasets.CustomDataset.__getitem__',
       MagicMock(side_effect=lambda idx: idx))
def test_custom_dataset_random_palette_is_generated():
    dataset = CustomDataset(
        pipeline=[],
        img_dir=MagicMock(),
        split=MagicMock(),
        test_mode=True)


@patch('mmcomp.datasets.CustomDataset.load_annotations', MagicMock)
@patch('mmcomp.datasets.CustomDataset.__getitem__',
       MagicMock(side_effect=lambda idx: idx))
def test_custom_dataset_custom_palette():
    dataset = CustomDataset(
        pipeline=[],
        img_dir=MagicMock(),
        split=MagicMock(),
        test_mode=True)
