# Copyright (c) OpenMMLab. All rights reserved.
from unittest.mock import MagicMock

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from mmcompression.apis import single_gpu_test


def collate_fn(data):
    img = [d['img'][None] for d in data]
    img_metas = [[{'ori_filename': '../test/color.jpg'}] for _ in data]
    return {'img': torch.cat(img, 0), 'img_metas': img_metas}


class ExampleDataset(Dataset):

    def __getitem__(self, idx):
        results = dict(img=torch.tensor([1]), img_metas=dict())
        return results

    def __len__(self):
        return 1


class ExampleModel(nn.Module):

    def __init__(self):
        super(ExampleModel, self).__init__()
        self.test_cfg = None
        self.conv = nn.Conv2d(3, 3, 3)

    def forward(self, img, img_metas, return_loss=False, **kwargs):
        return img


def test_single_gpu():
    test_dataset = ExampleDataset()
    data_loader = DataLoader(
        test_dataset,
        batch_size=1,
        sampler=None,
        num_workers=0,
        shuffle=False,
        collate_fn=collate_fn
    )
    model = ExampleModel()

    # Test format_only
    test_dataset.format_results = MagicMock(return_value=['success'])
    results = single_gpu_test(model, data_loader, show=False)
    assert results[0][0, 0] == 1
