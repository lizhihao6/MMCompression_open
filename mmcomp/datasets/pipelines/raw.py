# Copyright (c) OpenMMLab. All rights reserved.
import json
import os.path as osp

import cv2
import numpy as np
import rawpy
from imageio import imread

from mmcomp.utils import rearrange
from ..builder import PIPELINES


@PIPELINES.register_module()
class LoadRAWFromFile:
    """Load an image from file.

    Required keys are "img_prefix" and "img_info" (a dict that must contain the
    key "filename"). Added or updated keys are "filename", "img", "img_shape",
    "ori_shape" (same as `img_shape`), "pad_shape" (same as `img_shape`),
    "scale_factor" (1.0) and "img_norm_cfg" (means=0 and stds=1).
    """

    def __call__(self, results):
        """Call functions to load image and get image meta information.

        Args:
            results (dict): Result dict from :obj:`mmdet.CustomDataset`.

        Returns:
            dict: The dict contains loaded image and meta information.
        """
        if results['img_prefix'] is not None:
            filename = osp.join(results['img_prefix'],
                                results['img_info']['filename'])
        else:
            filename = results['img_info']['filename']

        if filename.endswith('TIF'):
            img = imread(filename)
        else:
            with rawpy.imread(filename) as f:
                img = f.raw_image_visible.copy()

        results['filename'] = filename
        results['ori_filename'] = results['img_info']['filename']
        results['img'] = img.astype(np.float32)
        results['img_shape'] = img.shape
        results['ori_shape'] = img.shape
        results['pad_shape'] = img.shape
        results['scale_factor'] = 1.0
        results['img_norm_cfg'] = dict(
            mean=np.zeros(1, dtype=np.float32),
            std=np.ones(1, dtype=np.float32),
            to_rgb=False)

        with open(results['img_info']['meta_path'], 'r') as f:
            results = {**results, **json.load(f)}
        return results

    def __repr__(self):
        repr_str = f'{self.__class__.__name__}()'
        return repr_str


@PIPELINES.register_module()
class RAWNormalization:
    """Load an image from file.

    Required keys are "img_prefix" and "img_info" (a dict that must contain the
    key "filename"). Added or updated keys are "filename", "img", "img_shape",
    "ori_shape" (same as `img_shape`), "pad_shape" (same as `img_shape`),
    "scale_factor" (1.0) and "img_norm_cfg" (means=0 and stds=1).
    """

    def __call__(self, results):
        """Call functions to load image and get image meta information.

        Args:
            results (dict): Result dict from :obj:`mmdet.CustomDataset`.

        Returns:
            dict: The dict contains loaded image and meta information.
        """
        bl, wl = results['black_level'], results['white_level']
        results['img'] = np.clip((results['img'] - bl) / (wl - bl), 0, 1)
        return results

    def __repr__(self):
        repr_str = f'{self.__class__.__name__}()'
        return repr_str


@PIPELINES.register_module()
class Rearrange:
    """Rearrange RAW to four channels.
    """

    def __call__(self, results):
        """Call functions to load image and get image meta information.

        Args:
            results (dict): Result dict from :obj:`mmdet.CustomDataset`.

        Returns:
            dict: The dict contains loaded image and meta information.
        """
        img = rearrange(results['img'], results['bayer_pattern'])
        results['img'] = img
        results['img_shape'] = img.shape
        results['ori_shape'] = img.shape
        results['pad_shape'] = img.shape
        results['img_norm_cfg'] = dict(
            mean=np.zeros(4, dtype=np.float32),
            std=np.ones(4, dtype=np.float32),
            to_rgb=False)

        """Resize segmentation map with 0.5."""
        for key in results.get('seg_fields', []):
            results[key] = cv2.resize(results[key], cv2.INTER_NEAREST)
        return results

    def __repr__(self):
        repr_str = f'{self.__class__.__name__}()'
        return repr_str
