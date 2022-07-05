# Copyright (c) OpenMMLab. All rights reserved.
import os.path

import cv2
import mmcv
import numpy as np
from mmcv.parallel import DataContainer as DC

from .formating import DefaultFormatBundle, to_tensor, ImageToTensor
from .transforms import RandomCrop, RandomFlip, Normalize
from ..builder import PIPELINES


@PIPELINES.register_module()
class LoadFramesFromFile:
    """Load an image from file.

    Required keys are "frames_info" (a dict that must contain the
    key "frames" and "filename"). Added or updated keys are "filename",
    "frames", "frames_shape", "ori_shape" (same as `frames_shape`),
    "pad_shape" (same as `frames_shape`), "scale_factor"  (1.0) and
    "frames_norm_cfg" (means=0 and stds=1).

     Args:
        to_float32 (bool): Whether to convert the loaded image to a float32
            numpy array. If set to False, the loaded image is an uint8 array.
            Defaults to False.
    """

    def __init__(self,
                 to_float32=False):
        self.to_float32 = to_float32

    def __call__(self, results):
        """Call functions to load image and get image meta information.

        Args:
            results (dict): Result dict from :obj:`mmdet.CustomDataset`.

        Returns:
            dict: The dict contains loaded image and meta information.
        """
        prefix = results.get('frames_prefix', None)
        filename = results['frames_info']['filename']
        if not os.path.isabs(filename) and prefix is not None:
            filename = os.path.join(prefix, filename)
        results['filename'] = filename
        results['ori_filename'] = results['frames_info']['filename']

        frames = results['frames_info']['frames']
        if not os.path.isabs(frames[0]) and prefix is not None:
            frames = [os.path.join(prefix, f) for f in frames]
        frames = np.concatenate([cv2.imread(f)[None] for f in frames])
        if self.to_float32:
            frames = frames.astype(np.float32)

        results['frames'] = frames
        results['frames_shape'] = frames.shape
        results['ori_shape'] = frames.shape
        results['pad_shape'] = frames.shape
        results['scale_factor'] = 1.0
        results['frames_norm_cfg'] = dict(
            mean=np.zeros(1, dtype=np.float32),
            std=np.ones(1, dtype=np.float32),
            to_rgb=False)

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(to_float32={self.to_float32})'
        return repr_str


@PIPELINES.register_module()
class FramesRandomCrop(RandomCrop):
    """Random crop the frames & comp.

    Args:
        crop_size (tuple): Expected size after cropping, (h, w).
        cat_max_ratio (float): The maximum ratio1 that single category could
            occupy.
    """

    def crop(self, frames, crop_bbox):
        """Crop from ``img``"""
        crop_y1, crop_y2, crop_x1, crop_x2 = crop_bbox
        if len(frames.shape) == 3:
            frames = frames[crop_y1:crop_y2, crop_x1:crop_x2, ...]
        elif len(frames.shape) == 4:
            frames = frames[:, crop_y1:crop_y2, crop_x1:crop_x2, ...]
        else:
            raise IndexError
        return frames

    def __call__(self, results):
        """Call function to randomly crop images, semantic segmentation maps.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Randomly cropped results, 'frames_shape' key in result dict is
                updated according to crop size.
        """

        frames = results['frames']
        crop_bbox = self.get_crop_bbox(frames[0])
        if self.cat_max_ratio < 1. and 'gt_semantic_seg' in results.keys():
            # Repeat 10 times
            for _ in range(10):
                seg_temp = self.crop(results['gt_semantic_seg'], crop_bbox)
                labels, cnt = np.unique(seg_temp, return_counts=True)
                cnt = cnt[labels != self.ignore_index]
                if len(cnt) > 1 and np.max(cnt) / np.sum(
                        cnt) < self.cat_max_ratio:
                    break
                crop_bbox = self.get_crop_bbox(frames[0])

        # crop the image
        frames = self.crop(frames, crop_bbox)
        frames_shape = frames.shape
        results['frames'] = frames
        results['frames_shape'] = frames_shape

        # crop semantic comp
        if 'seg_fields' in results.keys():
            for key in results.get('seg_fields', []):
                results[key] = self.crop(results[key], crop_bbox)

        return results


@PIPELINES.register_module()
class FramesRandomFlip(RandomFlip):
    """Flip the frames & comp.

    If the input dict contains the key "flip", then the flag will be used,
    otherwise it will be randomly decided by a ratio specified in the init
    method.

    Args:
        prob (float, optional): The flipping probability. Default: None.
        direction(str, optional): The flipping direction. Options are
            'horizontal' and 'vertical'. Default: 'horizontal'.
    """

    def __call__(self, results):
        """Call function to flip bounding boxes, masks, semantic segmentation
        maps.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Flipped results, 'flip', 'flip_direction' keys are added into
                result dict.
        """

        if 'flip' not in results:
            flip = True if np.random.rand() < self.prob else False
            results['flip'] = flip
        if 'flip_direction' not in results:
            results['flip_direction'] = self.direction
        if results['flip']:
            # flip frames
            if results['flip_direction'] == 'horizontal':
                results['frames'] = results['frames'][:, :, ::-1, :]
            else:
                results['frames'] = results['frames'][:, ::-1, :, :]
            # flip segs
            if 'seg_fields' in results.keys():
                for key in results.get('seg_fields', []):
                    # use copy() to make numpy stride positive
                    results[key] = mmcv.imflip(
                        results[key], direction=results['flip_direction']).copy()
        return results


@PIPELINES.register_module()
class FramesNormalize(Normalize):
    # Will auto convert uint8 to float32
    """Normalize the image.

    Added key is "frames_norm_cfg".

    Args:
        mean (sequence): Mean values of 3 channels.
        std (sequence): Std values of 3 channels.
        to_rgb (bool): Whether to convert the image from BGR to RGB,
            default is true.
    """

    def __call__(self, results):
        """Call function to normalize images.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Normalized results, 'img_norm_cfg' key is added into
                result dict.
        """
        mean, std = self.mean[None, None, None], self.std[None, None, None]
        results['frames'] = (results['frames'].astype(np.float32) - mean) / std
        if self.to_rgb:
            results['frames'] = results['frames'][..., ::-1]

        results['frames_norm_cfg'] = dict(
            mean=self.mean, std=self.std, to_rgb=self.to_rgb)
        return results


@PIPELINES.register_module()
class FramesFormatBundle(DefaultFormatBundle):
    """Default formatting bundle.

    It simplifies the pipeline of formatting common fields, including "frames"
    and "gt_semantic_seg". These fields are formatted as follows.

    - frames: (1)transpose, (2)to tensor, (3)to DataContainer (stack=True)
    - gt_semantic_seg: (1)unsqueeze dim-0 (2)to tensor,
                       (3)to DataContainer (stack=True)
    """

    def __call__(self, results):
        """Call function to transform and format common fields in results.

        Args:
            results (dict): Result dict contains the data to convert.

        Returns:
            dict: The result dict contains the data that is formatted with
                default bundle.
        """

        frames = results['frames']
        frames = np.ascontiguousarray(frames.transpose(0, 3, 1, 2))
        results['frames'] = DC(to_tensor(frames), stack=True)
        if 'mask' in results:
            results['mask'] = DC(to_tensor(results['mask'][None]).bool().float(), stack=True)
        return results


@PIPELINES.register_module()
class FramesToTensor(ImageToTensor):
    """Convert frames to :obj:`torch.Tensor` by given keys.

    The dimension order of input image is (F, H, W, C). The pipeline will convert
    it to (F, C, H, W).

    Args:
        keys (Sequence[str]): Key of images to be converted to Tensor.
    """

    def __call__(self, results):
        """Call function to convert image in results to :obj:`torch.Tensor` and
        transpose the channel order.

        Args:
            results (dict): Result dict contains the image data to convert.

        Returns:
            dict: The result dict contains the image converted
                to :obj:`torch.Tensor` and transposed to (C, H, W) order.
        """

        for key in self.keys:
            frames = results[key]
            results[key] = to_tensor(frames.transpose(0, 3, 1, 2).copy())
        return results
