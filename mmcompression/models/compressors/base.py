# Copyright (c) OpenMMLab. All rights reserved.
import os
import tempfile
import warnings
from abc import ABCMeta, abstractmethod
from collections import OrderedDict
from typing import Optional

import mmcv
import numpy as np
import torch
import torch.distributed as dist
from torch import nn

from mmcompression.core.evaluation.metrics import psnr


class BaseCompressor(nn.Module):
    """Base class for compressors."""

    __metaclass__ = ABCMeta

    def __init__(self):
        super().__init__()
        self.cache_files = []

    def _get_tmp_file(self, suffix: str = None):
        """
        Get a temporary file.
        Args:
            suffix (str): The suffix of the temporary file.
        Returns:
            str: The path of the temporary file.
        """
        tmp_file = os.path.join(
            tempfile.gettempdir(), next(tempfile._get_candidate_names())
        )
        tmp_file = tmp_file + f"_{os.getpid()}" + f".{suffix}"
        self.cache_files.append(tmp_file)
        return tmp_file

    def __del__(self):
        """
        Delete the temporary files.
        """
        for p in self.cache_files:
            try:
                os.remove(p)
            except OSError:
                pass
        self.cache_files.clear()

    @staticmethod
    def calculate_bits_num(bitstreams: dict):
        """Calculate the bits of the compressed image.
        Args:
            bitstreams (dict): The bitstreams of the compressed image.
        Returns:
            bits_num (int): The bits number of the compressed image.
        """
        bits_num = 0
        for k, v in bitstreams.items():
            bits_num += len(v) * 8
        return bits_num

    @staticmethod
    def calculate_psnr(img1: torch.Tensor, img2: torch.Tensor):
        """Calculate the PSNR between two images.
        Args:
            img1 (Tensor): The first image within shape of (C, H, W).
            img2 (Tensor): The second image within shape of (C, H, W).
            Typically, these should be scaled into zero to one.
        Returns:
            float: The PSNR between two images.
        """
        img1 = img1.detach().cpu().numpy() * 255.
        img2 = img2.detach().cpu().numpy() * 255.
        return psnr(img1, img2, input_order='CHW')

    def save_img(self, img: torch.Tensor):
        """Save the image to a temporary file.
        Args:
            img (Tensor): The image to be saved within shape of (C, H, W).
            Typically, these should be scaled into zero to one.
        Returns:
            str: The path of the temporary file.
        """
        tmp_file = self._get_tmp_file(suffix='png')
        img = img.detach().cpu().numpy() * 255.
        img = np.transpose(img, (1, 2, 0)).astype(np.uint8)
        # rgb to bgr
        img = img[..., ::-1]
        mmcv.imwrite(tmp_file, img)
        return tmp_file

    @abstractmethod
    def compress(self, img: Optional):
        """
        Args:
            img (Optional): np.ndarray or Tensor. The original images of shape (C, H, W).
                Typically, these should be scaled into zero to one within RGB format.
        Returns:
            bitstreams (dict):  The compressed bitstreams.
        """

    @abstractmethod
    def decompress(self, bitstreams: dict):
        """
        Args:
            bitstreams (dict): The compressed bitstreams.
        Returns:
            img (Optional): np.ndarray or Tensor. The recovered image of shape (C, H, W).
        """

    @abstractmethod
    def forward_train(self, img: torch.FloatTensor, img_metas: list, **kwargs):
        """
        Args:
            img (Tensor): The original images of shape (N, C, H, W).
                Typically, these should be scaled into zero to one within RGB format.
            img_metas (list[dict]): List of image info dict where each dict
                has: 'filename' and may also contain other keys.
                For details on the values of these keys, see
                :class:`mmcompress.datasets.pipelines.Collect`.
            kwargs (keyword arguments): Specific to concrete implementation.
        """

    @abstractmethod
    def forward_test(self, img: torch.FloatTensor, img_metas: list, return_image: bool = False, **kwargs):
        """
        Args:
            img (Tensor): The original images of shape (N, C, H, W).
                Typically, these should be scaled into zero to one within RGB format.
            img_metas (list[dict]): List of image info dict where each dict
                has: 'filename' and may also contain other keys.
                For details on the values of these keys, see
                :class:`mmcompress.datasets.pipelines.Collect`.
            return_image (bool): whether to return the image after compress-decompress.
            kwargs (keyword arguments): Specific to concrete implementation.
        """

    def forward(self, img: torch.FloatTensor, img_metas: list, return_loss=True, return_image=False, **kwargs):
        """
        Args:
            img (Tensor): The original images of shape (N, C, H, W).
                Typically, these should be scaled into zero to one.
            img_metas (list[dict]): List of image info dict where each dict
                has: 'filename' and may also contain other keys.
                For details on the values of these keys, see
                :class:`mmcompress.datasets.pipelines.Collect`.
            return_loss (bool): whether to return the loss.
            return_image (bool): whether to return the image after compress-decompress.
            kwargs (keyword arguments): Specific to concrete implementation.
        """
        if return_loss:
            return self.forward_train(img, img_metas, **kwargs)
        else:
            return self.forward_test(img, img_metas, return_image, **kwargs)

    def train_step(self, data_batch: dict, optimizer: torch.optim.Optimizer):
        """The iteration step during training.
        This method defines an iteration step during training, except for the
        back propagation and optimizer updating, which are done in an optimizer
        hook. Note that in some complicated cases or models, the whole process
        including back propagation and optimizer updating is also defined in
        this method, such as GAN.

        Args:
            data_batch (dict): The output of dataloader.
            optimizer (:obj:`torch.optim.Optimizer` | dict): The optimizer of
                runner is passed to ``train_step()``. This argument is unused
                and reserved.

        Returns:
            dict: It should contain at least 3 keys: ``loss``, ``log_vars``,
                ``num_samples``.
                ``loss`` is a tensor for back propagation, which can be a
                weighted sum of multiple losses.
                ``log_vars`` contains all the variables to be sent to the
                logger.
                ``num_samples`` indicates the batch size (when the model is
                DDP, it means the batch size on each GPU), which is used for
                averaging the logs.
        """
        losses = self(**data_batch)
        loss, log_vars = self._parse_losses(losses)

        outputs = dict(
            loss=loss,
            log_vars=log_vars,
            num_samples=len(data_batch['img_metas']))

        return outputs

    def val_step(self, data_batch: dict, **kwargs):
        """The iteration step during validation.
        This method shares the same signature as :func:`train_step`, but used
        during val epochs. Note that the evaluation after training epochs is
        not implemented with this method, but an evaluation hook.
        """
        output = self(**data_batch)
        return output

    @staticmethod
    def _parse_losses(losses):
        """Parse the raw outputs (losses) of the network.
        Args:
            losses (dict): Raw output of the network, which usually contain
                losses and other necessary information.
        Returns:
            tuple[Tensor, dict]: (loss, log_vars), loss is the loss tensor
                which may be a weighted sum of all losses, log_vars contains
                all the variables to be sent to the logger.
        """
        log_vars = OrderedDict()
        for loss_name, loss_value in losses.items():
            if isinstance(loss_value, torch.Tensor):
                log_vars[loss_name] = loss_value.mean()
            elif isinstance(loss_value, list):
                log_vars[loss_name] = sum(_loss.mean() for _loss in loss_value)
            else:
                raise TypeError(
                    f'{loss_name} is not a tensor or list of tensors')

        loss = sum(_value for _key, _value in log_vars.items()
                   if 'loss' in _key)

        log_vars['loss'] = loss
        for loss_name, loss_value in log_vars.items():
            # reduce loss when distributed training
            if dist.is_available() and dist.is_initialized():
                loss_value = loss_value.data.clone()
                dist.all_reduce(loss_value.div_(dist.get_world_size()))
            log_vars[loss_name] = loss_value.item()

        return loss, log_vars

    def show_result(self,
                    img,
                    rec_img,
                    win_name='',
                    show=False,
                    wait_time=0,
                    out_file=None):
        """Draw `result` over `img`.
        Args:
            img (str or Tensor): The original image.
            rec_img (str or Tensor): The recovered image.
            win_name (str): The window name.
            wait_time (int): Value of waitKey param.
                Default: 0.
            show (bool): Whether to show the image.
                Default: False.
            out_file (str or None): The filename to write the image.
                Default: None.
        Returns:
            img (Tensor): Only if not `show` or `out_file`
        """
        img = mmcv.imread(img)
        img = img.copy()
        rec_img = mmcv.imread(rec_img)
        rec_img = rec_img.copy()
        img = np.vstack([img, rec_img])

        # if out_file specified, do not show image in window
        if out_file is not None:
            show = False

        if show:
            mmcv.imshow(img, win_name, wait_time)
        if out_file is not None:
            mmcv.imwrite(img, out_file)

        if not (show or out_file):
            warnings.warn('show==False and out_file is not specified, only '
                          'result image will be returned')
            return img
