# Copyright (c) NJU Vision Lab. All rights reserved.
import os
import shutil
import tempfile
from abc import ABCMeta, abstractmethod
from collections import OrderedDict

import cv2
import numpy as np
import torch
import torch.distributed as dist
from mmcv.utils import print_log
from torch import nn

from mmcomp.utils import get_root_logger


class BaseCompressor(nn.Module):
    """Base class for compressors."""

    __metaclass__ = ABCMeta

    def __init__(self):
        super().__init__()
        self.cache_files = []

    def _get_tmp_file(self, suffix=None):
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

    @abstractmethod
    def forward_train(self, img, img_metas, **kwargs):
        """
        Args:
            img (Tensor): of shape (N, C, H, W) encoding input images.
                Typically these should be mean centered and std scaled.
            img_metas (list[dict]): List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys, see
                :class:`mmdet.datasets.pipelines.Collect`.
            kwargs (keyword arguments): Specific to concrete implementation.
        """

    @abstractmethod
    def forward_test(self, img, img_metas, return_image, **kwargs):
        """
        Args:
            img (Tensor): of shape (N, C, H, W) encoding input images.
                Typically these should be mean centered and std scaled.
            img_metas (list[dict]): List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys, see
                :class:`mmdet.datasets.pipelines.Collect`.
            return_image (bool): whether to return the image after enc-dec.
            kwargs (keyword arguments): Specific to concrete implementation.
        """

    def init_weights(self, pretrained):
        """Initialize the weights in network.
        pretrained (str): the path of pretrained model
        """
        logger = get_root_logger()
        if pretrained is None:
            return

        print_log(f"Loading from {pretrained}", logger=logger)
        state_dict = torch.load(pretrained, map_location="cpu")
        if "state_dict" in state_dict.keys():
            state_dict = state_dict["state_dict"]

        updated_state_dict = OrderedDict()
        missing_keys = []
        miss_match_keys = {}
        for k in self.state_dict():
            if k not in state_dict.keys():
                missing_keys.append(k)
            elif self.state_dict()[k].shape != state_dict[k].shape:
                miss_match_keys[k] = (state_dict[k].shape, self.state_dict()[k].shape)
            else:
                updated_state_dict[k] = state_dict[k]

        if len(missing_keys) > 0:
            print_log(
                f"Weights of {missing_keys} not initialized from pretrained model: {pretrained}",
                logger=logger,
                level=logger.WARNING,
            )

        if len(miss_match_keys) > 0:
            for k, v in miss_match_keys.items():
                print_log(
                    f"Shape of {k} mismatch: {v[0]} in pretrained model vs {v[1]} in model",
                    logger=logger,
                    level=logger.WARNING,
                )

        self.load_state_dict(updated_state_dict, strict=False)

    def forward(self, img, img_metas, return_loss=True, return_image=False, **kwargs):
        """
        Args:
            imgs (List[Tensor]): the outer list indicates test-time
                augmentations and inner Tensor should have a shape NxCxHxW,
                which contains all images in the batch.
            img_metas (List[List[dict]]): the outer list indicates test-time
                augs (multiscale, flip, etc.) and the inner list indicates
                images in a batch.
            return_loss (bool): whether to return the loss.
            return_image (bool): whether to return the image after enc-dec.
            kwargs (keyword arguments): Specific to concrete implementation.
        """
        if return_loss:
            return self.forward_train(img, img_metas, **kwargs)
        else:
            return self.forward_test(img, img_metas, return_image, **kwargs)

    def train_step(self, data, optimizer):
        """The iteration step during training.
        This method defines an iteration step during training, except for the
        back propagation and optimizer updating, which are done in an optimizer
        hook. Note that in some complicated cases or models, the whole process
        including back propagation and optimizer updating is also defined in
        this method, such as GAN.
        Args:
            data (dict): The output of dataloader.
            optimizer (:obj:`torch.optim.Optimizer` | dict): The optimizer of
                runner is passed to ``train_step()``. This argument is unused
                and reserved.
        Returns:
            dict: It should contain at least 3 keys: ``loss``, ``log_vars``, \
                ``num_samples``.
                - ``loss`` is a tensor for back propagation, which can be a
                  weighted sum of multiple losses.
                - ``log_vars`` contains all the variables to be sent to the
                  logger.
                - ``num_samples`` indicates the batch size (when the model is
                  DDP, it means the batch size on each GPU), which is used for
                  averaging the logs.
        """
        losses = self(**data)
        loss, log_vars = self._parse_losses(losses)

        outputs = dict(loss=loss, log_vars=log_vars, num_samples=len(data["img_metas"]))

        return outputs

    def val_step(self, data, optimizer=None):
        """The iteration step during validation.
        This method shares the same signature as :func:`train_step`, but used
        during val epochs. Note that the evaluation after training epochs is
        not implemented with this method, but an evaluation hook.
        """
        losses = self(**data)
        loss, log_vars = self._parse_losses(losses)

        outputs = dict(loss=loss, log_vars=log_vars, num_samples=len(data["img_metas"]))

        return outputs

    @staticmethod
    def _calculate_psnr(y, gt):
        """Calculate the PSNR.

        Args:
            y (tensor): Recover image with [B, C, H, W].
            gt (tensor): GT image with [B, C, H, W].

        Returns:
            psnr(tensor): value.
        """
        mse = (torch.clamp(y, 0, 1) - gt) ** 2
        mse = mse.view([mse.shape[0], -1]).mean(1)
        psnr = 10 * torch.log10(1.0 / mse)
        return psnr.mean()

    @staticmethod
    def _parse_losses(losses):
        """Parse the raw outputs (losses) of the network.
        Args:
            losses (dict): Raw output of the network, which usually contain
                losses and other necessary information.
        Returns:
            tuple[Tensor, dict]: (loss, log_vars), loss is the loss tensor \
                which may be a weighted sum of all losses, log_vars contains \
                all the variables to be sent to the logger.
        """
        log_vars = OrderedDict()
        for loss_name, loss_value in losses.items():
            if isinstance(loss_value, torch.Tensor):
                log_vars[loss_name] = loss_value.mean()
            elif isinstance(loss_value, list):
                log_vars[loss_name] = sum(_loss.mean() for _loss in loss_value)
            else:
                raise TypeError(f"{loss_name} is not a tensor or list of tensors")

        loss = sum(_value for _key, _value in log_vars.items() if "loss" in _key)

        # If the loss_vars has different length, GPUs will wait infinitely
        if dist.is_available() and dist.is_initialized():
            log_var_length = torch.tensor(len(log_vars), device=loss.device)
            dist.all_reduce(log_var_length)
            message = (
                    f"rank {dist.get_rank()}"
                    + f" len(log_vars): {len(log_vars)}"
                    + " keys: "
                    + ",".join(log_vars.keys())
            )
            assert log_var_length == len(log_vars) * dist.get_world_size(), (
                    "loss log variables are different across GPUs!\n" + message
            )

        log_vars["loss"] = loss
        for loss_name, loss_value in log_vars.items():
            # reduce loss when distributed training
            if dist.is_available() and dist.is_initialized():
                loss_value = loss_value.data.clone()
                dist.all_reduce(loss_value.div_(dist.get_world_size()))
            log_vars[loss_name] = loss_value.item()

        return loss, log_vars

    @staticmethod
    def show_result(result, show=False, out_file=None):
        """
        Show the result of the compression.
        Args:
            result (tensor): The compressed result.
            show (bool): Whether to show the result.
            out_file (str): The path to save the result.
        """
        if show:
            concat_img = np.concatenate(
                [cv2.imread(result["ori_img"]), cv2.imread(result["rec_img"])], axis=1
            )
            cv2.imshow("left:original right:recovery", concat_img)

        if out_file is not None:
            suffix = out_file.split(".")[-1]
            assert suffix == "png"
            shutil.move(result["ori_img"], out_file[:-3] + "gt.png")
            shutil.move(result["rec_img"], out_file)
