from abc import ABCMeta, abstractmethod
from collections import OrderedDict

import torch
import torch.distributed as dist
import torch.nn as nn


class BaseCompressor(nn.Module):
    """Base class for compressors."""

    __metaclass__ = ABCMeta

    def __init__(self):
        super(BaseCompressor, self).__init__()
        self.fp16_enabled = False

    @abstractmethod
    def forward_train(self, imgs, **kwargs):
        """Placeholder for Forward function for training."""
        pass

    @abstractmethod
    def forward_test(self, img, return_image, **kwargs):
        """Placeholder for Forward function for testing."""
        pass

    def init_weights(self, pretrained):
        """Placeholder for init weights"""
        if pretrained is not None:
            state_dict = torch.load(pretrained, map_location='cpu')
            if 'state_dict' in state_dict.keys():
                state_dict = state_dict['state_dict']
                print('Direct loading from {}'.format(pretrained))
            try:
                self.load_state_dict(state_dict, strict=True)
            except:
                pass
            updated_state_dict = OrderedDict()
            cache_init_keys = []
            non_same = {}
            for k in self.state_dict():
                if k not in state_dict.keys():
                    non_same[k] = (None, self.state_dict()[k].shape)
                elif self.state_dict()[k].shape == state_dict[k].shape:
                    if 'init' not in k:
                        updated_state_dict[k] = state_dict[k]
                    else:
                        cache_init_keys.append(k)
                else:
                    non_same[k] = (state_dict[k].shape, self.state_dict()[k].shape)
            if len(non_same.keys()) == 0:
                for k in cache_init_keys:
                    updated_state_dict[k] = state_dict[k]
            else:
                print('Not all state dict shape same')
                for k, v in non_same.items():
                    print('{}: {} -> {}'.format(k, v[0], v[1]))
            self.load_state_dict(updated_state_dict, strict=False)
        else:
            # todo add default init weights
            pass

    # @auto_fp16(apply_to=('img',))
    def forward(self, img, return_loss=True, return_image=False, **kwargs):
        """Calls either :func:`forward_train` or :func:`forward_test` depending
        on whether ``return_loss`` is ``True``.

        Note this setting will change the expected inputs. When
        ``return_loss=True``, img and img_meta are single-nested (i.e. Tensor
        and List[dict]), and when ``resturn_loss=False``, img and img_meta
        should be double nested (i.e.  List[Tensor], List[List[dict]]), with
        the outer list indicating test time augmentations.
        """
        if return_loss:
            return self.forward_train(img, **kwargs)
        else:
            return self.forward_test(img, return_image, **kwargs)

    def train_step(self, data_batch, optimizer, **kwargs):
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

        num_samples = len(data_batch['img'].data) if 'img' in data_batch.keys() else len(data_batch['frames'].data)
        outputs = dict(
            loss=loss,
            log_vars=log_vars,
            num_samples=num_samples)

        return outputs

    def val_step(self, data_batch, **kwargs):
        """The iteration step during validation.

        This method shares the same signature as :func:`train_step`, but used
        during val epochs. Note that the evaluation after training epochs is
        not implemented with this method, but an evaluation hook.
        """
        output = self(**data_batch, **kwargs)
        return output

    def _calculate_psnr(self, y, gt):
        """Calculate the PSNR.

        Args:
            y (tensor): Recover image with [B, C, H, W].
            gt (tensor): GT image with [B, C, H, W].

        Returns:
            psnr(tensor): value.
        """
        mse = (torch.clamp(y, 0, 1) - gt) ** 2
        mse = mse.view([mse.shape[0], -1]).mean(1)
        psnr = 10 * torch.log10(1. / mse)
        return psnr.mean()

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

    def show_result(self, data, result, show=False, out_file=None):
        raise NotImplementedError
