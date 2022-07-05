import os

import cv2
import torch
import torch.nn.functional as F
from mmcv.cnn.utils.weight_init import constant_init
from skvideo import setFFmpegPath

assert os.path.exists('/usr/bin/ffmpeg'), 'apt install ffmpeg'
setFFmpegPath('/usr/bin/')
from skvideo.io import FFmpegWriter
from .base import BaseCompressor
from .nic import NICCompressor
from .. import builder
from ..builder import COMPRESSOR, build_compressor
from mmcomp.utils import tensor_to_image


@COMPRESSOR.register_module()
class NVCInterCompressor(NICCompressor):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.lstm_memory = (None, None, None)
        self.num_output = self.main_decoder.num_output

    def clear_cache(self):
        self.lstm_memory = (None, None, None)

    def _compression(self, img, pre_img, rec_pre_img):
        features = self.main_encoder(img, pre_img)
        hyper = self.hyper_encoder(features)
        if self.training:
            features_quant = self.quant_model(features)
            hyper_quant = self.quant_model(hyper)
        else:
            features_quant = torch.round(features)
            hyper_quant = torch.round(hyper)
        features_prob = self.hyper_decoder(hyper_quant)
        rec_img_list, self.lstm_memory = self.main_decoder(features_quant, rec_pre_img, self.lstm_memory)
        main_prob = self.context_model(features_quant, features_prob)
        hyper_prob = self.entropy_model(hyper_quant)
        num_pixels = img.shape[0] * img.shape[2] * img.shape[3]
        k = -1. / torch.log(torch.FloatTensor([2])) / num_pixels
        main_bpp = torch.sum(torch.log(main_prob)) * k.to(main_prob.device)
        hyper_bpp = torch.sum(torch.log(hyper_prob)) * k.to(hyper_prob.device)
        return rec_img_list, main_bpp, hyper_bpp

    def init_weights(self, pretrained):
        super().init_weights(pretrained)

        # avoid init_cfg overwrite the initialization of `conv_offset`
        init_offset = True if pretrained is None else len(
            [k for k in torch.load(pretrained).keys() if 'conv_offset' in k]) == 0
        if init_offset:
            # avoid re-init when fine-tuning
            for m in self.modules():
                # DeformConv2dPack, ModulatedDeformConv2dPack
                if hasattr(m, 'conv_offset'):
                    constant_init(m.conv_offset, 0)


@COMPRESSOR.register_module()
class NVCCompressor(BaseCompressor):
    """Our Neural Image Compression
    """

    def __init__(self,
                 intra_cfg,
                 inter_cfg,
                 residual_cfg,
                 rec_loss,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super().__init__()
        self.intra_compressor = build_compressor(intra_cfg)
        self.inter_compressor = build_compressor(inter_cfg)
        self.use_residual = (residual_cfg is not None)
        if self.use_residual:
            self.residual_compressor = build_compressor(residual_cfg)

        self.rec_loss_fn = builder.build_loss(rec_loss)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.init_weights(pretrained=pretrained)

        self.num_output = self.inter_compressor.num_output
        if self.use_residual:
            self.num_output += 1

        for param in self.intra_compressor.parameters():
            param.requires_grad = False

    def _compression(self, frames, **kwargs):
        self.inter_compressor.clear_cache()
        rec_frames_list = [torch.zeros_like(frames) for _ in range(self.num_output)]

        # intra coding
        rec_img, main_bpp, hyper_bpp = self.intra_compressor._compression(frames[:, 0].detach())
        pre_img, rec_pre_img = frames[:, 0], rec_img
        for rec_frames in rec_frames_list:
            rec_frames[:, 0] = rec_img

        # inter coding
        for i, img in enumerate(frames[:, 1:].transpose(0, 1).detach()):
            # motion coding
            rec_img_list, inter_main_bpp, inter_hyper_bpp = self.inter_compressor._compression(img.clone().detach(),
                                                                                               pre_img.detach(),
                                                                                               rec_pre_img.detach())
            main_bpp, hyper_bpp = main_bpp + inter_main_bpp, hyper_bpp + inter_hyper_bpp
            # residual coding
            if self.use_residual:
                residual = rec_img_list[-1] - img
                rec_residual, residual_main_bpp, residual_hyper_bpp = self.residual_compressor._compression(
                    residual.detach())
                rec_img = rec_img_list[-1].detach() + rec_residual
                main_bpp, hyper_bpp = main_bpp + residual_main_bpp, hyper_bpp + residual_hyper_bpp
                rec_img_list.append(rec_img)

            for j, rec_img in enumerate(rec_img_list):
                rec_frames_list[j][:, i + 1] = rec_img

            pre_img, rec_pre_img = img, rec_frames_list[-1][:, i + 1]

        main_bpp, hyper_bpp = main_bpp / frames.shape[1], hyper_bpp / frames.shape[1]

        if not self.training:
            rec_frames = rec_frames_list[-1]
            rec_frames_list.clear()
            rec_frames_list = [rec_frames]

        return rec_frames_list, main_bpp, hyper_bpp

    def forward_train(self, frames, **kwargs):
        """Forward function for training.

        Args:
            frames (Tensor): Input frames.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        losses = dict()
        if self.train_cfg.pretrain:
            # pretrain inter
            frames = frames[:, :2]
            rec_frames_list, main_bpp, hyper_bpp = self._compression(frames, **kwargs)
            r = self.train_cfg.lambda_pretrain_rd
            losses['rec_loss'] = r * sum([self.rec_loss_fn(f, frames) for f in rec_frames_list])
        else:
            # training
            rec_frames_list, main_bpp, hyper_bpp = self._compression(frames, **kwargs)
            r = self.train_cfg.lambda_rd
            r_list = [r * 0.01 for _ in range(self.num_output - 1)] + [r]
            losses['rec_loss'] = sum([r * self.rec_loss_fn(f, frames) for r, f in zip(r_list, rec_frames_list)])

        losses["main_bpp_loss"] = 0.01 * main_bpp * frames.shape[1]
        losses["hyper_bpp_loss"] = 0.01 * hyper_bpp * frames.shape[1]
        losses["psnr"] = self._calculate_psnr(rec_frames_list[-1], frames)
        losses["bpp"] = main_bpp + hyper_bpp
        return losses

    # TODO refactor
    def slide_inference(self, frames, **kwargs):
        """Inference by sliding-window with overlap.

        If h_crop > h_img or w_crop > w_img, the small patch will be used to
        decode without padding.
        """
        h_stride, w_stride = self.test_cfg.stride
        h_crop, w_crop = self.test_cfg.crop_size
        batch_size, n_frames, c_img, h_img, w_img = frames.size()
        h_grids = max(h_img - h_crop + h_stride - 1, 0) // h_stride + 1
        w_grids = max(w_img - w_crop + w_stride - 1, 0) // w_stride + 1
        rec_frames = frames.new_zeros((batch_size, n_frames, c_img, h_img, w_img))
        count_mat = frames.new_zeros((batch_size, n_frames, 1, h_img, w_img))
        total_bits, pixel_counter = 0., 0
        for h_idx in range(h_grids):
            for w_idx in range(w_grids):
                y1 = h_idx * h_stride
                x1 = w_idx * w_stride
                y2 = min(y1 + h_crop, h_img)
                x2 = min(x1 + w_crop, w_img)
                y1 = max(y2 - h_crop, 0)
                x1 = max(x2 - w_crop, 0)
                crop_frames = frames[:, :, :, y1:y2, x1:x2]
                crop_rec_frames_list, main_bpp, hyper_bpp = self._compression(crop_frames, **kwargs)
                crop_rec_frames = crop_rec_frames_list[-1]
                rec_frames += F.pad(crop_rec_frames,
                                    (int(x1), int(rec_frames.shape[4] - x2), int(y1),
                                     int(rec_frames.shape[3] - y2)))
                bpp = main_bpp + hyper_bpp
                pixels = n_frames * (y2 - y1) * (x2 - x1)
                total_bits += float(bpp) * pixels
                pixel_counter += pixels
                count_mat[:, :, :, y1:y2, x1:x2] += 1
        assert (count_mat == 0).sum() == 0
        if torch.onnx.is_in_onnx_export():
            # cast count_mat to constant while exporting to ONNX
            count_mat = torch.from_numpy(
                count_mat.cpu().detach().numpy()).to(device=frames.device)
        rec_frames = rec_frames / count_mat
        return rec_frames, total_bits / pixel_counter

    def whole_inference(self, frames, **kwargs):
        """Inference with full image."""
        batch_size, n_frames, c_img, h_img, w_img = frames.size()
        pad_h, pad_w, target_h, target_w = 0, 0, h_img, w_img
        if h_img % 64 != 0:
            target_h = (h_img // 64 + 1) * 64
            pad_h = (target_h - h_img) // 2
        if w_img % 64 != 0:
            target_w = (w_img // 64 + 1) * 64
            pad_w = (target_w - w_img) // 2
        _frames = torch.nn.functional.pad(frames, [pad_w, target_w - pad_w - w_img, pad_h, target_h - pad_h - h_img])
        rec_frames_list, main_bpp, hyper_bpp = self._compression(_frames, **kwargs)
        rec_frames = rec_frames_list[-1]
        rec_frames = rec_frames[..., pad_h:pad_h + h_img, pad_w:pad_w + w_img]
        bpp = main_bpp + hyper_bpp
        ori_num_pixels = batch_size * n_frames * h_img * w_img
        tar_num_pixels = batch_size * n_frames * target_h * target_w
        bpp = bpp * tar_num_pixels / ori_num_pixels
        return rec_frames, bpp

    def inference(self, frames, **kwargs):
        """Inference with slide/whole style.

        Args:
            frames (Tensor): The input frames of shape (N, F, C, H, W).

        Returns:
            Tensor: The output segmentation map.
        """

        assert self.test_cfg.mode in ['slide', 'whole']
        if self.test_cfg.mode == 'slide':
            rec_frames, bpp = self.slide_inference(frames, **kwargs)
        else:
            rec_frames, bpp = self.whole_inference(frames, **kwargs)
        return rec_frames, bpp

    def forward_test(self, frames, return_image, **kwargs):
        """Simple test with single image."""
        self.eval()
        rec_frames, bpp = self.inference(frames, **kwargs)
        rec_loss = self.rec_loss_fn(rec_frames, frames)
        psnr = self._calculate_psnr(rec_frames, frames)
        results = dict(
            bpp=float(bpp),
            rec_loss=float(rec_loss),
            psnr=float(psnr)
        )
        if return_image:
            results['ori_frames'] = frames.detach().clone()
            results['rec_frames'] = rec_frames.detach().clone()
        del frames, rec_frames, bpp, rec_loss, psnr
        self.train()
        return results

    def show_result(self, data, result, show=False, out_file=None):
        if show:
            raise NotImplementedError

        if out_file is None:
            pass
        else:
            frames = [tensor_to_image(img) for img in result['ori_frames'].transpose(0, 1)]
            rec_frames = [tensor_to_image(img) for img in result['rec_frames'].transpose(0, 1)]

            suffix = out_file.split('.')[-1]
            assert suffix == 'png'
            cfg = {
                '-vcodec': 'libx264',  # use the h.264 codec
                '-crf': '0',  # set the constant rate factor to 0, which is lossless
            }
            mp4_dir = out_file.replace(os.path.basename(out_file), 'mp4')
            if not os.path.exists(mp4_dir):
                os.makedirs(mp4_dir)
            mp4_file = os.path.join(mp4_dir, os.path.basename(out_file)[:-4])
            ori_writer = FFmpegWriter(f'{mp4_file}.gt.mp4', outputdict=cfg)
            rec_writer = FFmpegWriter(f'{mp4_file}.mp4', outputdict=cfg)
            for i in range(len(frames)):
                ori_img, rec_img = frames[i][:, :, ::-1], rec_frames[i][:, :, ::-1]
                cv2.imwrite(f'{out_file[:-4]}_{i}.gt.png', ori_img)
                cv2.imwrite(f'{out_file[:-4]}_{i}.png', rec_img)
                ori_writer.writeFrame(ori_img)
                rec_writer.writeFrame(rec_img)
            ori_writer.close()
            rec_writer.close()

    # @auto_fp16(apply_to=('frames',))
    def forward(self, frames, return_loss=True, return_image=False, **kwargs):
        return super().forward(frames, return_loss, return_image, **kwargs)

    def init_weights(self, pretrained):
        if pretrained is not None:
            state_dict = torch.load(pretrained, map_location='cpu')
            remove_keys = [k for k in state_dict.keys() if k.startswith('intra_compressor')]
            for k in remove_keys:
                del state_dict[k]
            self.load_state_dict(state_dict)
        else:
            super().init_weights(pretrained)
