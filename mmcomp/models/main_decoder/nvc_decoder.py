import torch
import torch.nn as nn
from mmcv.ops import DeformConv2d

from ..builder import MAINDECODER
from ..utils import ResBlock, Non_local_Block, Prior_STLSTM


class AttentionBlock(nn.Module):
    def __init__(self, channels, main_blocks, mask_blocks, non_local):
        super().__init__()
        self.main_branch = nn.Sequential()
        self.mask_branch = nn.Sequential()
        for i in range(main_blocks):
            self.main_branch.add_module(f'res_{i}', ResBlock(channels, channels, 3, 1, 1))

        if non_local:
            self.mask_branch.add_module('non_local', Non_local_Block(channels, channels // 2))
        for i in range(mask_blocks):
            self.mask_branch.add_module(f'res_{i}', ResBlock(channels, channels, 3, 1, 1))
        self.mask_branch.add_module('conv', nn.Conv2d(channels, channels, 1, 1, 0))

    def forward(self, x):
        return x + self.main_branch(x) * torch.sigmoid(self.mask_branch(x))


class TinyBackbone(nn.Module):
    def __init__(self, input_channels, output_channels):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(input_channels, 64, 5, 2, 2),
            nn.ReLU(),
            AttentionBlock(64, 1, 1, False),
            nn.Conv2d(64, 96, 5, 2, 2),
            AttentionBlock(96, 2, 1, False),
            nn.Conv2d(96, 128, 5, 2, 2),
            AttentionBlock(128, 2, 1, False),
            nn.Conv2d(128, output_channels, 5, 2, 2),
            AttentionBlock(output_channels, 1, 1, True)
        )

    def forward(self, x):
        return self.backbone(x)


class TinyNeck(nn.Module):
    def __init__(self, input_channels, output_channels):
        super().__init__()
        self.neck = nn.Sequential(
            AttentionBlock(input_channels, 1, 1, True),
            nn.Conv2d(input_channels, 128 * 4, 5, 1, 2, padding_mode='reflect'),
            nn.PixelShuffle(2),
            # nn.ConvTranspose2d(input_channels, 128, 5, 2, 2, 1),
            AttentionBlock(128, 2, 1, False),
            nn.Conv2d(128, 96 * 4, 5, 1, 2, padding_mode='reflect'),
            nn.PixelShuffle(2),
            # nn.ConvTranspose2d(128, 96, 5, 2, 2, 1),
            AttentionBlock(96, 2, 1, False),
            nn.Conv2d(96, 64 * 4, 5, 1, 2, padding_mode='reflect'),
            nn.PixelShuffle(2),
            # nn.ConvTranspose2d(96, 64, 5, 2, 2, 1),
            AttentionBlock(64, 1, 1, False),
            nn.Conv2d(64, 64 * 4, 5, 1, 2, padding_mode='reflect'),
            nn.PixelShuffle(2),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, output_channels, 3, 1, 1)
            # nn.ConvTranspose2d(64, output_channels, 5, 2, 2, 1)
        )

    def forward(self, x):
        return self.neck(x)


@MAINDECODER.register_module()
class NVCDec(nn.Module):
    def __init__(self, output_channels=3, main_channels=192):
        super().__init__()
        self.num_output = 4
        # extract features
        self.st_lstm = Prior_STLSTM(main_channels, main_channels)
        self.ref_features_extractor = TinyBackbone(output_channels, main_channels)

        # for predict flow
        self.flow = TinyNeck(main_channels, 2)

        # for align features
        self.dcn = DeformConv2d(main_channels, main_channels, kernel_size=(3, 3), padding=1, deform_groups=4)
        self.conv_offset = nn.Conv2d(main_channels, 4 * 2 * 3 * 3, kernel_size=(3, 3), padding=1, bias=True)
        self.texture_enhance = TinyNeck(main_channels, 3)

        # for fusion mask
        self.mask = TinyNeck(main_channels, 1)

        # for recover residual
        self.residual_enhance = TinyNeck(main_channels, 3)

    def forward(self, cur_features, pre_img, lstm_memory):
        h, c, M = lstm_memory
        st_fused_info, h, c, M = self.st_lstm(cur_features, h, c, M)  # st_fused_info [B, main_channels, h_f, w_f]
        ref_features = self.ref_features_extractor(pre_img)

        # use flow to predict image
        flow = self.flow(st_fused_info)
        pred_frame1 = self.backward_warp(pre_img.clone().detach(), flow)

        # use dcn aligned features to predict image
        offset = self.conv_offset(st_fused_info)
        aligned_ref_features = self.dcn(ref_features, offset)
        pred_frame2 = self.texture_enhance(aligned_ref_features)

        # use frame1 and frame2 to fusion image
        mask = torch.sigmoid(self.mask(st_fused_info))
        pred_frame3 = pred_frame1.detach() * mask + pred_frame2.detach() * (1 - mask)

        # use estimated residual to predict image
        pred_frame4 = pred_frame3.detach() + self.residual_enhance(st_fused_info)

        return (pred_frame1, pred_frame2, pred_frame3, pred_frame4), (h, c, M)

    @staticmethod
    def backward_warp(x, flow):
        b, c, h, w = x.shape
        # mesh grid
        xx = torch.arange(0, w).view(1, -1).repeat(h, 1)
        yy = torch.arange(0, h).view(-1, 1).repeat(1, w)
        xx = xx.view(1, 1, h, w).repeat(b, 1, 1, 1)
        yy = yy.view(1, 1, h, w).repeat(b, 1, 1, 1)
        grid = torch.cat((xx, yy), 1).float().to(x.device)
        vgrid = grid + flow

        # scale grid to [-1,1]
        vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :].clone() / max(w - 1, 1) - 1.0
        vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :].clone() / max(h - 1, 1) - 1.0
        vgrid = vgrid.permute(0, 2, 3, 1)
        output = nn.functional.grid_sample(x, vgrid, align_corners=True).clone()
        mask = torch.ones_like(x)
        mask = nn.functional.grid_sample(mask, vgrid, align_corners=True).clone()
        mask[mask < 0.9999] = 0
        mask[mask > 0] = 1
        return output * mask
