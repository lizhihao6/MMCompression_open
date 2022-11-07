# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import torch


def _demo_mm_inputs(input_shape=(1, 3, 256, 256)):
    """Create a superset of inputs needed to run test or train batches.
    Args:
        input_shape (tuple):
            input batch dimensions
    """
    (N, C, H, W) = input_shape

    rng = np.random.RandomState(0)

    imgs = rng.rand(*input_shape)

    img_metas = [{
        'img_shape': (H, W, C),
        'ori_shape': (H, W, C),
        'pad_shape': (H, W, C),
        'filename': '<demo>.png',
        'scale_factor': 1.0,
        'flip': False,
        'flip_direction': 'horizontal'
    } for _ in range(N)]

    mm_inputs = {
        'imgs': torch.FloatTensor(imgs),
        'img_metas': img_metas,
    }
    return mm_inputs


#
#
# @BACKBONES.register_module()
# class ExampleBackbone(nn.Module):
#
#     def __init__(self):
#         super(ExampleBackbone, self).__init__()
#         self.conv = nn.Conv2d(3, 3, 3)
#
#     def init_weights(self, pretrained=None):
#         pass
#
#     def forward(self, x):
#         return [self.conv(x)]
#
#
# @HEADS.register_module()
# class ExampleDecodeHead(BaseDecodeHead):
#
#     def __init__(self):
#         super(ExampleDecodeHead, self).__init__(3, 3, num_classes=19)
#
#     def forward(self, inputs):
#         return self.cls_seg(inputs[0])
#
#
# @HEADS.register_module()
# class ExampleCascadeDecodeHead(BaseCascadeDecodeHead):
#
#     def __init__(self):
#         super(ExampleCascadeDecodeHead, self).__init__(3, 3, num_classes=19)
#
#     def forward(self, inputs, prev_out):
#         return self.cls_seg(inputs[0])


def _compressor_forward_train_test(compressor):
    # batch_size=2 for BatchNorm
    mm_inputs = _demo_mm_inputs()

    imgs = mm_inputs.pop('imgs')
    img_metas = mm_inputs.pop('img_metas')

    # convert to cuda Tensor if applicable
    if torch.cuda.is_available():
        compressor = compressor.cuda()
        imgs = imgs.cuda()

    # Test forward train
    losses = compressor.forward(
        imgs, img_metas, return_loss=True)
    assert isinstance(losses, dict)

    # Test forward simple test without return image
    with torch.no_grad():
        compressor.eval()
        # pack into lists
        img_list = [img[None, :] for img in imgs]
        img_meta_list = [[img_meta] for img_meta in img_metas]
        compressor.forward(img_list, img_meta_list, return_loss=False, return_img=False)

    # Test forward simple test with return image
    with torch.no_grad():
        compressor.eval()
        # pack into lists
        img_list = [img[None, :] for img in imgs]
        img_meta_list = [[img_meta] for img_meta in img_metas]
        compressor.forward(img_list, img_meta_list, return_loss=False, return_img=True)
