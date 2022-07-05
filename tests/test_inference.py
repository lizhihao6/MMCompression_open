import os.path as osp

import mmcv

from mmcomp.apis import inference_compressor, init_compressor


def test_test_time_inference_on_cpu():
    config_file = 'configs/nic/nic_256x256_12m_iphonescapes.py'
    config = mmcv.Config.fromfile(config_file)

    # Remove pretrain model download for testing
    config.model.pretrained = None

    # Enable test time augmentation
    # config.data.test.pipeline[1].flip = True

    checkpoint_file = None
    model = init_compressor(config, checkpoint_file, device='cpu')

    img = mmcv.imread(
        osp.join(osp.dirname(__file__), 'data/color.jpg'), 'color')
    img = mmcv.imresize(img, (512, 512))
    result = inference_compressor(model, img)
    assert result[0].shape == (1, 3, 512, 512)
