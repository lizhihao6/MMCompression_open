import os.path as osp

import mmcv

from mmcomp.apis import inference_compressor, init_compressor


def test_test_time_inference_on_cpu():
    config_file = 'configs/tinylic/tinylic_flicker2w_200epochs_qp_1.py'
    config = mmcv.Config.fromfile(config_file)

    # Remove pretrain model download for testing
    config.model.pretrained = None

    checkpoint_file = None
    model = init_compressor(config, checkpoint_file, device='cpu')

    img = mmcv.imread(
        osp.join(osp.dirname(__file__), 'data/color.jpg'), 'color')
    img = mmcv.imresize(img, (512, 512))
    result = inference_compressor(model, img, return_image=False)
    assert 'bpp' in result.keys()
    assert 'psnr' in result.keys()

    result = inference_compressor(model, img, return_image=True)
    ori_img = mmcv.imread(result['ori_img'], 'color')
    rec_img = mmcv.imread(result['rec_img'], 'color')
    assert ori_img.shape == (512, 512, 3)
    assert rec_img.shape == (512, 512, 3)
