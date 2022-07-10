import mmcv
import torch
from mmcv.parallel import collate, scatter
from mmcv.runner import load_checkpoint

from mmcomp.datasets.pipelines import Compose
from mmcomp.models import build_compressor


def init_compressor(config, checkpoint=None, device="cuda:0"):
    """Initialize a compressor from config file.

    Args:
        config (str or :obj:`mmcv.Config`): Config file path or the config
            object.
        checkpoint (str, optional): Checkpoint path. If left as None, the model
            will not load any weights.
        device (str, optional) CPU/CUDA device option. Default 'cuda:0'.
            Use 'cpu' for loading model on CPU.
    Returns:
        nn.Module: The constructed compressor.
    """
    if isinstance(config, str):
        config = mmcv.Config.fromfile(config)
    elif not isinstance(config, mmcv.Config):
        raise TypeError(
            f"config must be a filename or Config object,but got {type(config)}"
        )
    if checkpoint is not None:
        config.model.pretrained = None
    config.model.train_cfg = None
    model = build_compressor(config.model, test_cfg=config.get("test_cfg"))
    if checkpoint is not None:
        checkpoint = load_checkpoint(model, checkpoint, map_location="cpu")
    model.cfg = config  # save the config in the model for convenience
    model.to(device)
    model.eval()
    return model


class LoadImage:
    """A simple pipeline to load image."""

    def __call__(self, results):
        """Call function to load images into results.

        Args:
            results (dict): A result dict contains the file name
                of the image to be read.

        Returns:
            dict: ``results`` will be returned containing loaded image.
        """

        if isinstance(results["img"], str):
            results["filename"] = results["img"]
            results["ori_filename"] = results["img"]
        else:
            results["filename"] = None
            results["ori_filename"] = None
        img = mmcv.imread(results["img"])
        results["img"] = img
        results["img_shape"] = img.shape
        results["ori_shape"] = img.shape
        return results


def inference_compressor(model, img, return_image=False):
    """Inference image(s) with the compressor.

    Args:
        model (nn.Module): The loaded compressor.
        img (str/ndarray or list[str/ndarray]): Either image files or loaded
            images.
        return_image (bool): Whether return the image after inference.
    """
    cfg = model.cfg
    device = next(model.parameters()).device  # model device
    # build the data pipeline
    test_pipeline = [LoadImage()] + cfg.data.test.pipeline[1:]
    # test_pipeline = cfg.data.test.pipeline
    test_pipeline = Compose(test_pipeline)
    # prepare data
    data = dict(img=img)
    data = test_pipeline(data)
    data = collate([data], samples_per_gpu=1)
    if next(model.parameters()).is_cuda:
        # scatter to specified GPU
        data = scatter(data, [device])[0]
    else:
        data["img_metas"] = [i[0] for i in data["img_metas"].data]
    # forward the model
    with torch.no_grad():
        result = model(return_loss=False, return_image=return_image, **data)
    return result
