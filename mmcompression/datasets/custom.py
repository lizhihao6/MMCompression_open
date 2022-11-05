# Copyright (c) NJU Vision Lab. All rights reserved.
import os.path as osp

import mmcv
from mmcv.utils import print_log
from torch.utils.data import Dataset

from mmcompression.utils import get_root_logger
from .builder import DATASETS
from .pipelines import Compose


@DATASETS.register_module()
class CustomDataset(Dataset):
    """Custom dataset for compression. An example of file structure
    is as followed.

    .. code-block:: none

        ├── data
        │   ├── my_dataset
        │   │   ├── img_dir
        │   │   │   ├── train
        │   │   │   │   ├── xxx{img_suffix}
        │   │   │   │   ├── yyy{img_suffix}
        │   │   │   │   ├── zzz{img_suffix}
        │   │   │   ├── val


    Args:
        pipeline (list[dict]): Processing pipeline
        img_dir (str): Path to image directory
        img_suffix (str): Suffix of images. Default: '.jpg'
        split (str, optional): Split txt file. If split is specified, only
            file with suffix in the splits will be loaded. Otherwise, all
            images in img_dir/ann_dir will be loaded. Default: None
        data_root (str, optional): Data root for img_dir/ann_dir. Default:
            None.
    """

    def __init__(self,
                 pipeline,
                 img_dir,
                 img_suffix='.jpg',
                 split=None,
                 data_root=None):
        self.pipeline = Compose(pipeline)
        self.img_dir = img_dir
        self.img_suffix = img_suffix
        self.split = split
        self.data_root = data_root

        # join paths if data_root is specified
        if self.data_root is not None:
            if not osp.isabs(self.img_dir):
                self.img_dir = osp.join(self.data_root, self.img_dir)
            if not (self.split is None or osp.isabs(self.split)):
                self.split = osp.join(self.data_root, self.split)

        # load annotations
        self.img_infos = self.load_annotations(self.img_dir, self.img_suffix, self.split)

    def __len__(self):
        """Total number of samples of data."""
        return len(self.img_infos)

    @staticmethod
    def load_annotations(img_dir, img_suffix, split):
        """Load annotation from directory.

        Args:
            img_dir (str): Path to image directory
            img_suffix (str): Suffix of images.
            split (str|None): Split txt file. If split is specified, only file
                with suffix in the splits will be loaded. Otherwise, all images
                in img_dir/ann_dir will be loaded. Default: None

        Returns:
            list[dict]: All image info of dataset.
        """

        img_infos = []
        if split is not None:
            with open(split, 'r', encoding='utf-8') as f:
                for line in f:
                    img_name = line.strip()
                    img_info = dict(filename=img_name + img_suffix)
                    img_infos.append(img_info)
        else:
            for img in mmcv.scandir(img_dir, img_suffix, recursive=True):
                img_info = dict(filename=img)
                img_infos.append(img_info)

        print_log(f'Loaded {len(img_infos)} images', logger=get_root_logger())
        return img_infos

    def pre_pipeline(self, results):
        """Prepare results dict for pipeline."""
        results['seg_fields'] = []
        results['img_prefix'] = self.img_dir
        return results

    def __getitem__(self, idx):
        """Get training/test data after pipeline.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Training/test data (with annotation if `test_mode` is set
                False).
        """
        img_info = self.img_infos[idx]
        results = dict(img_info=img_info)
        results = self.pre_pipeline(results)
        return self.pipeline(results)

    def format_results(self, results, **kwargs):
        """Place holder to format result to dataset specific output."""
        return self.evaluate(results, **kwargs)

    @staticmethod
    def evaluate(results,
                 logger=None,
                 **kwargs):
        """Evaluate the dataset.

        Args:
            results (list): Testing results of the dataset.
            logger (logging.Logger | None | str): Logger used for printing
                related information during evaluation. Default: None.
            kwargs (dict): Other arguments for evaluation.

        Returns:
            dict[str, float]: Default metrics.
        """
        keys = [k for k in results[0].keys() if isinstance(results[0][k], float)]
        values = [0. for _ in keys]
        for r in results:
            for k in keys:
                values[keys.index(k)] += r[k] / len(results)

        eval_results = {}
        log_str = ''
        for k, v in zip(keys, values):
            eval_results[k] = v
            log_str += f'{k}: {v}, '
        print_log(log_str[:-2], logger=logger if logger is not None else get_root_logger())
        return eval_results
