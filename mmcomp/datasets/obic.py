import os.path as osp

from .builder import DATASETS
from .custom import CustomDataset


@DATASETS.register_module()
class OBICDataset(CustomDataset):
    def __init__(self, **kwargs):
        self.ann_dir = kwargs.pop('ann_dir')
        self.obj_idx_list = kwargs.pop('obj_idx_list')
        super().__init__(**kwargs)
        if self.data_root is not None and not (self.ann_dir is None or osp.isabs(self.ann_dir)):
            self.ann_dir = osp.join(self.data_root, self.ann_dir)
        self.seg_map_suffix = '.png'

    def pre_pipeline(self, results):
        """Prepare results dict for pipeline."""
        results = super().pre_pipeline(results)
        results['seg_prefix'] = self.ann_dir
        results['obj_idx_list'] = self.obj_idx_list
        results['ann_info'] = dict(
            seg_map=results['img_info']['filename'].replace(self.img_suffix, self.seg_map_suffix))
        return results
