import os.path as osp

from .builder import DATASETS
from .nic import NICDataset


@DATASETS.register_module()
class NICRAWDataset(NICDataset):
    def __init__(self, **kwargs):
        meta_dir = kwargs.pop('meta_dir')
        super(NICRAWDataset, self).__init__(**kwargs)

        if self.data_root is not None and not (meta_dir is None or osp.isabs(meta_dir)):
            meta_dir = osp.join(self.data_root, meta_dir)
        # load annotations
        meta_suffix = '.json'
        for i in self.img_infos:
            meta_name = osp.basename(i['filename']).replace(self.img_suffix, meta_suffix)
            i['meta_path'] = osp.join(meta_dir, meta_name)

    def pre_pipeline(self, results):
        """Prepare results dict for pipeline."""
        results = super().pre_pipeline(results)
        results[''] = []
        results['img_prefix'] = self.img_dir
        return results
