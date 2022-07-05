from .builder import DATASETS
from .custom import CustomDataset


@DATASETS.register_module()
class NICDataset(CustomDataset):
    def __init__(self, **kwargs):
        super(NICDataset, self).__init__(**kwargs)
