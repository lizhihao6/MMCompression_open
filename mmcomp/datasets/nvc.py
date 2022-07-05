import json
import os.path

from mmcv.utils import print_log
from torch.utils.data import Dataset

from mmcomp.utils import get_root_logger
from .builder import DATASETS
from .pipelines import Compose


@DATASETS.register_module()
class NVCDataset(Dataset):
    """Video dataset for compression.

    Args:
        pipeline (list[dict]): Processing pipeline
        data_root (str): Data root for frames
        sequence (str): Sequence txt file
        n_frames (int): frames to load in once compression
        test_mode (bool): Place Holder. Default: False
    """

    def __init__(self,
                 pipeline,
                 data_root,
                 sequence,
                 n_frames: int,
                 test_mode=False):
        self.pipeline = Compose(pipeline)
        self.data_root = data_root
        if not os.path.isabs(sequence):
            sequence = os.path.join(data_root, sequence)
        with open(sequence, 'r') as f:
            self.sequences = json.load(f)
        self.n_frames = n_frames
        self.test_mode = test_mode

        # load annotations
        self.frames_infos = self.load_annotations(self.sequences, self.n_frames, self.test_mode)

    def __len__(self):
        """Total number of samples of data."""
        return len(self.frames_infos)

    def load_annotations(self, sequences, n_frames, test_mode=False):
        """Load annotation from directory.

        Args:
            sequences (list[list[]]): List of frame sequence
            n_frames (int): frames to load in once compression
            test_mode (bool): Place Holder. Default: False

        Returns:
            list[dict]: All frames info of dataset.
        """

        frames_infos = []
        for seq in sequences:
            if not test_mode:
                if len(seq) < n_frames:
                    continue
                else:
                    for i in range(len(seq) - n_frames + 1):
                        frames_info = dict(frames=tuple(seq[i:i + n_frames]), filename=seq[i])
                        frames_infos.append(frames_info)
            else:
                pointer = 0
                while pointer < len(seq):
                    end = min(pointer + n_frames, len(seq))
                    frames_info = dict(frames=tuple(seq[pointer:end]), filename=seq[pointer])
                    frames_infos.append(frames_info)
                    pointer = end
        print_log(f'Loaded {len(frames_infos)} frames', logger=get_root_logger())
        return frames_infos

    def pre_pipeline(self, results):
        """Place holder to prepare results dict for pipeline."""
        return results

    def __getitem__(self, idx):
        """Get training/test data after pipeline.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Training/test data (with annotation if `test_mode` is set
                False).
        """
        results = dict(frames_prefix=self.data_root,
                       frames_info=self.frames_infos[idx])
        results = self.pre_pipeline(results)
        return self.pipeline(results)

    def format_results(self, results, **kwargs):
        """Place holder to format result to dataset specific output."""
        self.evaluate(results)

    def evaluate(self,
                 results,
                 logger=None,
                 **kwargs):
        """Evaluate the dataset.

        Args:
            results (list): Testing results of the dataset.
            logger (logging.Logger | None | str): Logger used for printing
                related information during evaluation. Default: None.

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
            log_str += '{}: {}, '.format(k, v)
        print_log(log_str[:-2])
        return eval_results
