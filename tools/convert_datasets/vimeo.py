import json
import os

from tqdm import tqdm

VIMEO_PATH = '/shared/vimeo_septuplet'

if __name__ == '__main__':
    sequences_list = []
    dirs = []
    with open(os.path.join(VIMEO_PATH, 'sep_trainlist.txt'), 'r') as f:
        dirs += [os.path.join('sequences', l.strip()) for l in f.readlines()]
    with open(os.path.join(VIMEO_PATH, 'sep_testlist.txt'), 'r') as f:
        dirs += [os.path.join('sequences', l.strip()) for l in f.readlines()]
    for d in tqdm(dirs):
        seq = []
        for i in range(1, 8):
            im = os.path.join(d, f'im{i}.png')
            assert os.path.exists(os.path.join(VIMEO_PATH, im))
            seq.append(im)
        sequences_list.append(seq)

    with open(os.path.join(VIMEO_PATH, 'nvc_train.json'), 'w+') as f:
        json.dump(sequences_list, f, indent=4)
