import argparse
import os.path
from collections import OrderedDict
from pathlib import Path

import torch
from tqdm import tqdm

parser = argparse.ArgumentParser(description='Convert old version nic state dicts to mmcompression.')
parser.add_argument('-p', '--paths', type=str, nargs='+', help='xxx.pkl xxx.pkl')
parser.add_argument('-d', '--dir', type=str, help='convert a dir')
parser.add_argument('-o', '--out_dir', type=str, help='saved dir', required=True)
args = parser.parse_args()

if __name__ == '__main__':
    paths = []
    if args.paths is not None:
        paths += [p for p in args.paths]
    if args.dir is not None:
        paths += [str(s) for s in Path(args.dir).glob('*.tar')]

    for p in tqdm(paths):
        old_weights = torch.load(p, map_location='cpu')['state_dict']
        new_weights = OrderedDict()
        old_keywords = list(old_weights.keys())

        for k in old_keywords:
            if k.startswith('g_a'):
                new_weights['main_encoder.' + k] = old_weights.pop(k)
            elif k.startswith('h_a'):
                new_weights['hyper_encoder.' + k] = old_weights.pop(k)
            elif k.startswith('g_s'):
                new_weights['main_decoder.' + k] = old_weights.pop(k)
                # v = old_weights.pop(k)
                # new_weights['main_decoder.' + k] = v
                # if 'g_s7' not in k:
                #     new_weights['residual_decoder.' + k] = v
            elif k.startswith('h_s'):
                new_weights['hyper_decoder.' + k] = old_weights.pop(k)
            elif k.startswith('entropy_parameters') or k.startswith('context_prediction'):
                new_weights['context_model.' + k] = old_weights.pop(k)
            elif '_matrix' in k:
                new_weights['entropy_model.' + 'matrix_{}'.format(k[-1])] = old_weights.pop(k)
            elif '_bias' in k:
                new_weights['entropy_model.' + 'bias_{}'.format(k[-1])] = old_weights.pop(k)
            elif '_factor' in k:
                new_weights['entropy_model.' + 'factor_{}'.format(k[-1])] = old_weights.pop(k)

        if not os.path.exists(args.out_dir):
            os.makedirs(args.out_dir)
        save_path = os.path.join(args.out_dir, os.path.basename(p))
        assert save_path != p
        torch.save(new_weights, save_path)
