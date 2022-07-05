import argparse
import os.path
import pickle
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
        paths += [str(s).replace('p.pkl', '.pkl') for s in Path(args.dir).glob('*p.pkl')]

    for p in tqdm(paths):
        old_weights = torch.load(p, map_location='cpu')
        new_weights = OrderedDict()
        old_keywords = list(old_weights.keys())
        with open('new_state_dict_keys.pkl', 'rb+') as f:
            new_keywords = pickle.load(f)

        useless_keys = ['encoder.mask1', 'decoder.mask2', 'factorized_entropy_func.matrix',
                        'factorized_entropy_func.bias', 'factorized_entropy_func.factor']
        for k in old_keywords:
            if True in [k.startswith(u) for u in useless_keys]:
                del old_weights[k]

        old_groups = ['encoder', 'decoder', 'encoder', 'hyper_dec', 'p.context_p']
        new_groups = ['main_encoder', 'main_decoder', 'hyper_encoder', 'hyper_decoder', 'hyper_decoder.context_p']
        for o, n in zip(old_groups, new_groups):
            for k in [k for k in old_keywords if k.startswith(o) and k.replace(o, n) in new_keywords]:
                new_k = k.replace(o, n)
                new_weights[new_k] = old_weights.pop(k)

        old_keywords = ['._matrices.{}'.format(i) for i in range(4)] + ['._bias.{}'.format(i) for i in range(4)] + [
            '._factor.{}'.format(i) for i in range(3)]
        old_keywords = ['factorized_entropy_func' + k for k in old_keywords]
        new_keywords = ['.matrix_{}'.format(i) for i in range(4)] + ['.bias_{}'.format(i) for i in range(4)] + [
            '.factor_{}'.format(i) for i in range(3)]
        new_keywords = ['entropy_model' + k for k in new_keywords]
        for o, n in zip(old_keywords, new_keywords):
            new_weights[n] = old_weights.pop(o)
        assert not old_weights

        old_weights = torch.load(p.replace('.pkl', 'p.pkl'), map_location='cpu')
        old_keywords = list(old_weights.keys())
        for o in old_keywords:
            n = 'context_model.' + o
            new_weights[n] = old_weights.pop(o)
        assert not old_weights

        if not os.path.exists(args.out_dir):
            os.makedirs(args.out_dir)
        save_path = os.path.join(args.out_dir, os.path.basename(p))
        assert save_path != p
        torch.save(new_weights, save_path)
