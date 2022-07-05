import json
import os
from pathlib import Path

from tqdm import tqdm

HEVC_TEST_PATH = '/shared/HEVC_test_sequences'

if __name__ == '__main__':
    dirs = ['ClassA', 'ClassB', 'ClassC', 'ClassD', 'CLassE']
    for d in tqdm(dirs):
        sequences_list = []
        out_dir = os.path.join(HEVC_TEST_PATH, 'png', d)
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        for yuv in [str(s) for s in Path(os.path.join(HEVC_TEST_PATH, d)).glob('*.yuv')]:
            size = os.path.basename(yuv).split('_')[1]
            framerate = os.path.basename(yuv).split('_')[-1][:2]
            out_paths = os.path.join(out_dir, f'{os.path.basename(yuv)}_%03d.png')
            command = f'ffmpeg -s {size} -pix_fmt yuv420p -i {yuv} -framerate {framerate} {out_paths} > /dev/null 2>&1'
            os.system(command)
            seq = sorted([os.path.basename(str(s)) for s in Path(out_dir).glob(f'{os.path.basename(yuv)}*.png')])
            sequences_list.append(seq)

        with open(os.path.join(out_dir, 'nvc_test.json'), 'w+') as f:
            json.dump(sequences_list, f, indent=4)
