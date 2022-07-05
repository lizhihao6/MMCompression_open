import argparse
import os
import tempfile
from pathlib import Path

import numpy as np

from utils import load_raw, load_metainfo, rearrange, inverse_rearrange, linearization, gamma, inv_gamma, rgb_to_yuv420, \
    yuv420_to_rgb, save_yuv420, save_yuv400, load_yuv420, load_yuv400


def parse_args():
    parser = argparse.ArgumentParser(
        description='hm compression')
    parser.add_argument('-d', '--dir', type=str, help='compression the whole dir, use with suffix')
    parser.add_argument('-i', '--imgs', type=str, nargs='+', help='compression images, use with base_dir')
    parser.add_argument('--base_dir', type=str, help='compression base dir, use with split_file and suffix')
    parser.add_argument('--split_file', type=str, help='split file, use with base_dir and suffix')
    parser.add_argument('--suffix', type=str, help='split file, use with base_dir and split_file or dir')
    parser.add_argument('--hm_path', default='/home/HM-HM-16.24/', type=str, help='the path of hm dir')
    parser.add_argument('-q', '--quality', type=int, nargs='+', required=True, help='the compression level')
    args = parser.parse_args()
    return args


def get_imgs_path(args):
    imgs = []
    if args.dir is not None:
        imgs += [str(s) for s in Path(args.dir).glob(f'*{args.suffix}')]
    if args.imgs is not None:
        for p in args.imgs:
            if not os.path.abspath(p) and hasattr(args, 'base_dir'):
                p = os.path.join(args.base_dir, p)
            imgs.append(p)
    if args.split_file is not None:
        with open(args.split_file, 'r') as f:
            imgs += [os.path.join(args.base_dir, f'{p.strip()}.{args.suffix}') for p in f.readlines()]
    return imgs


def compression(args, path):
    # base tmp path
    tmp_path = os.path.join(tempfile.gettempdir(), next(tempfile._get_candidate_names()))

    # load raw image
    raw = load_raw(path)
    meta_path = path.replace('raw', 'metainfo')[:-4] + '.json'
    meta = load_metainfo(meta_path)

    # preprocess
    raw = rearrange(raw, meta['bayer_pattern'])
    raw = linearization(raw, meta['black_level'], meta['white_level'])
    raw = gamma(raw)
    rgb, g = raw[..., [0, 1, 3]], raw[..., 2]
    g -= rgb[..., 1]
    g_min = g.min()
    g -= g_min
    g = np.clip(g, 0, 1)

    # raw to yuv
    yuv_path = tmp_path + '.yuv'
    yuv420 = rgb_to_yuv420(rgb)
    yuv420 = tuple(np.round(i * meta['white_level']).astype(np.uint16) for i in yuv420)
    save_yuv420(yuv420, yuv_path)
    g_path = tmp_path + '.g.yuv'
    yuv400 = (np.round(g * meta['white_level']).astype(np.uint16), None, None)
    save_yuv400(yuv400, g_path)

    # use hm compression
    hm_bin = os.path.join(args.hm_path, 'bin', 'TAppEncoderStatic')
    hm_base_cfg = os.path.join(args.hm_path, 'cfg', 'encoder_intra_main_rext.cfg')
    bin_path = tmp_path + '.hm.bin'
    recon_path = tmp_path + '.hm.rec.yuv'
    h, w = yuv420[0].shape[:2]
    config = f'--InputBitDepth={len(bin(meta["white_level"])) - 2} ' \
             f'--InternalBitDepth={len(bin(meta["white_level"])) - 2} ' \
             f'--InputChromaFormat=420 ' \
             f'--FrameRate=50 ' \
             f'--FrameSkip=0 ' \
             f'--FramesToBeEncoded=1 ' \
             f'--SourceWidth={w} ' \
             f'--SourceHeight={h} ' \
             f'--Level=2.1 ' \
             f'--QP={args.quality}'
    command = f'{hm_bin} -c {hm_base_cfg} -i {yuv_path} -b {bin_path} -o {recon_path} {config} > /dev/null 2>&1'
    os.system(command)
    g_bin_path = tmp_path + '.g.bin'
    g_recon_path = tmp_path + '.g.hm.rec.yuv'
    config = config.replace('420', '400')
    command = f'{hm_bin} -c {hm_base_cfg}  -i {g_path} -b {g_bin_path} -o {g_recon_path} {config} > /dev/null 2>&1'
    os.system(command)

    # recover
    yuv420 = load_yuv420(h, w, 16, recon_path)
    yuv400 = load_yuv400(h, w, 16, g_recon_path)
    yuv420 = tuple(i.astype(np.float32) / meta['white_level'] for i in yuv420)
    g = yuv400[0].astype(np.float32) / meta['white_level']
    rgb = yuv420_to_rgb(yuv420)
    g += g_min
    g += rgb[..., 1]
    g = np.clip(g, 0, 1)
    r, gb, gr, b = rgb[..., 0][..., None], rgb[..., 1][..., None], g[..., None], rgb[..., 2][..., None]
    raw = inv_gamma(np.concatenate([r, gb, gr, b], axis=-1))
    raw = inverse_rearrange(raw)

    # eval
    ori_raw = load_raw(path)
    ori_raw = linearization(ori_raw, meta['black_level'], meta['white_level'])
    psnr = 10 * np.log10(1. / np.power(ori_raw - raw, 2).mean())
    bpp = (float(os.path.getsize(bin_path)) + float(os.path.getsize(g_bin_path))) * 8 / (raw.shape[0] * raw.shape[1])

    # clean up
    os.system(f'rm {tmp_path}*')
    return psnr, bpp


if __name__ == '__main__':
    args = parse_args()
    files = get_imgs_path(args)
    quality = args.quality
    for q in quality:
        args.quality = q
        psnr, bpp = compression(args, files[0])
        print(f'{psnr}, {bpp}')
