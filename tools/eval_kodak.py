import argparse
import os
from pathlib import Path
from pprint import pprint

import numpy as np
from imageio import imread
from tqdm import trange

VALID_PATH = "/shared/NIC_Dataset/test_crop/ClassD_Kodak/"
HM_ENC_PATH = "/workspace/HM/bin/TAppEncoderStatic"
HM_CFG_PATH = "/workspace/HM/cfg/encoder_intra_main_rext.cfg"
VVC_ENC_PATH = "/workspace/VVCSoftware_VTM-master/bin/EncoderAppStatic"
VVC_CFG_PATH = "/workspace/VVCSoftware_VTM-master/cfg/encoder_intra_vtm.cfg"


def preprocess():
    paths = [str(s) for s in Path(VALID_PATH).glob("*.png")]
    for _i in trange(len(paths)):
        p = paths[_i]
        yuv_path = p.replace(".png", ".yuv")
        os.system("ffmpeg -y -i {} -pix_fmt yuv444p {}".format(p, yuv_path))


def bpg_comp(src_yuv_path, qp, cf):
    bpg_path = "/workspace/.hm.bpg"
    png_path = "/workspace/.hm.png"
    H, W = imread(src_yuv_path.replace(".yuv", ".png")).shape[:2]
    os.system("bpgenc -o {} -q {} -f {} {}".format(bpg_path, qp, cf, src_yuv_path.replace(".yuv", ".png")))
    os.system("bpgdec -o {} {}".format(png_path, bpg_path))
    bpp = float(os.path.getsize(bpg_path)) * 8 / (H * W)
    mse = imread(src_yuv_path.replace(".yuv", ".png")).astype(np.float32) / 255. - imread(png_path).astype(
        np.float32) / 255.
    mse = mse ** 2
    os.remove(png_path)
    return mse.mean(), bpp


def hm_comp(src_yuv_path, qp, cf):
    bin_path = "/workspace/.hm.bin"
    yuv_path = "/workspace/.hm.yuv"
    png_path = "/workspace/.hm.png"
    H, W = imread(yuv_path.replace(".yuv", ".png")).shape[:2]
    os.system(
        "{} -i {} -fr 1 -f 1 -hgt {} -wdt {} -c {} --InputBitDepth=8 --InternalBitDepth=8 --OutputBitDepth=8 -q {} -b {} -o {} -cf {} > /dev/null".format(
            HM_ENC_PATH, src_yuv_path, H // 2, W // 2, HM_CFG_PATH, qp, bin_path, yuv_path, cf))
    bpp = float(os.path.getsize(bin_path)) * 8 / (H * W)
    os.system("ffmpeg -y -i {} -pix_fmt yuv444p {}".format(yuv_path, png_path))
    mse = imread(src_yuv_path.replace(".yuv", ".png")).astype(np.float32) / 255. - imread(png_path).astype(
        np.float32) / 255.
    mse = mse ** 2
    os.remove(bin_path)
    os.remove(yuv_path)
    os.remove(png_path)
    return mse.mean(), bpp


def vvc_comp(src_yuv_path, qp, cf):
    bin_path = "/workspace/.vvc.bin"
    yuv_path = "/workspace/.vvc.yuv"
    H, W = imread(yuv_path.replace(".yuv", ".png")).shape[:2]
    os.system(
        "{} -i {} -fr 1 -f 1 -hgt {} -wdt {} -c {} --InputBitDepth=8 --OutputBitDepth=8 -q {} -b {} -o {} -cf {} > /dev/null".format(
            VVC_ENC_PATH, src_yuv_path, H // 2, W // 2, VVC_CFG_PATH, qp, bin_path, yuv_path, cf))
    bpp = float(os.path.getsize(bin_path)) * 8 / (H * W)
    os.system("ffmpeg -y -i {} -pix_fmt yuv444p {}".format(yuv_path, png_path))
    mse = imread(src_yuv_path.replace(".yuv", ".png")).astype(np.float32) / 255. - imread(png_path).astype(
        np.float32) / 255.
    mse = mse ** 2
    os.remove(bin_path)
    os.remove(yuv_path)
    os.remove(png_path)
    return mse.mean(), bpp


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Eval Script")
    parser.add_argument("--method", required=True, choices={"bpg", "hm", "vvc"}, type=str, help="compression method")
    parser.add_argument("--cf", required=True, choices={420, 444}, type=int, help="compression method")
    opts = parser.parse_args()
    if not os.path.exists(os.path.join(VALID_PATH, "1.yuv")):
        preprocess()
    paths = [str(s)[:-4] for s in Path(VALID_PATH).glob("*.png")]

    # measure qp
    results = ["Qulity, BPP, PSNR\n"]
    for quality in trange(20, 1, -1):
        bpp_counter, psnr_counter = 0., 0.
        if opts.method == "bpg":
            comp_fn = bpg_comp
        elif opts.method == "hm":
            comp_fn = hm_comp
        elif opts.method == "vvc":
            comp_fn = vvc_comp
        else:
            print("compression method do not support.")
            exit(-1)
        for _i in trange(len(paths)):
            mse, bpp = comp_fn(paths[_i] + ".yuv", quality, opts.cf)
            bpp_counter += bpp
            psnr_counter += 10 * np.log10(1. / mse)
        log = "{}, {}, {}\n".format(quality, bpp_counter / len(paths), psnr_counter / len(paths))
        pprint(log)
        results.append(log)
    with open(os.path.join("output", "nic_fvae_{}.csv".format(opts.method)), "w") as f:
        f.writelines(results)
