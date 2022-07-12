## Prepare datasets

It is recommended to symlink the dataset root to `$MMCOMPRESSION/data`.
If your folder structure is different, you may need to change the corresponding paths in config files.

```none
mmcompression
├── mmcomp
├── tools
├── configs
├── data
│   ├── flicker
│   │   ├── train
│   │   ├── test
│   ├── Kodak24
│   ├── IEEE1857_test
```

### flicker2w

The data could be found [here](https://github.com/liujiaheng/CompressionData).

### kodak24

Kodak24 could be downloaded from [here](http://www.cs.albany.edu/~xypan/research/snr/Kodak.html).
Noted that, the last image `kodim25.png` is dropped according to the common settings of this filed.

If you would like to download using script, please run following command to batch download the images.

```shell
python tools/convert_datasets/voc_aug.VOCaug --nproc 8
```

### ieee1857

Unfortunately, the IEEE1857 dataset is not available in the public repository.

Please refer to [concat dataset](https://github.com/open-mmlab/mmsegmentation/blob/master/docs/en/tutorials/customize_datasets.md#concatenate-dataset) for details about how to concatenate them and train them together.

