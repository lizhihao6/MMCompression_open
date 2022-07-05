_base_ = ['../_base_/datasets/huawei_raw.py']

FLIF = '/home/FLIF/src/flif'

model = dict(
    type='FLIFCompressor',
    flif=FLIF,
    depth=12,
    test_cfg=None,
)
