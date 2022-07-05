_base_ = ['../_base_/datasets/huawei_raw.py']

model = dict(
    type='PNGCompressor',
    depth=12,
    test_cfg=None,
)
