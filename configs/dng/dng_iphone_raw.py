_base_ = ['../_base_/datasets/iphone_raw.py']

model = dict(
    type='DNGCompressor',
    depth=12,
    test_cfg=None,
)
