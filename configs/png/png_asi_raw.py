_base_ = ['../_base_/datasets/asi_raw.py']

model = dict(
    type='PNGCompressor',
    depth=14,
    test_cfg=None,
)
