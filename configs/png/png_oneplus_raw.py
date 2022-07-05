_base_ = ['../_base_/datasets/oneplus_raw.py']

model = dict(
    type='PNGCompressor',
    depth=10,
    test_cfg=None,
)
