_base_ = ['../_base_/datasets/flicker2w.py']

model = dict(
    type='PNGCompressor',
    depth=8,
    test_cfg=None,
)
