_base_ = ['../_base_/datasets/oneplus_raw.py']

FLIF = '/home/FLIF/src/flif'

model = dict(
    type='FLIFCompressor',
    flif=FLIF,
    depth=10,
    test_cfg=None,
)
