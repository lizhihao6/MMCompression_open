_base_ = ['../_base_/datasets/asi_raw.py']

FLIF = '/home/FLIF/src/flif'

model = dict(
    type='FLIFCompressor',
    flif=FLIF,
    depth=14,
    test_cfg=None,
)
