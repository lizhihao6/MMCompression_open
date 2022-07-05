_base_ = ['../_base_/datasets/asi_raw.py']

JPEGXL = '/home/libjxl/build/tools/cjxl'

model = dict(
    type='JPEGXLCompressor',
    jpeg_xl=JPEGXL,
    depth=14,
    test_cfg=None,
)
