_base_ = ['../_base_/datasets/iphone_raw.py']

JPEGXL = '/home/libjxl/build/tools/cjxl'

model = dict(
    type='JPEGXLCompressor',
    jpeg_xl=JPEGXL,
    depth=12,
    test_cfg=None,
)
