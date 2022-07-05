_base_ = ['../_base_/datasets/oneplus_raw.py']

JPEGXL = '/home/libjxl/build/tools/cjxl'

model = dict(
    type='JPEGXLCompressor',
    jpeg_xl=JPEGXL,
    depth=10,
    test_cfg=None,
)
