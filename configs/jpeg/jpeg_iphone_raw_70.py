_base_ = ['../_base_/datasets/iphone_raw.py']

QP = 70

model = dict(
    type='JPEGCompressor',
    test_cfg=dict(qp=QP),
)
