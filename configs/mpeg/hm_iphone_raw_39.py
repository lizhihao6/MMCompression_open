_base_ = ['../_base_/datasets/iphone_raw.py']

HM = '/home/HM-HM-16.24/bin/TAppEncoderStatic'
HM_CFG = '/home/HM-HM-16.24/cfg/encoder_intra_main_rext.cfg'
QP = 39

model = dict(
    type='MPEGCompressor',
    mpeg=HM,
    mpeg_cfg=HM_CFG,
    depth=8,
    mode=444,
    test_cfg=dict(qp=QP),
)
