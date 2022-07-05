_base_ = ['../_base_/datasets/iphone.py']

VVC = '/home/VVCSoftware_VTM-VTM-14.0/bin/EncoderAppStatic'
VVC_CFG = '/home/VVCSoftware_VTM-VTM-14.0/cfg/encoder_intra_vtm.cfg'
QP = 48

model = dict(
    type='MPEGCompressor',
    mpeg=VVC,
    mpeg_cfg=VVC_CFG,
    depth=8,
    mode=420,
    test_cfg=dict(qp=QP),
)
