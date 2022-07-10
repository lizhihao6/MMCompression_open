VVC = '/home/VVCSoftware_VTM-VTM-14.0/bin/EncoderAppStatic'
VVC_CFG = '/home/VVCSoftware_VTM-VTM-14.0/cfg/encoder_intra_vtm.cfg'
QP = 48

model = dict(
    type='MPEGCompressor',
    mpeg=VVC,
    mpeg_cfg=VVC_CFG,
    depth=8,
    mode=420,
    qp=QP,
    test_cfg=dict(mode='whole')
)
