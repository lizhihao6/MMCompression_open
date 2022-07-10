# model settings
IN_CHANNELS = 3
STEM_CHANNELS = 128
MAIN_CHANNELS = 192
HYPER_CHANNELS = 128
QP = 1
LAMBDA_RD = [0.0018, 0.0035, 0.0067, 0.013, 0.025, 0.0483, 0.0932, 0.18][QP - 1] * 255 ** 2
LAMBDA_BPP = 1

model = dict(
    type='NICCompressor',
    pretrained='.pretrain/tinylic/quality_3.pth.tar',
    vae=dict(
        type='SwinVAE',
        in_channels=IN_CHANNELS,
        stem_channels=STEM_CHANNELS,
        main_channels=MAIN_CHANNELS,
        hyper_channels=HYPER_CHANNELS,
    ),
    entropy_model=dict(type="FactorizedEntropy",
                       hyper_channels=HYPER_CHANNELS,
                       filters=(3, 3, 3, 3)),
    context_model=dict(type="MCM",
                       main_channels=MAIN_CHANNELS),
    quant=dict(type="UniverseQuant"),
    rec_loss=dict(type="MSELoss"),
    # model training and testing settings
    train_cfg=dict(lambda_rd=LAMBDA_RD, lambda_bpp=LAMBDA_BPP),
    test_cfg=dict(mode='whole', mod_size=(256, 256)))
