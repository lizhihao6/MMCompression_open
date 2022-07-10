# model settings
IN_CHANNELS = 3
STEM_CHANNELS = 96
MAIN_CHANNELS = 192
HYPER_CHANNELS = 128
QP = 1
LAMBDA_RD = [2, 4, 8, 16, 32, 64, 128, 256][QP - 1]
LAMBDA_BPP = 0.01

model = dict(
    type='NICCompressor',
    pretrained='.pretrain/rgb/mse800.pkl',
    vae=dict(
        type='NonLocalVAE',
        in_channels=IN_CHANNELS,
        stem_channels=STEM_CHANNELS,
        main_channels=MAIN_CHANNELS,
        hyper_channels=HYPER_CHANNELS,
    ),
    entropy_model=dict(type="FactorizedEntropy",
                       hyper_channels=HYPER_CHANNELS),
    context_model=dict(type="WeightedGaussian",
                       main_channels=MAIN_CHANNELS),
    quant=dict(type="UniverseQuant"),
    rec_loss=dict(type="MSELoss"),
    # model training and testing settings
    train_cfg=dict(lambda_rd=LAMBDA_RD, lambda_bpp=LAMBDA_BPP),
    test_cfg=dict(mode='whole', mod_size=(64, 64)))
