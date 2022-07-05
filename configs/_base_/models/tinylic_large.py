# model settings
IN_CHANNELS = 3
STEM_CHANNELS = 192
MAIN_CHANNELS = 320
HYPER_CHANNELS = 192
LAMBDA_RD = 0.013 * 255 ** 2

model = dict(
    type='NICCompressor',
    pretrained='.pretrain/tinylic/quality_7.pth.tar',
    main_encoder=dict(type="TinyLICEnc",
                      input_channels=IN_CHANNELS,
                      stem_channels=STEM_CHANNELS,
                      main_channels=MAIN_CHANNELS),
    main_decoder=dict(type="TinyLICDec",
                      output_channels=IN_CHANNELS,
                      root_channels=STEM_CHANNELS,
                      main_channels=MAIN_CHANNELS),
    hyper_encoder=dict(type="TinyLICEnc_hyper",
                       main_channels=MAIN_CHANNELS,
                       hyper_channels=HYPER_CHANNELS),
    hyper_decoder=dict(type="TinyLICDec_hyper",
                       main_channels=MAIN_CHANNELS,
                       hyper_channels=HYPER_CHANNELS),
    entropy_model=dict(type="Factorized_Entropy",
                       hyper_channels=HYPER_CHANNELS,
                       filters=(3, 3, 3, 3)),
    context_model=dict(type="MCM",
                       main_channels=MAIN_CHANNELS),
    quant=dict(type="UniverseQuant"),
    rec_loss=dict(type="MSELoss"),
    # model training and testing settings
    train_cfg=dict(lambda_rd=LAMBDA_RD, lambda_bpp_scale=1),
    test_cfg=dict(mode='whole'))
