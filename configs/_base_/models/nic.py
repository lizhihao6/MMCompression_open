# model settings
IN_CHANNELS = 3
STEM_CHANNELS = 96
MAIN_CHANNELS = 192
HYPER_CHANNELS = 128
LAMBDA_RD = 8.

model = dict(
    type='NICCompressor',
    pretrained='.pretrain/rgb/mse800.pkl',
    main_encoder=dict(type="NICEnc",
                      input_channels=IN_CHANNELS,
                      stem_channels=STEM_CHANNELS,
                      main_channels=MAIN_CHANNELS),
    main_decoder=dict(type="NICDec",
                      output_channels=IN_CHANNELS,
                      root_channels=STEM_CHANNELS,
                      main_channels=MAIN_CHANNELS),
    hyper_encoder=dict(type="NICEnc_hyper",
                       main_channels=MAIN_CHANNELS,
                       hyper_channels=HYPER_CHANNELS),
    hyper_decoder=dict(type="NICDec_hyper",
                       main_channels=MAIN_CHANNELS,
                       hyper_channels=HYPER_CHANNELS),
    entropy_model=dict(type="Factorized_Entropy",
                       hyper_channels=HYPER_CHANNELS),
    context_model=dict(type="Weighted_Gaussian",
                       main_channels=MAIN_CHANNELS),
    quant=dict(type="UniverseQuant"),
    rec_loss=dict(type="MSELoss"),
    # model training and testing settings
    train_cfg=dict(lambda_rd=LAMBDA_RD),
    test_cfg=dict(mode='slide', crop_size=(1024, 1024), stride=(1024, 1024)))
