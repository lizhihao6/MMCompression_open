# model settings
_base_ = ['nic.py']
STEM_CHANNELS = 128
MAIN_CHANNELS = 256
HYPER_CHANNELS = 192
LAMBDA_RD = 128.

model = dict(
    type='NICCompressor',
    pretrained='.pretrain/rgb/mse12800.pkl',
    main_encoder=dict(type="NICEnc",
                      stem_channels=STEM_CHANNELS,
                      main_channels=MAIN_CHANNELS),
    main_decoder=dict(type="NICDec",
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
    # model training and testing settings
    train_cfg=dict(lambda_rd=LAMBDA_RD))
