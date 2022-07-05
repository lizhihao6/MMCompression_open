# dataset settings
dataset_type = 'NICDataset'
img_norm_cfg = dict(mean=[0, 0, 0], std=[255, 255, 255], to_rgb=True)
crop_size = (256, 256)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='RandomCrop', crop_size=crop_size),
    dict(type='RandomFlip', prob=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='Collect', keys=['img']),
]

data = dict(
    samples_per_gpu=16,
    workers_per_gpu=16,
    train=dict(
        type=dataset_type,
        data_root='/shared/',
        img_dir='flicker/train',
        img_suffix='.jpg',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        data_root='/shared/',
        img_dir='IEEE1857_test',
        img_suffix='.png',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        data_root='/shared/',
        img_dir='IEEE1857_test',
        img_suffix='.png',
        pipeline=test_pipeline))
