# optimizer
optimizer = dict(type='Adam', lr=1e-4)
optimizer_config = dict()
# learning policy
lr_config = dict(policy='step', step=[150, 250], by_epoch=True, gamma=0.1, min_lr=1e-5)
# runtime settings
runner = dict(type='EpochBasedRunner', max_epochs=200)
checkpoint_config = dict(by_epoch=True, interval=10)
evaluation = dict(interval=10)
