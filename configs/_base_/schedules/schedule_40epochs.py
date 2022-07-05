# optimizer
optimizer = dict(type='Adam', lr=1e-4)
optimizer_config = dict()
# learning policy
lr_config = dict(policy='step', step=[30, 50], by_epoch=True, gamma=0.1, min_lr=1e-5)
# runtime settings
runner = dict(type='EpochBasedRunner', max_epochs=40)
checkpoint_config = dict(by_epoch=True, interval=1)
evaluation = dict(interval=1)
