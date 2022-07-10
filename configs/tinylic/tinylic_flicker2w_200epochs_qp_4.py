_base_ = [
    '../_base_/models/tinylic.py', '../_base_/datasets/flicker2w.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_200epochs.py'
]

QP = 4
LAMBDA_RD = [0.0018, 0.0035, 0.0067, 0.013, 0.025, 0.0483, 0.0932, 0.18][QP - 1] * 255 ** 2
model = dict(train_cfg=dict(lambda_rd=LAMBDA_RD))
