_base_ = [
    '../_base_/models/tinylic.py', '../_base_/datasets/flicker2w.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_400epochs.py'
]

LAMBDA_RD = 0.0067 * 255 ** 2
model = dict(train_cfg=dict(lambda_rd=LAMBDA_RD, lambda_bpp_scale=1))
