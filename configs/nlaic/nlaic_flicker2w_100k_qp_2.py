_base_ = [
    '../_base_/models/nlaic.py', '../_base_/datasets/flicker2w.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_100k.py'
]
QP = 2
LAMBDA_RD = [2, 4, 8, 16, 32, 64, 128, 256][QP-1]
model = dict(pretrained=f'.pretrain/nlaic/mse{LAMBDA_RD}00.pkl',
             train_cfg=dict(lambda_rd=LAMBDA_RD))
