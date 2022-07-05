_base_ = [
    '../_base_/models/nic.py', '../_base_/datasets/nic_dataset.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_100k.py'
]

LAMBDA_RD = 64.
model = dict(pretrained=f'.pretrain/rgb/mse{int(LAMBDA_RD)}00.pkl',
             train_cfg=dict(lambda_rd=LAMBDA_RD))
