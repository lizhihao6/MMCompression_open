_base_ = [
    '../_base_/models/.obic.py', '../_base_/datasets/iphone_obic.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_100k.py'
]

FACTOR_OBJ = 1.
FACTOR_BKG = 0.2
model = dict(pretrained='.pretrain/rgb/mse800.pkl',
             test_cfg=dict(factor_obj=FACTOR_OBJ, factor_bkg=FACTOR_BKG))
