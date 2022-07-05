_base_ = [
    '../_base_/models/nvc.py', '../_base_/datasets/vimeo.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_100k.py'
]

# model settings
PRETRAIN = True
LAMBDA_PRETRAIN_RD = 32.
LAMBDA_RD = 16.

model = dict(
    pretrained=None,
    intra_cfg=dict(pretrained=f'.pretrain/rgb/mse{int(LAMBDA_RD * 4)}00.pkl'),
    inter_cfg=dict(pretrained=f'.pretrain/rgb/mse{int(LAMBDA_PRETRAIN_RD * 4)}00.pkl', ),
    residual_cfg=None,
    # model training and testing settings
    train_cfg=dict(pretrain=PRETRAIN,
                   lambda_pretrain_rd=LAMBDA_PRETRAIN_RD,
                   lambda_rd=LAMBDA_RD)
)

# datasets settings
BATCH_SIZE = 32
N_FRAMES = 2
data = dict(
    samples_per_gpu=BATCH_SIZE,
    workers_per_gpu=BATCH_SIZE,
    train=dict(n_frames=N_FRAMES),
    val=dict(n_frames=N_FRAMES),
    test=dict(n_frames=N_FRAMES)
)
