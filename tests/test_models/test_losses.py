import torch


def test_recover_loss():
    from mmcompression.models import build_loss

    # test L1 Loss
    loss_cfg = dict(type='L1Loss')
    loss_recover = build_loss(loss_cfg)
    y = gt = torch.randn([1, 3, 30, 30])
    loss = loss_recover(y, gt)
    assert torch.allclose(loss, torch.zeros_like(loss))

    # test MSE Loss
    loss_cfg = dict(type='MSELoss')
    loss_recover = build_loss(loss_cfg)
    y = gt = torch.randn([1, 3, 30, 30])
    loss = loss_recover(y, gt)
    assert torch.allclose(loss, torch.zeros_like(loss))

    # test MSSSIMLoss
    loss_cfg = dict(type='MSSSIMLOSS', channel=3)
    loss_recover = build_loss(loss_cfg)
    y = gt = torch.randn([1, 3, 180, 180])
    loss = loss_recover(y, gt)
    assert torch.allclose(loss, torch.zeros_like(loss))
