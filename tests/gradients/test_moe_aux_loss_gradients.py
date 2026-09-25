import torch

from initium.MoE.moe_layer import SwitchMoE


def test_aux_loss_alone_trains_router():
    layer = SwitchMoE(4, num_experts=4, router_noise_eps=0)
    layer(torch.randn(8, 4)).detach()
    layer.pop_aux_loss().backward()
    grad = layer.router.weight.grad
    assert grad is not None
    assert torch.isfinite(grad).all()
    assert grad.abs().sum() > 0
