import pytest
import torch

from MoE.moe_layer import MultiSourceMoEBlock, SwitchMoE


def test_switch_moe_validation_routing_and_aux_gradient():
    with pytest.raises(ValueError, match="num_experts=3"):
        SwitchMoE(4, num_experts=3)
    moe = SwitchMoE(4, num_experts=4, router_noise_eps=0, load_balance_alpha=0.1)
    with torch.no_grad():
        moe.router.weight.zero_()
        moe.router.weight[2, 0] = 1
        moe.router.weight[0, 0] = -1
    x = torch.tensor([[1.0, 0, 0, 0], [-1.0, 0, 0, 0]])
    y = moe(x)
    assert y.shape == x.shape
    assert moe.last_routing().shape == (2, 1)
    expected = moe.router(x).argmax(dim=-1, keepdim=True)
    torch.testing.assert_close(moe.last_routing(), expected)
    aux = moe.pop_aux_loss()
    aux.backward()
    assert moe.router.weight.grad is not None
    assert torch.isfinite(moe.router.weight.grad).all()
    assert moe.pop_aux_loss().item() == 0


def test_capacity_drop_zeros_overflow_token_output():
    moe = SwitchMoE(2, num_experts=4, capacity_factor=0, router_noise_eps=0)
    with torch.no_grad():
        moe.router.weight.zero_()
        for expert in moe.experts:
            expert.w_in.weight.zero_()
            expert.w_in.bias.fill_(1)
            expert.w_out.weight.fill_(1)
            expert.w_out.bias.zero_()
    y = moe(torch.ones(4, 2))
    assert y[0].abs().sum() > 0
    torch.testing.assert_close(y[1:], torch.zeros_like(y[1:]))


def test_multisource_moe_preserves_source_shapes():
    block = MultiSourceMoEBlock(num_sources=2, d_model=4, num_experts=4)
    outputs = block([torch.randn(2, 4), torch.randn(1, 4)])
    assert [tuple(x.shape) for x in outputs] == [(2, 4), (1, 4)]
    diagnostics = block.last_source_diagnostics()
    assert len(diagnostics) == 2
    assert all(sum(item["expert_frac"]) == pytest.approx(1.0) for item in diagnostics)
