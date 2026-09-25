import importlib.util

import pytest
import torch


def test_cfc_with_moe_forward_and_aux_loss(build_tiny_rnn):
    model, _, _, _ = build_tiny_rnn("cfc", moe_enabled=True, moe_num_experts=4)
    output, _ = model(torch.randn(2, 2, 8))
    assert output.shape[:2] == (2, 2)
    output.square().mean().backward()
    moe_layers = model.moe_layers
    assert moe_layers
    assert torch.isfinite(moe_layers[0].pop_aux_loss())


@pytest.mark.gpu
@pytest.mark.skipif(
    importlib.util.find_spec("mamba_ssm") is None, reason="mamba-ssm is unavailable"
)
def test_hybrid_mamba_cfc_with_moe(build_tiny_rnn):
    model, _, _, _ = build_tiny_rnn("mamba+cfc", moe_enabled=True, moe_num_experts=4)
    output, _ = model(torch.randn(2, 2, 8))
    output.square().mean().backward()
    assert model.moe_layers
    assert all(torch.isfinite(layer.pop_aux_loss()) for layer in model.moe_layers)
