import importlib.util

import pytest
import torch

requires_mamba_ssm = pytest.mark.skipif(
    importlib.util.find_spec("mamba_ssm") is None, reason="mamba-ssm is unavailable"
)


@pytest.mark.parametrize(
    "controller,split_graph,variant",
    [
        ("lstm", False, None),
        ("cfc", False, None),
        ("cfc+cfc", False, None),
        pytest.param("mamba", False, None, marks=requires_mamba_ssm),
        pytest.param("mamba2", False, None, marks=requires_mamba_ssm),
        pytest.param("mamba3", False, None, marks=requires_mamba_ssm),
        pytest.param("mamba+cfc", False, None, marks=requires_mamba_ssm),
        ("cfc", True, "cfc"),
        pytest.param("mamba", True, "mamba1", marks=requires_mamba_ssm),
    ],
)
def test_controller_forward_backward(build_tiny_rnn, controller, split_graph, variant):
    options = {"split_graph": True, "split_graph_variant": variant} if split_graph else {}
    model, _, _, _ = build_tiny_rnn(controller, **options)
    x = torch.randn(3, 2, 8)
    output, _ = model(x, None) if not split_graph else model(x)
    # Both controller implementations accept batch-first input and return
    # sequence-first output, so the returned leading dimension is T.
    assert output.shape[0] == x.shape[1]
    loss = output.float().square().mean()
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.requires_grad and p.grad is not None]
    assert grads
    assert all(torch.isfinite(grad).all() for grad in grads)
