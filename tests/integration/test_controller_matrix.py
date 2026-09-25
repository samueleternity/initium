import pytest
import torch


@pytest.mark.parametrize(
    "controller,split_graph,variant",
    [
        ("lstm", False, None),
        ("cfc", False, None),
        ("cfc+cfc", False, None),
        ("cfc", True, "cfc"),
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
