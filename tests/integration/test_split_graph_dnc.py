import pytest
import torch


def test_split_graph_forward_backward_and_no_memory_contract(build_tiny_rnn):
    model, _, _, _ = build_tiny_rnn("lstm", split_graph=True, combine_reads=False)
    x = torch.randn(2, 3, 8)
    addressing_inputs = []
    handle = model.output.register_forward_pre_hook(
        lambda _module, args: addressing_inputs.append(args[0].detach().clone())
    )
    output, _ = model(x, pass_through_memory=False)
    handle.remove()
    assert output.shape == (3, 2, 8)
    backbone_output = model.backbone(x)
    observed = torch.stack(addressing_inputs, dim=0)
    torch.testing.assert_close(observed[:, :, :16], backbone_output.transpose(0, 1))
    torch.testing.assert_close(observed[:, :, 16:], torch.zeros_like(observed[:, :, 16:]))
    output.square().mean().backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())


def test_split_graph_resume_requires_memory_state(build_tiny_rnn):
    model, _, _, _ = build_tiny_rnn("lstm", split_graph=True)
    x = torch.randn(2, 3, 8)
    with pytest.raises(ValueError, match="requires a resumed"):
        model(x, start_step=1)


def test_split_graph_resumed_tail_matches_full_sequence(build_tiny_rnn):
    model, _, _, _ = build_tiny_rnn("lstm", split_graph=True)
    model.eval()
    x = torch.randn(2, 3, 8)
    with torch.no_grad():
        full, _ = model(x)
        _, state_after_prefix = model(x[:, :1, :])
        tail, _ = model(x, hx=state_after_prefix, start_step=1)
    torch.testing.assert_close(tail, full[1:])
