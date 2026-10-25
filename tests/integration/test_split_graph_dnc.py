import pytest
import torch

from initium.mamba_controller.split_graph_dnc import SplitGraphDNC


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


@pytest.mark.parametrize("memory_type", ["kv", "delta"])
def test_sequence_memory_zero_init_matches_split_backbone(memory_type):
    torch.manual_seed(21)
    baseline = SplitGraphDNC(
        input_size=8,
        hidden_size=16,
        dnc_enabled=False,
        memory_type="none",
        mamba_variant="cfc",
        cfc_kwargs={"backbone_units": 16},
    ).eval()
    torch.manual_seed(21)
    candidate = SplitGraphDNC(
        input_size=8,
        hidden_size=16,
        dnc_enabled=False,
        memory_type=memory_type,
        memory_heads=2,
        memory_dk=4,
        memory_dv=3,
        delta_chunk=2,
        mamba_variant="cfc",
        cfc_kwargs={"backbone_units": 16},
    ).eval()
    x = torch.randn(2, 5, 8)
    with torch.no_grad():
        expected, _ = baseline(x, reset_experience=True)
        actual, state = candidate(x, reset_experience=True)
    torch.testing.assert_close(actual, expected)
    assert state[1] is not None
