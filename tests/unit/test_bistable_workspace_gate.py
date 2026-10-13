import torch

from initium.workspace.bistability import BistableWriteGate
from initium.workspace.broadcast import WorkspaceBroadcast


def test_bistable_gate_commits_hard_writes_with_finite_gradients():
    gate = BistableWriteGate(noise_scale=0.75)
    selection = torch.full((1024, 5), 0.2)
    confidence = torch.full((1024,), 0.2)
    torch.manual_seed(1234)

    committed, probabilities = gate(
        selection,
        selection_confidence=confidence,
    )
    assert torch.all((committed == 0) | (committed == 1))
    assert torch.isfinite(probabilities).all()

    committed.sum().backward()
    total_gradient = 0.0
    for parameter in gate.parameters():
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
        total_gradient += float(parameter.grad.abs().sum())
    assert total_gradient > 0.0

    diagnostics = gate.last_diagnostics()
    assert diagnostics["gate_low_frac"] >= 0.1
    assert diagnostics["gate_high_frac"] >= 0.1
    assert diagnostics["gate_mid_frac"] <= 0.2


def test_bistable_gate_noise_is_independent_and_noise_can_be_disabled():
    gate = BistableWriteGate(noise_scale=0.75)
    selection = torch.full((128, 5), 0.2)
    confidence = torch.full((128,), 0.2)

    torch.manual_seed(10)
    _, noisy_a = gate(selection, selection_confidence=confidence)
    torch.manual_seed(11)
    _, noisy_b = gate(selection, selection_confidence=confidence)
    assert not torch.equal(noisy_a, noisy_b)

    torch.manual_seed(10)
    _, quiet_a = gate(
        selection,
        selection_confidence=confidence,
        noise_scale=0.0,
    )
    torch.manual_seed(11)
    _, quiet_b = gate(
        selection,
        selection_confidence=confidence,
        noise_scale=0.0,
    )
    assert torch.equal(quiet_a, quiet_b)

    torch.manual_seed(42)
    expected_global = torch.rand(4)
    torch.manual_seed(42)
    gate(selection, selection_confidence=confidence)
    actual_global = torch.rand(4)
    assert torch.equal(actual_global, expected_global)


def test_bistable_gate_noise_stream_roundtrips_in_state_dict():
    gate = BistableWriteGate()
    selection = torch.full((16, 5), 0.2)
    confidence = torch.full((16,), 0.2)
    checkpoint = gate.state_dict()

    _, expected = gate(selection, selection_confidence=confidence)
    restored = BistableWriteGate()
    restored.load_state_dict(checkpoint)
    _, actual = restored(selection, selection_confidence=confidence)
    assert torch.equal(actual, expected)


def test_workspace_bistable_variant_preserves_state_contract_and_logs_modes():
    torch.manual_seed(3)
    workspace = WorkspaceBroadcast(
        d_model=8,
        num_experts=4,
        num_slots=5,
        gate_variant="bistable",
    )
    specialists = torch.randn(2, 3, 4, 8)
    routes = torch.zeros(2, 3, 1, dtype=torch.long)
    weights = torch.full((2, 3, 1), 0.2)
    state = workspace.init_state(2, device=specialists.device, dtype=specialists.dtype)

    output, next_state = workspace(
        specialists,
        routes,
        weights,
        state,
    )
    diagnostics = workspace.last_diagnostics()
    assert output.shape == (2, 3, 8)
    assert next_state.shape == (2, 5, 8)
    assert diagnostics["gate_low_frac"] >= 0
    assert diagnostics["gate_mid_frac"] >= 0
    assert diagnostics["gate_high_frac"] >= 0
    assert diagnostics["writer_frequency"].shape == (4,)
