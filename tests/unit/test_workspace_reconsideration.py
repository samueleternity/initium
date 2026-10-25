import pytest
import torch

from initium.workspace.broadcast import WorkspaceBroadcast


def _workspace(*, reconsideration: bool) -> WorkspaceBroadcast:
    workspace = WorkspaceBroadcast(
        d_model=2,
        num_experts=1,
        num_slots=2,
        reconsideration_enabled=reconsideration,
        reconsideration_surprise_threshold=0.5,
    )
    with torch.no_grad():
        workspace.query.weight.zero_()
        workspace.key.weight.zero_()
        workspace.value.weight.copy_(torch.eye(2))
        workspace.write_gate.weight.zero_()
        workspace.write_gate.bias.fill_(-10.0)
    return workspace


def test_reconsideration_reopens_gate_and_replaces_occupied_slot_content():
    workspace = _workspace(reconsideration=True)
    deltas = torch.tensor([[[0.0, 1.0]]])
    route_indices = torch.tensor([[0]])
    route_weights = torch.ones(1, 1)
    state = torch.tensor([[[1.0, 0.0], [1.0, 0.0]]])

    _, updated_state = workspace(
        deltas.unsqueeze(1),
        route_indices.unsqueeze(1),
        route_weights.unsqueeze(1),
        state,
    )

    assert torch.allclose(updated_state, torch.tensor([[[0.0, 1.0], [0.0, 1.0]]]))
    diagnostics = workspace.last_diagnostics()
    assert float(diagnostics["reconsideration_trigger_frac"]) == pytest.approx(1.0)
    assert float(diagnostics["reconsideration_changed_frac"]) == pytest.approx(1.0)
    assert float(diagnostics["reconsideration_eligible_count"]) == pytest.approx(2.0)
    assert float(diagnostics["reconsideration_trigger_count"]) == pytest.approx(2.0)
    assert float(diagnostics["reconsideration_changed_count"]) == pytest.approx(2.0)
    assert float(diagnostics["reconsideration_state_delta_rms"]) > 0.0


def test_base_gate_control_does_not_reconsider_the_same_surprising_input():
    workspace = _workspace(reconsideration=False)
    deltas = torch.tensor([[[0.0, 1.0]]])
    route_indices = torch.tensor([[0]])
    route_weights = torch.ones(1, 1)
    state = torch.tensor([[[1.0, 0.0], [1.0, 0.0]]])

    _, updated_state = workspace(
        deltas.unsqueeze(1),
        route_indices.unsqueeze(1),
        route_weights.unsqueeze(1),
        state,
    )

    assert torch.allclose(updated_state, state, atol=1e-4)
    assert "reconsideration_trigger_frac" not in workspace.last_diagnostics()


def test_reconsideration_needs_an_existing_commitment_and_active_writer():
    workspace = _workspace(reconsideration=True)
    deltas = torch.tensor([[[0.0, 1.0]]])
    route_indices = torch.tensor([[0]])
    state = torch.zeros(1, 2, 2)

    # No prior content is present, so ordinary writes remain governed by the
    # closed continuous gate and no reconsideration is eligible.
    _, updated_state = workspace(
        deltas.unsqueeze(1),
        route_indices.unsqueeze(1),
        torch.ones(1, 1, 1),
        state,
    )
    assert torch.allclose(updated_state, state, atol=1e-4)
    assert workspace.last_diagnostics()["reconsideration_trigger_frac"] == 0

    # Likewise, without an active MoE writer, new evidence cannot overwrite a
    # previously committed slot.
    occupied = torch.tensor([[[1.0, 0.0], [1.0, 0.0]]])
    _, no_writer_state = workspace(
        deltas.unsqueeze(1),
        route_indices.unsqueeze(1),
        torch.zeros(1, 1, 1),
        occupied,
    )
    assert torch.equal(no_writer_state, occupied)
    assert workspace.last_diagnostics()["reconsideration_trigger_frac"] == 0


@pytest.mark.parametrize("threshold", [-0.1, 1.1, float("nan")])
def test_reconsideration_rejects_invalid_surprise_threshold(threshold):
    with pytest.raises(ValueError, match="threshold"):
        WorkspaceBroadcast(
            d_model=2,
            num_experts=1,
            reconsideration_surprise_threshold=threshold,
        )
