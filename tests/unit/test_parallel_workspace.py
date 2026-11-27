"""Correctness checks for the dense parallel workspace recurrence."""

import pytest
import torch

from initium.workspace.parallel_workspace import ParallelWorkspace, _affine_scan


def _step_scan(a, b, c0):
    carry = c0
    states = []
    for index in range(a.shape[1]):
        carry = a[:, index, :, None] * carry + b[:, index]
        states.append(carry)
    return torch.stack(states, dim=1)


def test_affine_scan_matches_reference_with_near_zero_and_near_one_decay():
    torch.manual_seed(301)
    batch, length, slots, width = 2, 11, 3, 5
    a = torch.rand(batch, length, slots) * 0.8 + 0.1
    a[:, 2] = 1e-3
    a[:, 8] = 0.999
    b = torch.randn(batch, length, slots, width)
    c0 = torch.randn(batch, slots, width)

    actual = _affine_scan(a, b, c0, chunk=4)
    expected = _step_scan(a, b, c0)

    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)


def test_workspace_initial_delta_is_zero_and_state_carries_across_calls():
    torch.manual_seed(302)
    workspace = ParallelWorkspace(d_model=12, num_slots=4, key_dim=6, chunk=3)
    with torch.no_grad():
        workspace.Wo.weight.normal_(0, 0.1)
    h = torch.randn(2, 9, 12)
    state = workspace.init_state(2)

    full_delta, full_state, full_aux = workspace(h, state)
    first_delta, middle_state, first_aux = workspace(h[:, :4], state)
    second_delta, split_state, second_aux = workspace(h[:, 4:], middle_state)

    torch.testing.assert_close(torch.cat((first_delta, second_delta), dim=1), full_delta)
    torch.testing.assert_close(split_state, full_state, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(
        torch.cat((first_aux["readout"], second_aux["readout"]), dim=1),
        full_aux["readout"],
        atol=1e-5,
        rtol=1e-5,
    )


def test_workspace_readout_is_causal():
    torch.manual_seed(303)
    workspace = ParallelWorkspace(d_model=12, num_slots=4, key_dim=6, chunk=3)
    with torch.no_grad():
        workspace.Wo.weight.normal_(0.0, 0.1)
    h = torch.randn(2, 8, 12)
    changed = h.clone()
    changed[:, 5:] += torch.randn_like(changed[:, 5:]) * 4
    state = workspace.init_state(2)

    delta, _, aux = workspace(h, state)
    changed_delta, _, changed_aux = workspace(changed, state)

    torch.testing.assert_close(delta[:, :5], changed_delta[:, :5], atol=0, rtol=0)
    torch.testing.assert_close(aux["readout"][:, :5], changed_aux["readout"][:, :5], atol=0, rtol=0)


def test_padding_does_not_change_carried_workspace_state():
    torch.manual_seed(304)
    workspace = ParallelWorkspace(d_model=10, num_slots=3, key_dim=5, chunk=4)
    h = torch.randn(2, 7, 10)
    state = workspace.init_state(2)

    _, expected, _ = workspace(h[:, :3], state)
    _, actual, _ = workspace(
        h,
        state,
        valid_mask=torch.tensor([[1, 1, 1, 0, 0, 0, 0], [1, 1, 1, 0, 0, 0, 0]]),
    )

    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)


def test_workspace_parameters_receive_finite_nonzero_gradients():
    torch.manual_seed(305)
    workspace = ParallelWorkspace(d_model=12, num_slots=4, key_dim=6, chunk=3)
    with torch.no_grad():
        workspace.Wo.weight.normal_(0.0, 0.1)
    h = torch.randn(2, 7, 12)
    delta, _, _ = workspace(h, workspace.init_state(2))
    delta.square().mean().backward()

    for parameter in (
        workspace.Wk.weight,
        workspace.Wu.weight,
        workspace.Wv.weight,
        workspace.Wq.weight,
        workspace.slot_keys,
        workspace.read_keys,
        workspace.Wo.weight,
    ):
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
        assert parameter.grad.abs().sum() > 0


def test_workspace_integration_is_step_zero_identity_when_mamba_cuda_is_available():
    pytest.importorskip("mamba_ssm")
    if not torch.cuda.is_available():
        pytest.skip("Mamba-1 workspace integration comparison requires CUDA")
    from initium.mamba_controller.mamba_backbone_parallel import (
        Mamba1,
        MambaBackboneParallel,
    )

    if Mamba1 is None:
        pytest.skip("Mamba-1 extension is unavailable")
    torch.manual_seed(306)
    device = torch.device("cuda")
    backbone = MambaBackboneParallel(
        in_dim=16,
        d_model=16,
        num_blocks=4,
        variant="mamba1",
        d_state=4,
        d_conv=2,
        expand=1,
        device=device,
    ).eval()
    workspace = ParallelWorkspace(d_model=16, num_slots=4, key_dim=8, chunk=4).to(device)
    x = torch.randn(2, 8, 16, device=device)

    with torch.no_grad():
        expected = backbone(x)
        actual, _state, _aux = backbone.forward_with_parallel_workspace(
            x,
            workspace=workspace,
            ws_block=1,
            ws_state=workspace.init_state(2, device=device),
        )

    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
