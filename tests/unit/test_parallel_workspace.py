"""Unit tests for dense parallel workspace and chunked affine recurrence (Phase A)."""

from __future__ import annotations

import pytest
import torch
from torch import nn

from initium.mamba_controller.split_graph_dnc import SplitGraphDNC
from initium.workspace.parallel_workspace import ParallelWorkspace, _affine_scan


def _python_affine_scan_ref(
    a: torch.Tensor,
    b: torch.Tensor,
    c0: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sequential reference recurrence c_t = a_t * c_{t-1} + b_t."""
    B, T, S, D = b.shape
    c = torch.zeros(B, T, S, D, dtype=torch.float32, device=b.device)
    curr = c0.float() if c0 is not None else torch.zeros(B, S, D, dtype=torch.float32, device=b.device)
    for t in range(T):
        curr = a[:, t].unsqueeze(-1) * curr + b[:, t]
        c[:, t] = curr
    return c, curr


@pytest.mark.parametrize("T", [16, 32, 75])
def test_affine_scan_correctness_vs_python_loop(T: int) -> None:
    """A5-1: Scan correctness vs Python loop reference, including a near 0 and a near 1."""
    torch.manual_seed(42)
    B, S, D = 2, 5, 16

    # General values
    a_gen = torch.rand(B, T, S) * 0.8 + 0.1
    b_gen = torch.randn(B, T, S, D)
    c0_gen = torch.randn(B, S, D)
    c_scan, c_T_scan = _affine_scan(a_gen, b_gen, c0_gen, chunk_size=32)
    c_ref, c_T_ref = _python_affine_scan_ref(a_gen, b_gen, c0_gen)
    torch.testing.assert_close(c_scan, c_ref, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(c_T_scan, c_T_ref, atol=1e-5, rtol=1e-5)

    # Extreme case 1: a near 0 (u = 0.999 => a = 1 - 0.999 = 0.001)
    a_low = torch.full((B, T, S), 0.001)
    b_low = torch.randn(B, T, S, D)
    c_scan_low, c_T_scan_low = _affine_scan(a_low, b_low, c0_gen, chunk_size=32)
    c_ref_low, c_T_ref_low = _python_affine_scan_ref(a_low, b_low, c0_gen)
    torch.testing.assert_close(c_scan_low, c_ref_low, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(c_T_scan_low, c_T_ref_low, atol=1e-5, rtol=1e-5)

    # Extreme case 2: a near 1 (u ≈ 0 => a = 0.99999)
    a_high = torch.full((B, T, S), 0.99999)
    b_high = torch.randn(B, T, S, D)
    c_scan_high, c_T_scan_high = _affine_scan(a_high, b_high, c0_gen, chunk_size=32)
    c_ref_high, c_T_ref_high = _python_affine_scan_ref(a_high, b_high, c0_gen)
    torch.testing.assert_close(c_scan_high, c_ref_high, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(c_T_scan_high, c_T_ref_high, atol=1e-5, rtol=1e-5)


def test_parallel_workspace_causality() -> None:
    """A5-2: Perturb h at t=T/2; outputs at t<T/2 must be bit-identical."""
    torch.manual_seed(101)
    B, T, D = 2, 32, 32
    workspace = ParallelWorkspace(d_model=D, num_slots=5, key_dim=16, chunk=16)

    h = torch.randn(B, T, D)
    delta1, c_T1, aux1 = workspace(h)

    # Perturb h starting from t = T // 2
    split_t = T // 2
    h_perturbed = h.clone()
    h_perturbed[:, split_t:] += torch.randn_like(h_perturbed[:, split_t:]) * 2.0

    delta2, c_T2, aux2 = workspace(h_perturbed)

    # Timesteps t < split_t must be bit-identical
    assert torch.equal(delta1[:, :split_t], delta2[:, :split_t])
    assert torch.equal(aux1["c"][:, :split_t], aux2["c"][:, :split_t])
    assert torch.equal(aux1["r_t"][:, :split_t], aux2["r_t"][:, :split_t])

    # Timesteps t >= split_t must differ
    assert not torch.equal(delta1[:, split_t:], delta2[:, split_t:])


def test_parallel_workspace_step_zero_identity() -> None:
    """A5-3: Model with parallel workspace at init matches model without it (Wo=0)."""
    torch.manual_seed(202)
    baseline = SplitGraphDNC(
        input_size=16,
        hidden_size=32,
        num_backbone_blocks=3,
        mamba_variant="mamba1",
        dnc_enabled=False,
        memory_type="none",
        workspace_enabled=False,
        workspace_parallel_dense=False,
    ).eval()

    torch.manual_seed(202)
    candidate = SplitGraphDNC(
        input_size=16,
        hidden_size=32,
        num_backbone_blocks=3,
        mamba_variant="mamba1",
        dnc_enabled=False,
        memory_type="none",
        workspace_enabled=False,
        workspace_parallel_dense=True,
        workspace_block_index=1,
        workspace_key_dim=16,
        workspace_scan_chunk=16,
    ).eval()

    x = torch.randn(2, 20, 16)
    with torch.no_grad():
        baseline_out, _ = baseline(x, reset_experience=True)
        candidate_out, _ = candidate(x, reset_experience=True)

    torch.testing.assert_close(candidate_out, baseline_out, atol=1e-6, rtol=1e-6)


def test_parallel_workspace_carry_equivalence() -> None:
    """A5-4: Run [0:T] in one call vs [0:T/2] then [T/2:T] with carried state."""
    torch.manual_seed(303)
    B, T, D = 2, 40, 32
    workspace = ParallelWorkspace(d_model=D, num_slots=5, key_dim=16, chunk=16)

    # Give Wo non-zero weights so delta is non-trivial
    nn.init.normal_(workspace.Wo.weight, std=0.05)
    nn.init.normal_(workspace.Wo.bias, std=0.05)

    h = torch.randn(B, T, D)

    # 1-call full pass
    delta_full, c_T_full, aux_full = workspace(h)

    # 2-call segmented pass
    mid = T // 2
    delta_p1, c_T_p1, aux_p1 = workspace(h[:, :mid])
    delta_p2, c_T_p2, aux_p2 = workspace(h[:, mid:], state=c_T_p1)

    delta_concat = torch.cat([delta_p1, delta_p2], dim=1)
    c_concat = torch.cat([aux_p1["c"], aux_p2["c"]], dim=1)

    torch.testing.assert_close(delta_concat, delta_full, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(c_T_p2, c_T_full, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(c_concat, aux_full["c"], atol=1e-5, rtol=1e-5)


def test_parallel_workspace_gradients_finite_and_nonzero() -> None:
    """A5-5: Gradients finite and nonzero for Wk, Wu, Wv, Wq, κ, κ', Wo after perturbing Wo."""
    torch.manual_seed(404)
    B, T, D = 2, 16, 32
    workspace = ParallelWorkspace(d_model=D, num_slots=5, key_dim=16, chunk=16)

    # Perturb Wo since zero-init yields zero gradients upstream of Wo
    nn.init.normal_(workspace.Wo.weight, std=0.05)
    nn.init.normal_(workspace.Wo.bias, std=0.05)

    h = torch.randn(B, T, D, requires_grad=True)
    delta, c_T, _ = workspace(h)

    loss = delta.sum() + c_T.sum()
    loss.backward()

    params_to_check = [
        ("Wk.weight", workspace.Wk.weight),
        ("Wk.bias", workspace.Wk.bias),
        ("Wu.weight", workspace.Wu.weight),
        ("Wu.bias", workspace.Wu.bias),
        ("Wv.weight", workspace.Wv.weight),
        ("Wv.bias", workspace.Wv.bias),
        ("Wq.weight", workspace.Wq.weight),
        ("Wq.bias", workspace.Wq.bias),
        ("slot_keys (κ)", workspace.slot_keys),
        ("read_keys (κ')", workspace.read_keys),
        ("Wo.weight", workspace.Wo.weight),
        ("Wo.bias", workspace.Wo.bias),
        ("read_gate", workspace.read_gate),
    ]

    for name, param in params_to_check:
        assert param.grad is not None, f"Gradient is None for {name}"
        assert torch.isfinite(param.grad).all(), f"Gradient has non-finite values for {name}"
        assert param.grad.abs().sum() > 0.0, f"Gradient is all zeros for {name}"


def test_parallel_workspace_padding_zero_write() -> None:
    """A5-6: Zero-padded tokens must not write to slot state."""
    torch.manual_seed(505)
    B, T, D = 2, 10, 32
    workspace = ParallelWorkspace(d_model=D, num_slots=5, key_dim=16, chunk=16)

    h = torch.randn(B, T, D)
    # Mask: steps 4, 5, 6 are padding (0), rest are valid (1)
    valid_mask = torch.ones(B, T, dtype=torch.bool)
    valid_mask[:, 4:7] = False

    _, c_T, aux = workspace(h, valid_mask=valid_mask)
    c = aux["c"]

    # Slot state must not change on padded steps 4, 5, 6
    assert torch.equal(c[:, 4], c[:, 3]), "Slot state changed on padded token 4"
    assert torch.equal(c[:, 5], c[:, 4]), "Slot state changed on padded token 5"
    assert torch.equal(c[:, 6], c[:, 5]), "Slot state changed on padded token 6"

    # Perturbing input at padded steps must not affect carried state or future states
    h_perturbed = h.clone()
    h_perturbed[:, 4:7] += 50.0

    _, c_T_pert, aux_pert = workspace(h_perturbed, valid_mask=valid_mask)
    assert torch.equal(c_T_pert, c_T), "Carried state changed despite padding mask"
    assert torch.equal(aux_pert["c"][:, 7:], aux["c"][:, 7:]), "Future slot states affected by padded tokens"
