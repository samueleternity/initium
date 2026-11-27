"""Unit coverage for bounded, causal workspace-to-Mamba modulation."""

import math
from types import SimpleNamespace

import pytest
import torch

from initium.mamba_controller.split_graph_dnc import SplitGraphDNC
from initium.workspace.modulation import ModulationHead
from initium.workspace.parallel_workspace import ParallelWorkspace


def _head(source="workspace", targets=("dt", "a", "B", "C", "D"), **kwargs):
    return ModulationHead(
        d_model=8,
        d_inner=12,
        d_state=4,
        mod_blocks=(3,),
        targets=targets,
        source=source,
        rank=6,
        **kwargs,
    )


def _randomize_head_outputs(head):
    if head.source == "static":
        with torch.no_grad():
            for value in head.static.values():
                value.normal_(0, 0.5)
    else:
        with torch.no_grad():
            for output in head.out.values():
                output.weight.normal_(0, 0.5)


def test_zero_initialized_head_and_workspace_preserve_baseline_at_step_zero():
    torch.manual_seed(411)
    head = _head()
    h = torch.randn(2, 5, 8)
    readout = torch.randn_like(h)
    modulation = head(h, readout)

    assert set(modulation[3]) == {"dt", "a", "B", "C", "D"}
    assert all(torch.equal(value, torch.zeros_like(value)) for value in modulation[3].values())

    workspace = ParallelWorkspace(d_model=8, num_slots=3, key_dim=4, chunk=2)
    delta, _, _ = workspace(h, workspace.init_state(2))
    torch.testing.assert_close(h + delta, h, atol=0, rtol=0)

    _randomize_head_outputs(head)
    head.alpha_scale = 0.0
    warmed = head(h, readout)
    assert all(torch.equal(value, torch.zeros_like(value)) for value in warmed[3].values())


def test_head_is_causal_and_zeroes_padding():
    torch.manual_seed(412)
    head = _head()
    _randomize_head_outputs(head)
    h = torch.randn(2, 7, 8)
    r = torch.randn_like(h)
    valid = torch.tensor([[1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 0, 0]])
    changed_h, changed_r = h.clone(), r.clone()
    changed_h[:, 5:] += torch.randn_like(changed_h[:, 5:]) * 3
    changed_r[:, 5:] += torch.randn_like(changed_r[:, 5:]) * 3

    actual = head(h, r, valid)
    changed = head(changed_h, changed_r, valid)

    for target, value in actual[3].items():
        torch.testing.assert_close(value[:, :5], changed[3][target][:, :5], atol=0, rtol=0)
        assert torch.equal(value[1, 5:], torch.zeros_like(value[1, 5:]))


def test_head_aux_regularizers_ignore_padding_and_clear_after_pop():
    torch.manual_seed(4121)
    head = _head(targets=("dt",), lam=0.1, temporal_lam=0.2)
    _randomize_head_outputs(head)
    h = torch.randn(2, 6, 8)
    valid = torch.tensor([[1, 1, 1, 0, 0, 0], [1, 1, 1, 1, 0, 0]])

    head(h, torch.randn_like(h), valid)
    penalty = head.pop_aux_loss()

    assert penalty is not None and torch.isfinite(penalty)
    assert penalty > 0
    assert head.pop_aux_loss() is None


def test_meta_responsiveness_is_zero_initialized_bounded_and_trainable():
    torch.manual_seed(4122)
    head = _head(targets=("dt",), meta_responsiveness=True)
    h = torch.randn(2, 5, 8)
    r = torch.randn_like(h)
    baseline = head(h, r)[3]["dt"]
    assert torch.equal(head.last_diagnostics["responsiveness_rms"], torch.tensor(1.0))

    with torch.no_grad():
        head.responsiveness.weight.fill_(1.0)
        head.out["3_dt"].weight.normal_(0, 0.5)
    modulated = head(h, r)[3]["dt"]
    assert not torch.equal(modulated, baseline)
    assert torch.isfinite(modulated).all()
    modulated.square().mean().backward()
    assert head.responsiveness.weight.grad is not None
    assert head.responsiveness.weight.grad.abs().sum() > 0
    assert head.last_diagnostics["responsiveness_rms"] <= torch.exp(torch.tensor(0.5))


def test_modulation_outputs_obey_alpha_bounds_and_a_scale_stays_contracting():
    torch.manual_seed(413)
    alpha = {"dt": 0.4, "a": 0.5, "B": 0.25, "C": 0.3, "D": 0.2}
    head = _head(alphas=alpha)
    _randomize_head_outputs(head)
    modulation = head(torch.randn(2, 5, 8), torch.randn(2, 5, 8))[3]

    for target, value in modulation.items():
        assert value.abs().max() <= alpha[target] + 1e-6
    assert torch.exp(modulation["dt"]).min() >= torch.exp(torch.tensor(-alpha["dt"]))
    assert torch.exp(modulation["dt"]).max() <= torch.exp(torch.tensor(alpha["dt"]))
    scale = torch.exp(modulation["a"])
    assert scale.min() >= torch.exp(torch.tensor(-alpha["a"]))
    assert scale.max() <= torch.exp(torch.tensor(alpha["a"]))
    negative_A = -torch.exp(torch.randn_like(scale))
    assert (negative_A * scale < 0).all()


def test_a_scale_changes_state_decay_while_preserving_discretized_input(monkeypatch):
    from torch import nn
    from torch.nn import functional as F

    import initium.mamba_controller.mamba_backbone_parallel as backbone

    class FakeMamba(nn.Module):
        def __init__(self):
            super().__init__()
            self.d_state = 2
            self.dt_rank = 2
            self.in_proj = nn.Linear(8, 16)
            self.conv1d = nn.Conv1d(8, 8, kernel_size=1, groups=8)
            self.act = F.silu
            self.x_proj = nn.Linear(8, 6)
            self.dt_proj = nn.Linear(2, 8)
            self.A_log = nn.Parameter(torch.randn(8, 2))
            self.D = nn.Parameter(torch.randn(8))
            self.out_proj = nn.Linear(8, 8)

    def rearrange(value, pattern, **shape):
        if pattern == "b d l -> (b l) d":
            return value.transpose(1, 2).reshape(-1, value.shape[1])
        if pattern == "d (b l) -> b d l":
            return value.reshape(value.shape[0], shape["b"], shape["l"]).permute(1, 0, 2)
        if pattern == "(b l) n -> b n l":
            return value.reshape(shape["b"], shape["l"], value.shape[-1]).permute(0, 2, 1)
        if pattern == "b d l -> b l d":
            return value.transpose(1, 2)
        raise AssertionError(pattern)

    calls = []

    def selective_scan(x, dt, A, B, C, D, **kwargs):
        calls.append((dt.detach().clone(), A.detach().clone(), B.detach().clone()))
        state = torch.zeros(x.shape[0], x.shape[1], A.shape[1])
        return torch.zeros_like(x), state

    monkeypatch.setattr(backbone, "rearrange", rearrange)
    monkeypatch.setattr(backbone, "selective_scan_fn", selective_scan)
    block = backbone._ParallelBlock.__new__(backbone._ParallelBlock)
    nn.Module.__init__(block)
    block.variant = "mamba1"
    block.norm = nn.Identity()
    block.mixer = FakeMamba()
    h = torch.randn(2, 5, 8)

    block.forward_modulated(h, {})
    dt_base, A, B_base = calls[-1]
    scale = 1.4
    block.forward_modulated(h, {"a": torch.full((2, 5), math.log(scale))})
    dt_scaled, A_scaled, B_scaled = calls[-1]

    torch.testing.assert_close(dt_scaled, dt_base * scale)
    torch.testing.assert_close(
        dt_scaled.unsqueeze(2) * B_scaled.unsqueeze(1),
        dt_base.unsqueeze(2) * B_base.unsqueeze(1),
    )
    torch.testing.assert_close(A_scaled, A)
    assert (dt_scaled.unsqueeze(2) * A_scaled[None, :, :, None] < 0).all()


def test_workspace_source_routes_gradients_back_through_readout_and_delta():
    torch.manual_seed(414)
    workspace = ParallelWorkspace(d_model=8, num_slots=3, key_dim=4, chunk=2)
    head = _head(targets=("dt",))
    _randomize_head_outputs(head)
    with torch.no_grad():
        workspace.Wo.weight.normal_(0, 0.1)
    h = torch.randn(2, 6, 8)
    delta, _, aux = workspace(h, workspace.init_state(2))
    output = head(h + delta, aux["readout"])[3]["dt"]
    output.square().mean().backward()

    for parameter in (
        workspace.Wo.weight,
        workspace.Wk.weight,
        workspace.Wv.weight,
        workspace.Wq.weight,
        workspace.slot_keys,
        workspace.read_keys,
    ):
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
        assert parameter.grad.abs().sum() > 0


def test_backbone_source_has_no_gradient_path_into_workspace_readout():
    torch.manual_seed(415)
    workspace = ParallelWorkspace(d_model=8, num_slots=3, key_dim=4, chunk=2)
    head = _head(source="backbone", targets=("dt",))
    _randomize_head_outputs(head)
    h = torch.randn(2, 6, 8)
    _delta, _, aux = workspace(h, workspace.init_state(2))

    head(h, aux["readout"])[3]["dt"].square().mean().backward()

    assert all(parameter.grad is None for parameter in workspace.parameters())


def test_shuffle_ablation_shuffles_contextual_outputs_but_not_static_outputs():
    torch.manual_seed(416)
    h = torch.randn(3, 4, 8)
    readout = torch.randn_like(h)
    valid = torch.ones(3, 4, dtype=torch.bool)
    contextual = _head(targets=("dt",))
    static = _head(source="static", targets=("dt",))
    _randomize_head_outputs(contextual)
    _randomize_head_outputs(static)

    def modulation_fn(head, mode):
        stub = SimpleNamespace(
            modulation_head=head,
            workspace_eval_bypass=False,
            eval_modulation_mode=mode,
        )
        return SplitGraphDNC._modulation_fn(stub, valid)

    control = modulation_fn(contextual, "checkpoint")(h, {"readout": readout})
    shuffled = modulation_fn(contextual, "shuffle")(h, {"readout": readout})
    torch.testing.assert_close(shuffled[3]["dt"], control[3]["dt"].roll(1, 0))

    static_control = modulation_fn(static, "checkpoint")(h, {"readout": readout})
    static_shuffled = modulation_fn(static, "shuffle")(h, {"readout": readout})
    torch.testing.assert_close(static_shuffled[3]["dt"], static_control[3]["dt"])


def test_zero_modulation_matches_normal_mamba1_block_when_cuda_kernel_is_available():
    pytest.importorskip("mamba_ssm")
    if not torch.cuda.is_available():
        pytest.skip("Mamba-1 fused/modulated comparison requires CUDA")
    from initium.mamba_controller.mamba_backbone_parallel import Mamba1, _ParallelBlock

    if Mamba1 is None:
        pytest.skip("Mamba-1 extension is unavailable")
    torch.manual_seed(417)
    block = _ParallelBlock(
        d_model=16,
        variant="mamba1",
        mamba_kwargs={"d_state": 4, "d_conv": 2, "expand": 1},
        device=torch.device("cuda"),
    ).eval()
    x = torch.randn(2, 8, 16, device="cuda")
    zeros = {
        target: torch.zeros(2, 8, size, device="cuda")
        for target, size in (("dt", 16), ("B", 4), ("C", 4), ("D", 16))
    }
    zeros["a"] = torch.zeros(2, 8, device="cuda")

    with torch.no_grad():
        expected = block(x)
        actual, _ = block.forward_modulated(x, zeros)

    torch.testing.assert_close(actual, expected, atol=1e-4, rtol=1e-4)
