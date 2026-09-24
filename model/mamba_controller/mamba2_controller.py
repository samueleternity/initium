"""
file: mamba2_controller.py -- v1

Companion to mamba_controller.py (Mamba-1), implementing a genuine,
standalone Mamba-2 DNC controller: MambaDNC's rnn_type='mamba2'. This is
NOT the same thing as split_graph_dnc.py's mamba_variant='mamba2' backbone
(mamba_backbone_parallel.py) -- that backbone drives mamba_ssm's own
Mamba2.forward() over the WHOLE sequence at once (Option 5's parallel
"no memory dependency" half). This file drives Mamba2 ONE DNC timestep at
a time, interleaved with dnc.memory.Memory reads/writes exactly like
mamba_controller.py's Mamba-1 controller does -- i.e. this is Alternate
Phase 3 Step 1's controller swap, generalized to Mamba-2, not Option 5's
split-graph mechanism. The two are orthogonal and can't substitute for
each other: split_graph_dnc.py never threads a read vector into its
backbone at all (by design -- see that file's module docstring), while
this controller's whole point, like mamba_controller.py's, is that the
read vector from step t-1 is part of this controller's input at step t.

Library used: `mamba-ssm`, specifically `mamba_ssm.modules.mamba2.Mamba2`
-- reused purely as a parameter container (in_proj, conv1d, A_log, D,
dt_bias, out_proj), exactly the same "construct via the library, bypass
its forward()/step()" strategy mamba_controller.py's MambaControllerCell
already uses for Mamba-1 -- see that file's "Why not just call
Mamba.step()?" section, which applies here unchanged: Mamba2's own
`.step()` (mamba_ssm/modules/mamba2.py) is the right per-token
granularity but mutates conv_state/ssm_state in place via `.copy_(...)`,
which is fine for `torch.no_grad()` autoregressive decoding but breaks
backprop-through-time the moment more than one `.step()` call is chained
under a shared loss. This file reimplements that same step's math
line-for-line, out-of-place, reusing the parameter container's submodules
and constants only.

ASSUMPTION FLAGGED FOR VERIFICATION (same convention as split_graph_dnc.py's
own flagged assumption about Memory's return shape): the exact attribute
names/shapes below (self.mamba2.d_ssm/nheads/headdim/ngroups/dt_bias/A_log/D,
in_proj's [z, xBC, dt] split sizes) are reconstructed from mamba_ssm's public
Mamba2 source, not verified against a live install. Run one batch through
this module and confirm out.shape == (B, d_model) and that no shape-mismatch
error fires before trusting a full run.

--------------------------------------------------------------------------
Differences from Mamba-1 that matter for this reimplementation
--------------------------------------------------------------------------
Mamba-2's SSD parameterization (Dao & Gu 2024) differs from Mamba-1's S6
in ways that change the per-step math, not just the constructor kwargs:
  - A is a SCALAR PER HEAD (`A_log` has shape (nheads,)), not a per-
    (channel, state) matrix like Mamba-1's `A_log` (d_inner, d_state) --
    this is exactly the "scalar-identity SSM" structure the SSD paper's
    Section 5.1 shows is dual to 1-semiseparable structured attention
    (Section 5.2/5.3, Corollary 5.1). So `dA` here is `exp(dt * A)`
    broadcast per head, not `exp(einsum("bd,dn->bdn", dt, A))` per
    (channel, state) as in Mamba-1's cell.
  - The recurrent state has an extra head axis: `ssm_state` is
    `(B, nheads, headdim, d_state)` (Mamba-1's is `(B, d_inner, d_state)`,
    with d_inner playing the role nheads*headdim plays here).
  - `in_proj` produces `[z, x, B, C, dt]` in one shot (Mamba-2's "parallel
    parameter projections", Section 7.1) rather than Mamba-1's
    `in_proj -> [x, z]` followed by a separate `x_proj -> [dt, B, C]` on
    the post-conv activation. Concretely, Mamba-2's `dt` is already
    available before the conv step, and the conv only ever acts on
    `[x, B, C]` (packed as `xBC`), never on `z` or `dt`.
  - Mamba-2 has an extra gated RMSNorm (`self.norm`, Section 7.1 "Extra
    Normalization") applied to `y` before `out_proj`, when `rmsnorm=True`.
    We construct the underlying `Mamba2` with `rmsnorm=False` here -- same
    rationale as `use_fast_path=False` on the Mamba-1 cell: `RMSNormGated`
    (mamba_ssm.ops.triton.layernorm_gated) is a Triton kernel with no
    CPU/no-Triton fallback, so keeping it out avoids a second hard
    GPU-kernel dependency beyond `mamba_ssm` itself. With `rmsnorm=False`,
    Mamba-2's own step() reduces to the same `y = y * silu(z)` gating
    Mamba-1 uses -- only the extra post-hoc normalization layer is
    skipped, not any of SSD's own math.
  - `ngroups` (B/C sharing across heads, the MVA/MQA/MKA head-pattern
    distinction of Section 7.2) is fixed at 1 here, matching every other
    Mamba-2 use in this project (mamba_backbone_parallel.py also never
    exposes it) -- Proposition 7.2 / Table 5 in the paper is what
    motivates Mamba-1's own MVA pattern (B/C shared across all channels)
    as the strongest single-head-pattern default; Mamba-2's ngroups=1
    case reduces to that same sharing.
  - `D_has_hdim=False` (Mamba-2's own default) is kept: `D` is per-head
    (nheads,), not per-(head,headdim).

Everything else -- the Cell/Block/Wrapper split, the pre-norm residual
pattern, the nn.LSTM-compatible call convention, the fp32-forced SSM math
with dt/A_log clamping and hard state clamps (chronic-NaN fix, see
mamba_controller.py's step() for the original diagnosis) -- is copied as
directly as possible from mamba_controller.py, changed only where
Mamba-2's parameterization forces a different shape or formula.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from MoE.moe_layer import MoEBlock

try:
    from mamba_ssm.modules.mamba2 import Mamba2
except ImportError as _mamba2_import_error:  # pragma: no cover - environment-dependent
    Mamba2 = None
    _MAMBA2_IMPORT_ERROR = (
        "mamba2_controller.py requires the `mamba-ssm` package "
        "(with Mamba-2 support -- mamba_ssm.modules.mamba2.Mamba2). "
        "Original import error: "
        f"{_mamba2_import_error}"
    )
else:
    _MAMBA2_IMPORT_ERROR = None


def _require_mamba2_ssm() -> None:
    if Mamba2 is None:
        raise ImportError(_MAMBA2_IMPORT_ERROR)


# ==========================================================================
# 1. Mamba2ControllerCell -- one Mamba-2 block, single-timestep, BPTT-safe
# ==========================================================================
class Mamba2ControllerCell(nn.Module):
    """One Mamba-2 (SSD) block, driven one timestep at a time. See module
    docstring for the full design rationale and how this differs from
    mamba_controller.py's Mamba-1 MambaControllerCell.

    State: `(conv_state, ssm_state)`, shapes `(B, conv_dim, d_conv)` and
    `(B, nheads, headdim, d_state)` respectively, matching
    `Mamba2.allocate_inference_cache()`'s own shapes (we don't call that
    method -- construct zero state directly -- but stay shape-compatible
    with it on purpose, same convention as the Mamba-1 cell).
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 64,
        d_conv: int = 4,
        expand: int = 2,
        headdim: int = 64,
        ngroups: int = 1,
        layer_idx: int | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        _require_mamba2_ssm()

        # rmsnorm=False, use_mem_eff_path=False: same "portable,
        # dependency-light path" rationale as the Mamba-1 cell's
        # use_fast_path=False -- see module docstring. We never call
        # self.mamba2.forward()/.step(); only its submodules (in_proj,
        # conv1d, A_log, D, dt_bias, out_proj) are used, by step() below.
        self.mamba2 = Mamba2(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            headdim=headdim,
            ngroups=ngroups,
            rmsnorm=False,
            use_mem_eff_path=False,
            layer_idx=layer_idx,
            device=device,
            dtype=dtype,
        )
        self.d_model = d_model
        self.d_inner = self.mamba2.d_inner
        self.d_ssm = self.mamba2.d_ssm  # == d_inner here (d_ssm kwarg left at default None)
        self.d_state = self.mamba2.d_state
        self.d_conv = self.mamba2.d_conv
        self.headdim = self.mamba2.headdim
        self.ngroups = self.mamba2.ngroups
        self.nheads = self.mamba2.nheads
        self.conv_dim = self.d_ssm + 2 * self.ngroups * self.d_state

    def init_state(
        self, batch_size: int, device: torch.device | None = None, dtype: torch.dtype | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Zero-initialized (conv_state, ssm_state) -- same zero-init
        rationale as the Mamba-1 cell (Mamba2.allocate_inference_cache's
        own convention is also zero)."""
        conv_state = torch.zeros(batch_size, self.conv_dim, self.d_conv, device=device, dtype=dtype)
        ssm_state = torch.zeros(
            batch_size, self.nheads, self.headdim, self.d_state, device=device, dtype=dtype
        )
        return conv_state, ssm_state

    def step(
        self,
        hidden_states: torch.Tensor,
        conv_state: torch.Tensor,
        ssm_state: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Single-timestep SSD update. `hidden_states`: (B, d_model).

        Line-for-line equivalent to `Mamba2.step()`'s ngroups==1,
        non-fast-path branch (mamba_ssm/modules/mamba2.py), except every
        state update is out-of-place (`new_state = ...`, no `.copy_()`) --
        see module docstring / mamba_controller.py's "Why not just call
        Mamba.step()?" section for why this is required under BPTT.
        """
        m = self.mamba2
        dtype = hidden_states.dtype

        # ---- parallel [z, x, B, C, dt] projection ------------------------
        zxbcdt = m.in_proj(
            hidden_states
        )  # (B, 2*d_ssm + 2*ngroups*d_state + nheads), d_mlp==0 here
        z, xBC, dt = torch.split(
            zxbcdt, [self.d_ssm, self.d_ssm + 2 * self.ngroups * self.d_state, self.nheads], dim=-1
        )

        # ---- causal depthwise conv over [x, B, C], functional (rolling
        # window) -- same shift-and-append pattern as the Mamba-1 cell,
        # just over the wider xBC channel set.
        new_conv_state = torch.cat([conv_state[:, :, 1:], xBC.unsqueeze(-1)], dim=-1)
        conv_weight = m.conv1d.weight.squeeze(1)  # (conv_dim, 1, d_conv) -> (conv_dim, d_conv)
        xBC = torch.sum(new_conv_state * conv_weight, dim=-1)  # (B, conv_dim)
        if m.conv1d.bias is not None:
            xBC = xBC + m.conv1d.bias
        xBC = m.act(xBC).to(dtype=dtype)

        x, B, C = torch.split(
            xBC, [self.d_ssm, self.ngroups * self.d_state, self.ngroups * self.d_state], dim=-1
        )

        # ---- SSD recurrence, forced fp32 (same chronic-NaN fix as the
        # Mamba-1 cell's step() -- ssm_state persists across an entire
        # episode, so a single fp16 overflow anywhere in this chain
        # poisons every later step; see mamba_controller.py's FIX comment
        # for the full diagnosis, which applies here unchanged).
        with torch.autocast(device_type=hidden_states.device.type, enabled=False):
            dt32 = dt.float()
            x32 = x.float()
            B32 = B.float()
            C32 = C.float()
            dt_bias32 = m.dt_bias.float()
            A_log_c = m.A_log.float().clamp(
                min=-20.0, max=20.0
            )  # same lower-bound guard as Mamba-1's cell
            A32 = -torch.exp(
                A_log_c
            )  # (nheads,) -- scalar-per-head, SSD's defining structure (paper Sec. 5.1)

            dt32 = F.softplus(dt32 + dt_bias32)  # (B, nheads)
            dt32 = dt32.clamp(
                min=1e-6, max=100.0
            )  # normal dt is ~0.001-0.1, same bound as Mamba-1's cell
            dA = torch.exp(
                dt32 * A32
            )  # (B, nheads) -- scalar decay per head, not per (channel,state)

            x32 = x32.view(x32.shape[0], self.nheads, self.headdim)  # (B, H, P)
            # ngroups == 1: B/C are shared across every head (Mamba-2's own
            # MVA-equivalent default here -- see module docstring).
            B32 = B32.view(B32.shape[0], self.d_state)  # (B, N)
            C32 = C32.view(C32.shape[0], self.d_state)  # (B, N)

            dBx = torch.einsum("bh,bn,bhp->bhpn", dt32, B32, x32)  # (B, H, P, N)
            new_ssm_state32 = ssm_state.float() * dA.view(-1, self.nheads, 1, 1) + dBx
            new_ssm_state32 = new_ssm_state32.clamp(
                min=-1e4, max=1e4
            )  # same hard stop as Mamba-1's cell

            y32 = torch.einsum("bhpn,bn->bhp", new_ssm_state32, C32)  # (B, H, P)
            y32 = y32 + m.D.float().view(1, self.nheads, 1) * x32
            y32 = y32.reshape(y32.shape[0], self.d_ssm)  # (B, d_ssm)
            y32 = y32 * m.act(z).float()  # rmsnorm=False -> plain silu gate
            y32 = y32.clamp(min=-1e4, max=1e4)  # stay in fp16 range before out_proj
        new_ssm_state = new_ssm_state32.to(dtype)
        y = y32.to(dtype)

        out = m.out_proj(y)  # (B, d_model) -- back under ambient autocast
        return out, new_conv_state, new_ssm_state


# ==========================================================================
# 2. Mamba2ControllerBlock -- pre-norm residual wrapper around one cell
# ==========================================================================
class Mamba2ControllerBlock(nn.Module):
    """Add -> LN -> Mixer residual block around one `Mamba2ControllerCell`,
    identical pattern to mamba_controller.py's `MambaControllerBlock`."""

    def __init__(
        self,
        d_model: int,
        d_state: int = 64,
        d_conv: int = 4,
        expand: int = 2,
        headdim: int = 64,
        ngroups: int = 1,
        layer_idx: int | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.norm = nn.LayerNorm(d_model, device=device, dtype=dtype)
        self.cell = Mamba2ControllerCell(
            d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            headdim=headdim,
            ngroups=ngroups,
            layer_idx=layer_idx,
            device=device,
            dtype=dtype,
        )

    def init_state(
        self, batch_size: int, device: torch.device | None = None, dtype: torch.dtype | None = None
    ):
        return self.cell.init_state(batch_size, device=device, dtype=dtype)

    def step(self, x: torch.Tensor, state: tuple[torch.Tensor, torch.Tensor]):
        conv_state, ssm_state = state
        out, new_conv_state, new_ssm_state = self.cell.step(self.norm(x), conv_state, ssm_state)
        return x + out, (new_conv_state, new_ssm_state)


# ==========================================================================
# 3. Mamba2ControllerWrapper -- stack of blocks, nn.LSTM-compatible call API
# ==========================================================================
class Mamba2ControllerWrapper(nn.Module):
    """Stacks `num_blocks` `Mamba2ControllerBlock`s and exposes the exact
    same `dnc.dnc.DNC._layer_forward`-compatible calling convention as
    mamba_controller.py's `MambaControllerWrapper` -- see that class's
    docstring for the exact shape contract. Identical MoE-interleaving
    support (Option 4, moe_layer.py), same convention as the Mamba-1
    wrapper, so `--moe` works unchanged for `--controller mamba2`.
    """

    def __init__(
        self,
        in_dim: int,
        d_model: int,
        num_blocks: int = 2,
        moe_enabled: bool = False,
        moe_num_experts: int = 8,
        moe_expert_dim: int | None = None,
        moe_capacity_factor: float = 1.5,
        moe_load_balance_alpha: float = 0.01,
        moe_top_k: int = 1,
        d_state: int = 64,
        d_conv: int = 4,
        expand: int = 2,
        headdim: int = 64,
        ngroups: int = 1,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_blocks = num_blocks
        # Same first-block-only dimension-matching adapter as MambaControllerWrapper.
        self.in_adapter: nn.Module = (
            nn.Identity()
            if in_dim == d_model
            else nn.Linear(in_dim, d_model, device=device, dtype=dtype)
        )
        self.blocks = nn.ModuleList(
            [
                Mamba2ControllerBlock(
                    d_model,
                    d_state=d_state,
                    d_conv=d_conv,
                    expand=expand,
                    headdim=headdim,
                    ngroups=ngroups,
                    layer_idx=i,
                    device=device,
                    dtype=dtype,
                )
                for i in range(num_blocks)
            ]
        )
        self.moe_enabled = moe_enabled
        self.moe_blocks: nn.ModuleList | None = None
        if moe_enabled:
            self.moe_blocks = nn.ModuleList(
                [
                    MoEBlock(
                        d_model,
                        num_experts=moe_num_experts,
                        expert_dim=moe_expert_dim,
                        capacity_factor=moe_capacity_factor,
                        load_balance_alpha=moe_load_balance_alpha,
                        top_k=moe_top_k,
                        device=device,
                        dtype=dtype,
                    )
                    for _ in range(num_blocks)
                ]
            )

    def init_state(
        self, batch_size: int, device: torch.device | None = None, dtype: torch.dtype | None = None
    ):
        if device is None or dtype is None:
            p = next(self.parameters())
            device = device if device is not None else p.device
            dtype = dtype if dtype is not None else p.dtype
        return [blk.init_state(batch_size, device=device, dtype=dtype) for blk in self.blocks]

    def forward(self, input: torch.Tensor, hx):
        # input: (B, 1, in_dim) -- same single-timestep contract as
        # MambaControllerWrapper.forward(); see that class's docstring.
        assert input.dim() == 3 and input.size(1) == 1, (
            "Mamba2ControllerWrapper only supports single-timestep calls "
            f"(got shape {tuple(input.shape)}); this mirrors how DNC drives "
            "nn.LSTM one step at a time, never a full sequence at once."
        )
        x = input.squeeze(1)
        x = self.in_adapter(x)

        if hx is None:
            hx = self.init_state(x.size(0), device=x.device, dtype=x.dtype)

        new_hx = []
        for i, (block, state) in enumerate(zip(self.blocks, hx)):
            x, new_state = block.step(x, state)
            if self.moe_enabled:
                x = self.moe_blocks[i](x)
            new_hx.append(new_state)

        return x.unsqueeze(1), new_hx
