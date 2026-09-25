"""
file: LNN_controller/cfc_controller.py -- v1

Standalone Closed-form Continuous-time (CfC) DNC controller (Hasani et al.,
"Closed-form Continuous-time Neural Networks", arXiv 2106.13898):
MambaDNC's rnn_type='cfc'. Uses the `ncps` package's CfC module as-is
(pip install ncps -> from ncps.torch import CfC), driven ONE DNC timestep at
a time as a length-1 sequence, interleaved with Memory reads/writes exactly
like mamba_controller.py's controllers. ncps' CfC forward is out-of-place
(returns new h), so it is BPTT-safe with no custom step() needed. Its state
is bounded (convex mix of tanh heads), so no fp32-forcing/clamp workarounds.
timespans are not used (ts=1.0 per step): this task has no timestamps.

CONTROLLER-WRAPPER PROTOCOL (what makes this composable -- see also
chained_controller.py):
    wrapper.init_state(batch_size, device=None, dtype=None) -> state
    wrapper(x: (B,1,in_dim), state)                         -> (out: (B,1,d_model), new_state)
    wrapper.d_model, wrapper.in_adapter, wrapper.moe_enabled, wrapper.moe_blocks
Any object following this can be a DNC controller, a split-graph combiner,
or a stage in a ChainedControllerWrapper.

Block = pre-norm residual (Add -> LN -> CfC), same pattern as the Mamba blocks.
State per block: h (B, units) fp32, or (h, c) if mixed_memory=True (CfC-mmRNN).
"""

from __future__ import annotations

import torch
import torch.nn as nn
from MoE.moe_layer import MoEBlock, MultiSourceMoEBlock

_NCPS_IMPORT_ERROR: str | None

try:
    from ncps.torch import CfC
except ImportError as _e:  # pragma: no cover - environment-dependent
    CfC = None
    _NCPS_IMPORT_ERROR = (
        "cfc_controller.py requires the `ncps` package (`pip install ncps`). "
        f"Original import error: {_e}"
    )
else:
    _NCPS_IMPORT_ERROR = None


def _require_ncps() -> None:
    if CfC is None:
        raise ImportError(_NCPS_IMPORT_ERROR)


def _state_to_fp32(state):
    # Keep the recurrent state fp32 between steps (ncps' cell returns fp16 under autocast).
    if isinstance(state, tuple):
        return tuple(s.float() for s in state)
    return state.float()


class CfCControllerBlock(nn.Module):
    def __init__(
        self,
        d_model,
        units=None,
        mode="default",
        backbone_units=512,
        backbone_layers=1,
        backbone_dropout=0.0,
        activation="lecun_tanh",
        mixed_memory=False,
        residual=True,
        device=None,
        dtype=None,
    ):
        super().__init__()
        _require_ncps()
        units = d_model if units is None else units
        self.d_model, self.units = d_model, units
        self.mixed_memory, self.residual = mixed_memory, residual
        self.norm = nn.LayerNorm(d_model, device=device, dtype=dtype)
        self.cfc = CfC(
            input_size=d_model,
            units=units,
            proj_size=None
            if units == d_model
            else d_model,  # project back to d_model if state is wider/narrower
            return_sequences=True,
            batch_first=True,
            mixed_memory=mixed_memory,
            mode=mode,
            activation=activation,
            backbone_units=backbone_units,
            backbone_layers=backbone_layers,
            backbone_dropout=backbone_dropout,
        )
        if device is not None or dtype is not None:
            self.cfc.to(device=device, dtype=dtype)

    def init_state(self, batch_size, device=None, dtype=None):  # dtype ignored: always fp32
        device = device if device is not None else next(self.parameters()).device
        h = torch.zeros(batch_size, self.units, device=device, dtype=torch.float32)
        return (h, torch.zeros_like(h)) if self.mixed_memory else h

    def step(self, x, state):
        with torch.autocast(device_type=x.device.type, enabled=False):
            out, new_state = self.cfc(
                self.norm(x.float()).unsqueeze(1),
                _state_to_fp32(state) if state is not None else None,
            )
        out = out.squeeze(1).clamp(min=-1e4, max=1e4)
        y = x + out.to(x.dtype) if self.residual else out.to(x.dtype)
        return y, _state_to_fp32(new_state)


class CfCControllerWrapper(nn.Module):
    """Stack of `num_blocks` CfCControllerBlocks with the nn.LSTM-compatible
    call convention DNC._layer_forward expects (see module docstring)."""

    def __init__(
        self,
        in_dim,
        d_model,
        num_blocks=2,
        units=None,
        mode="default",
        backbone_units=512,
        backbone_layers=1,
        backbone_dropout=0.0,
        activation="lecun_tanh",
        mixed_memory=False,
        residual=True,
        moe_enabled: bool = False,
        moe_num_experts: int = 8,
        moe_expert_dim: int | None = None,
        moe_top_k: int = 1,
        moe_capacity_factor: float = 1.5,
        moe_load_balance_alpha: float = 0.01,
        moe_source_dims: list[int] | None = None,
        device=None,
        dtype=None,
    ):
        super().__init__()
        self.d_model, self.num_blocks = d_model, num_blocks

        # moe_source_dims: when given (>=1 entries, each an input SOURCE's
        # OWN raw width -- e.g. [hidden_size, read_vectors_size] for
        # SplitGraphDNC's [backbone_output, prev_read_vector] combiner
        # input), this wrapper is driven via forward_multi_source() instead
        # of forward(); widths need not match each other, which is exactly
        # the "variable-length inputs" property CfC must keep once MoE
        # sits in front of it. sum(moe_source_dims) must still equal
        # in_dim, since in_dim is what every other bookkeeping path in this
        # project (checkpoints, dim-compat checks) reads as one number.
        self.moe_source_dims = list(moe_source_dims) if moe_source_dims else None
        if self.moe_source_dims is not None:
            if sum(self.moe_source_dims) != in_dim:
                raise ValueError(
                    f"CfCControllerWrapper: sum(moe_source_dims)="
                    f"{sum(self.moe_source_dims)} != in_dim={in_dim}"
                )
            if not moe_enabled:
                raise ValueError("CfCControllerWrapper: moe_source_dims requires moe_enabled=True")

        self.in_adapter: nn.Module = (
            nn.Identity()
            if in_dim == d_model
            else nn.Linear(in_dim, d_model, device=device, dtype=dtype)
        )
        self.blocks: nn.ModuleList[CfCControllerBlock] = nn.ModuleList(
            [
                CfCControllerBlock(
                    d_model,
                    units=units,
                    mode=mode,
                    backbone_units=backbone_units,
                    backbone_layers=backbone_layers,
                    backbone_dropout=backbone_dropout,
                    activation=activation,
                    mixed_memory=mixed_memory,
                    residual=residual,
                    device=device,
                    dtype=dtype,
                )
                for _ in range(num_blocks)
            ]
        )

        self.moe_enabled = moe_enabled
        self.moe_blocks: nn.ModuleList | None = None
        self.source_in_adapters: nn.ModuleList | None = None
        if moe_enabled and self.moe_source_dims is not None:
            # Multi-source front-end: each raw source is projected to
            # d_model, routed+combined by ONE source-aware Top-K MoE bank
            # BEFORE entering the recurrent CfC stack (see
            # forward_multi_source). self.moe_blocks holds this single
            # MultiSourceMoEBlock so every existing "extend moe_layers from
            # layer_controller.moe_blocks" call site (MambaDNC,
            # ChainedControllerWrapper) keeps working with zero changes.
            self.source_in_adapters = nn.ModuleList(
                [
                    nn.Identity()
                    if w == d_model
                    else nn.Linear(w, d_model, device=device, dtype=dtype)
                    for w in self.moe_source_dims
                ]
            )
            self.moe_blocks = nn.ModuleList(
                [
                    MultiSourceMoEBlock(
                        d_model,
                        num_sources=len(self.moe_source_dims),
                        num_experts=moe_num_experts,
                        expert_dim=moe_expert_dim,
                        top_k=moe_top_k,
                        capacity_factor=moe_capacity_factor,
                        load_balance_alpha=moe_load_balance_alpha,
                        device=device,
                        dtype=dtype,
                    )
                ]
            )
        elif moe_enabled:
            # Plain per-block external interleave, same convention as
            # MambaControllerWrapper / Mamba2ControllerWrapper.
            self.moe_blocks = nn.ModuleList(
                [
                    MoEBlock(
                        d_model,
                        num_experts=moe_num_experts,
                        expert_dim=moe_expert_dim,
                        top_k=moe_top_k,
                        capacity_factor=moe_capacity_factor,
                        load_balance_alpha=moe_load_balance_alpha,
                        device=device,
                        dtype=dtype,
                    )
                    for _ in range(num_blocks)
                ]
            )

    def init_state(self, batch_size, device=None, dtype=None):
        if device is None:
            device = next(self.parameters()).device
        return [blk.init_state(batch_size, device=device) for blk in self.blocks]

    def forward(self, input, hx):
        assert input.dim() == 3 and input.size(1) == 1, (
            "CfCControllerWrapper only supports single-timestep calls "
            f"(got shape {tuple(input.shape)})."
        )
        if self.moe_source_dims is not None:
            raise RuntimeError(
                "CfCControllerWrapper was configured with moe_source_dims -- "
                "call forward_multi_source(sources, hx) instead of forward()."
            )
        x = self.in_adapter(input.squeeze(1))
        if hx is None:
            hx = self.init_state(x.size(0), device=x.device)
        new_hx = []
        for i, (block, state) in enumerate(zip(self.blocks, hx)):
            x, new_state = block.step(x, state)
            if self.moe_enabled:
                assert self.moe_blocks is not None
                x = self.moe_blocks[i](x)
            new_hx.append(new_state)
        return x.unsqueeze(1), new_hx

    def forward_multi_source(self, sources: list[torch.Tensor], hx):
        """Multi-source entry point (requires moe_source_dims). `sources[i]`
        is a (B, moe_source_dims[i]) tensor -- e.g. SplitGraphDNC's
        controller combiner passes [h_t, read_vec] directly instead of
        pre-concatenating them. Each source is projected to d_model by its
        own adapter, then routed+combined by the shared, source-aware
        Top-K MoE bank BEFORE entering the recurrent CfC stack -- this is
        what preserves CfC's "many inputs, specialized handling per input"
        property once MoE sits in front of it, while the recurrent stack
        itself still only ever sees one fused (B, d_model) vector per step
        (CfC block internals are completely unchanged)."""
        if self.moe_source_dims is None:
            raise RuntimeError("forward_multi_source requires moe_source_dims to have been set")
        assert self.source_in_adapters is not None and self.moe_blocks is not None
        projected = [adapter(s) for adapter, s in zip(self.source_in_adapters, sources)]
        fused_per_source = self.moe_blocks[0](projected)  # list[Tensor], one per source
        x = torch.stack(fused_per_source, dim=0).sum(dim=0)  # combine into one fused input
        if hx is None:
            hx = self.init_state(x.size(0), device=x.device)
        new_hx = []
        for block, state in zip(self.blocks, hx):
            x, new_state = block.step(x, state)
            new_hx.append(new_state)
        return x.unsqueeze(1), new_hx


if (
    __name__ == "__main__"
):  # smoke test: python -m LNN_controller.cfc_controller (from project root)
    from LNN_controller.chained_controller import ChainedControllerWrapper

    B, T, in_dim, d = 4, 6, 40, 32
    for mm in (False, True):
        w = CfCControllerWrapper(in_dim, d, num_blocks=2, backbone_units=64, mixed_memory=mm)
        hx, loss = w.init_state(B), torch.zeros(())
        for _ in range(T):
            out, hx = w(torch.randn(B, 1, in_dim), hx)
            assert out.shape == (B, 1, d), out.shape
            loss = loss + out.pow(2).mean()
        loss.backward()  # BPTT across chained steps must not raise
        print(f"standalone mixed_memory={mm}: OK (loss {float(loss):.4f})")
    ch = ChainedControllerWrapper(
        [
            CfCControllerWrapper(in_dim, d, 1, backbone_units=64),
            CfCControllerWrapper(d, d, 1, backbone_units=64),
        ]
    )
    hx, loss = ch.init_state(B), torch.zeros(())
    for _ in range(T):
        out, hx = ch(torch.randn(B, 1, in_dim), hx)
        loss = loss + out.pow(2).mean()
    loss.backward()
    print("chained: OK")
