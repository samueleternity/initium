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
    def __init__(self, d_model, units=None, mode="default", backbone_units=512,
                 backbone_layers=1, backbone_dropout=0.0, activation="lecun_tanh",
                 mixed_memory=False, residual=True, device=None, dtype=None):
        super().__init__()
        _require_ncps()
        units = d_model if units is None else units
        self.d_model, self.units = d_model, units
        self.mixed_memory, self.residual = mixed_memory, residual
        self.norm = nn.LayerNorm(d_model, device=device, dtype=dtype)
        self.cfc = CfC(
            input_size=d_model, units=units,
            proj_size=None if units == d_model else d_model,  # project back to d_model if state is wider/narrower
            return_sequences=True, batch_first=True, mixed_memory=mixed_memory,
            mode=mode, activation=activation, backbone_units=backbone_units,
            backbone_layers=backbone_layers, backbone_dropout=backbone_dropout,
        )
        if device is not None or dtype is not None:
            self.cfc.to(device=device, dtype=dtype)

    def init_state(self, batch_size, device=None, dtype=None):  # dtype ignored: always fp32
        device = device if device is not None else next(self.parameters()).device
        h = torch.zeros(batch_size, self.units, device=device, dtype=torch.float32)
        return (h, torch.zeros_like(h)) if self.mixed_memory else h

    def step(self, x, state):
        with torch.autocast(device_type=x.device.type, enabled=False):
            out, new_state = self.cfc(self.norm(x.float()).unsqueeze(1), 
                                        _state_to_fp32(state) if state is not None else None)
        out = out.squeeze(1).clamp(min=-1e4, max=1e4)
        y = x + out.to(x.dtype) if self.residual else out.to(x.dtype)
        return y, _state_to_fp32(new_state)


class CfCControllerWrapper(nn.Module):
    """Stack of `num_blocks` CfCControllerBlocks with the nn.LSTM-compatible
    call convention DNC._layer_forward expects (see module docstring)."""

    def __init__(self, in_dim, d_model, num_blocks=2, units=None, mode="default",
                 backbone_units=512, backbone_layers=1, backbone_dropout=0.0,
                 activation="lecun_tanh", mixed_memory=False, residual=True,
                 device=None, dtype=None):
        super().__init__()
        self.d_model, self.num_blocks = d_model, num_blocks
        self.in_adapter: nn.Module = (
            nn.Identity() if in_dim == d_model else nn.Linear(in_dim, d_model, device=device, dtype=dtype)
        )
        self.blocks = nn.ModuleList([
            CfCControllerBlock(d_model, units=units, mode=mode, backbone_units=backbone_units,
                               backbone_layers=backbone_layers, backbone_dropout=backbone_dropout,
                               activation=activation, mixed_memory=mixed_memory, residual=residual,
                               device=device, dtype=dtype)
            for _ in range(num_blocks)
        ])
        # MoE deliberately not wired (Concept 16/SP-10 isolation); attributes exist so
        # MambaDNC / ChainedControllerWrapper can read them uniformly.
        self.moe_enabled = False
        self.moe_blocks = None

    def init_state(self, batch_size, device=None, dtype=None):
        if device is None:
            device = next(self.parameters()).device
        return [blk.init_state(batch_size, device=device) for blk in self.blocks]

    def forward(self, input, hx):
        assert input.dim() == 3 and input.size(1) == 1, (
            "CfCControllerWrapper only supports single-timestep calls "
            f"(got shape {tuple(input.shape)})."
        )
        x = self.in_adapter(input.squeeze(1))
        if hx is None:
            hx = self.init_state(x.size(0), device=x.device)
        new_hx = []
        for block, state in zip(self.blocks, hx):
            x, new_state = block.step(x, state)
            new_hx.append(new_state)
        return x.unsqueeze(1), new_hx


if __name__ == "__main__":  # smoke test: python -m LNN_controller.cfc_controller (from project root)
    from src.initium.LNN_controller.chained_controller import ChainedControllerWrapper
    B, T, in_dim, d = 4, 6, 40, 32
    for mm in (False, True):
        w = CfCControllerWrapper(in_dim, d, num_blocks=2, backbone_units=64, mixed_memory=mm)
        hx, loss = w.init_state(B), 0.0
        for _ in range(T):
            out, hx = w(torch.randn(B, 1, in_dim), hx)
            assert out.shape == (B, 1, d), out.shape
            loss = loss + out.pow(2).mean()
        loss.backward()  # BPTT across chained steps must not raise
        print(f"standalone mixed_memory={mm}: OK (loss {float(loss):.4f})")
    ch = ChainedControllerWrapper([CfCControllerWrapper(in_dim, d, 1, backbone_units=64),
                                   CfCControllerWrapper(d, d, 1, backbone_units=64)])
    hx, loss = ch.init_state(B), 0.0
    for _ in range(T):
        out, hx = ch(torch.randn(B, 1, in_dim), hx)
        loss = loss + out.pow(2).mean()
    loss.backward()
    print("chained: OK")