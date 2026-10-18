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

from collections.abc import Iterable
from typing import cast

import torch
import torch.nn as nn
import torch.nn.functional as F

from initium.MoE.moe_layer import MoEBlock, MultiSourceMoEBlock

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

    def forward(self, x, state):
        """Run one recurrent step through ``Module.__call__``.

        Keeping the step behind ``forward`` preserves the equations while
        ensuring module hooks (including segment activation diagnostics) see
        the CfC block boundary.
        """
        return self.step(x, state)


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
        # Disabled during normal training; the trainer enables edge tracing
        # only on a selected anomaly-detection step.
        self._gradient_trace_enabled = False
        self._gradient_trace_step: int | None = None
        self._gradient_trace_call = 0
        self._gradient_trace_reports = 0
        self._gradient_trace_forward_reported: set[str] = set()

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
        self.blocks: nn.ModuleList = nn.ModuleList(
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
        blocks = cast(Iterable[CfCControllerBlock], self.blocks)
        return [blk.init_state(batch_size, device=device) for blk in blocks]

    def forward(self, input, hx):
        if isinstance(input, list):
            if self.moe_source_dims is None:
                raise RuntimeError(
                    "list input is supported only when moe_source_dims is configured"
                )
            return self.forward_multi_source(input, hx)
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
        blocks = cast(Iterable[CfCControllerBlock], self.blocks)
        for i, (block, state) in enumerate(zip(blocks, hx)):
            x, new_state = block(x, state)
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

        trace_call = self._gradient_trace_call
        self._gradient_trace_call += 1

        def trace_gradient(label: str, tensor: torch.Tensor) -> torch.Tensor:
            if not self._gradient_trace_enabled:
                return tensor
            if tensor.requires_grad and label not in self._gradient_trace_forward_reported:
                self._gradient_trace_forward_reported.add(label)
                print(
                    "[cfc-gradient-trace:forward] "
                    f"step={self._gradient_trace_step} call={trace_call} "
                    f"tensor={label} dtype={tensor.dtype} "
                    f"requires_grad={tensor.requires_grad} shape={tuple(tensor.shape)}"
                )
            if not tensor.requires_grad:
                return tensor

            def report(gradient: torch.Tensor) -> torch.Tensor:
                if self._gradient_trace_reports >= 96:
                    return gradient
                finite = torch.isfinite(gradient)
                bad_count = int((~finite).sum().detach().cpu())
                if not bad_count:
                    return gradient
                nan_count = int(torch.isnan(gradient).sum().detach().cpu())
                inf_count = int(torch.isinf(gradient).sum().detach().cpu())
                finite_values = gradient.detach()[finite].float()
                finite_absmax = (
                    float(finite_values.abs().max().cpu())
                    if finite_values.numel()
                    else float("nan")
                )
                print(
                    "[cfc-gradient-trace] "
                    f"step={self._gradient_trace_step} call={trace_call} "
                    f"tensor={label} bad={bad_count / max(1, gradient.numel()):.3%} "
                    f"nan={nan_count} inf={inf_count} "
                    f"finite_absmax={finite_absmax:.4g} "
                    f"grad_dtype={gradient.dtype} "
                    f"shape={tuple(gradient.shape)}"
                )
                self._gradient_trace_reports += 1
                return gradient

            tensor.register_hook(report)
            return tensor

        def report_parameter_health(label: str, module: nn.Module | None) -> None:
            if not self._gradient_trace_enabled or trace_call != 0 or module is None:
                return
            parameters = list(module.parameters())
            if not parameters:
                print(
                    "[cfc-gradient-trace:params] "
                    f"step={self._gradient_trace_step} module={label} parameters=none"
                )
                return
            bad_count = torch.zeros((), device=parameters[0].device, dtype=torch.long)
            total = 0
            absmax = torch.zeros((), device=parameters[0].device, dtype=torch.float32)
            dtypes = sorted({str(parameter.dtype) for parameter in parameters})
            for parameter in parameters:
                values = parameter.detach()
                bad_count = bad_count + (~torch.isfinite(values)).sum()
                total += values.numel()
                if values.numel():
                    absmax = torch.maximum(absmax, values.float().abs().amax())
            bad, maximum = torch.stack((bad_count.float(), absmax)).cpu().tolist()
            print(
                "[cfc-gradient-trace:params] "
                f"step={self._gradient_trace_step} module={label} "
                f"dtypes={','.join(dtypes)} nonfinite={int(bad)}/{total} "
                f"absmax={maximum:.4g}"
            )

        def trace_state(label: str, state):
            if isinstance(state, tuple):
                return tuple(
                    trace_state(f"{label}.{index}", value) for index, value in enumerate(state)
                )
            if isinstance(state, list):
                return [trace_state(f"{label}.{index}", value) for index, value in enumerate(state)]
            if torch.is_tensor(state):
                return trace_gradient(label, state)
            return state

        if self._gradient_trace_enabled:
            report_parameter_health("in_adapter", self.in_adapter)
            for index, adapter in enumerate(self.source_in_adapters):
                report_parameter_health(f"source_adapter.{index}", adapter)
            assert self.moe_blocks is not None
            multisource_moe = self.moe_blocks[0]
            assert isinstance(multisource_moe, MultiSourceMoEBlock)
            report_parameter_health("multisource.norm", multisource_moe.norm)
            report_parameter_health("multisource.source_embed", multisource_moe.source_embed)
            report_parameter_health("switch_moe.router", multisource_moe.moe.router)
            report_parameter_health("switch_moe.experts", multisource_moe.moe.experts)
            sources = [
                trace_gradient(f"source.{index}", source) for index, source in enumerate(sources)
            ]
            hx = trace_state("state.in", hx)
        # Preserve the stock CfC input-fusion path exactly: the baseline
        # receives one affine projection of concat([backbone, previous
        # read]). The source-aware MoE is an additive residual around that
        # fused representation. This keeps the memory signal and controller
        # initialization intact while still allowing the experts to learn
        # source-specific corrections.
        concatenated = torch.cat(sources, dim=-1)
        concatenated = trace_gradient("fusion.concat", concatenated)
        # Keep the CFC fusion projections in fp32. The CfC block below already
        # disables autocast; letting these Linear layers emit fp16 introduces
        # a low-precision backward boundary before the recurrent controller.
        with torch.autocast(device_type=concatenated.device.type, enabled=False):
            if isinstance(self.in_adapter, nn.Linear):
                x = F.linear(
                    concatenated.float(),
                    self.in_adapter.weight.float(),
                    None if self.in_adapter.bias is None else self.in_adapter.bias.float(),
                )
            else:
                x = self.in_adapter(concatenated.float())
            projected = []
            for adapter, source in zip(self.source_in_adapters, sources):
                source = source.float()
                if isinstance(adapter, nn.Linear):
                    projected.append(
                        F.linear(
                            source,
                            adapter.weight.float(),
                            None if adapter.bias is None else adapter.bias.float(),
                        )
                    )
                else:
                    projected.append(adapter(source))
        x = trace_gradient("fusion.base_projection", x)
        if self._gradient_trace_enabled:
            projected = [
                trace_gradient(f"source_projection.{index}", value)
                for index, value in enumerate(projected)
            ]
        fused_per_source = self.moe_blocks[0](
            projected,
            gradient_trace=(trace_gradient if self._gradient_trace_enabled else None),
        )  # source + routed delta
        if self._gradient_trace_enabled:
            fused_per_source = [
                trace_gradient(f"source_moe.{index}", value)
                for index, value in enumerate(fused_per_source)
            ]
        moe_delta = torch.stack(
            [fused - source for fused, source in zip(fused_per_source, projected)],
            dim=0,
        ).sum(dim=0)
        moe_delta = trace_gradient("fusion.moe_delta", moe_delta)
        x = x + moe_delta
        x = trace_gradient("fusion.combined", x)
        if hx is None:
            hx = self.init_state(x.size(0), device=x.device)
        new_hx = []
        blocks = cast(Iterable[CfCControllerBlock], self.blocks)
        for index, (block, state) in enumerate(zip(blocks, hx)):
            x = trace_gradient(f"cfc.{index}.input", x)
            state = trace_state(f"cfc.{index}.state", state)
            x, new_state = block(x, state)
            x = trace_gradient(f"cfc.{index}.output", x)
            new_state = trace_state(f"cfc.{index}.state_out", new_state)
            new_hx.append(new_state)
        return x.unsqueeze(1), new_hx

    def set_gradient_trace(self, enabled: bool, *, step: int | None = None) -> None:
        """Enable low-volume edge tracing for one targeted training step."""
        self._gradient_trace_enabled = bool(enabled)
        self._gradient_trace_step = step if enabled else None
        self._gradient_trace_call = 0
        self._gradient_trace_reports = 0
        self._gradient_trace_forward_reported.clear()


if (
    __name__ == "__main__"
):  # smoke test: python -m LNN_controller.cfc_controller (from project root)
    from initium.LNN_controller.chained_controller import ChainedControllerWrapper

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
