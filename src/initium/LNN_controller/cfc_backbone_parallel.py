"""
file: LNN_controller/cfc_backbone_parallel.py -- v1

CfC as a whole-sequence backbone for SplitGraphDNC, plus a factory that
builds ANY parallel backbone (or a stack of them) from a variant string.

CfCBackboneParallel: (B, L, in_dim) -> (B, L, d_model), same contract as
mamba_backbone_parallel.MambaBackboneParallel, and like it has NO notion of
DNC/Memory/read vectors (it only ever sees the raw input stream X).
NOTE: ncps' CfC loops over time internally (no parallel scan) and its backbone
MLP consumes the previous hidden state, so this is sequential inside. The
split-graph benefit is structural (backbone can never see reads), not a
Mamba-style scan speedup.
Block = pre-norm residual (Add -> LN -> CfC). Runs in real fp32 by default
(force_fp32=True), same policy as mamba_backbone_parallel._ParallelBlock.

build_parallel_backbone(variant=...): variant is "mamba1" | "mamba2" |
"mamba3" | "cfc", or a "+"-joined stack such as "mamba2+cfc" (stage 0 maps
in_dim -> d_model, later stages d_model -> d_model). num_blocks applies PER
STAGE. This is the single dispatch point for new parallel backbones.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from mamba_controller.mamba_backbone_parallel import MambaBackboneParallel
from src.initium.LNN_controller.cfc_controller import CfC, _require_ncps

_BACKBONE_KINDS = ("mamba1", "mamba2", "mamba3", "cfc")


class _CfCParallelBlock(nn.Module):
    def __init__(
        self,
        d_model,
        units,
        mode,
        backbone_units,
        backbone_layers,
        backbone_dropout,
        activation,
        mixed_memory,
        residual,
        force_fp32,
        device=None,
        dtype=None,
    ):
        super().__init__()
        _require_ncps()
        units = d_model if units is None else units
        self.residual, self.force_fp32 = residual, force_fp32
        self.norm = nn.LayerNorm(d_model, device=device, dtype=dtype)
        self.cfc = CfC(
            input_size=d_model,
            units=units,
            proj_size=None if units == d_model else d_model,
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

    def forward(self, x):  # x: (B, L, D)
        if self.force_fp32:
            with torch.autocast(device_type=x.device.type, enabled=False):
                out, _ = self.cfc(self.norm(x.float()))
        else:
            out, _ = self.cfc(self.norm(x))
        out = out.to(x.dtype)
        return x + out if self.residual else out


class CfCBackboneParallel(nn.Module):
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
        force_fp32=True,
        device=None,
        dtype=None,
    ):
        super().__init__()
        self.d_model = d_model
        self.in_adapter: nn.Module = (
            nn.Identity()
            if in_dim == d_model
            else nn.Linear(in_dim, d_model, device=device, dtype=dtype)
        )
        self.blocks = nn.ModuleList(
            [
                _CfCParallelBlock(
                    d_model,
                    units,
                    mode,
                    backbone_units,
                    backbone_layers,
                    backbone_dropout,
                    activation,
                    mixed_memory,
                    residual,
                    force_fp32,
                    device=device,
                    dtype=dtype,
                )
                for _ in range(num_blocks)
            ]
        )

    def forward(self, x):  # (B, L, in_dim) -> (B, L, d_model), called ONCE per training step
        h = self.in_adapter(x)
        for block in self.blocks:
            h = block(h)
        return h


def build_parallel_backbone(
    in_dim,
    d_model,
    num_blocks=2,
    variant="mamba1",
    d_state=16,
    d_conv=4,
    expand=2,
    headdim=64,
    cfc_kwargs=None,
    device=None,
    dtype=None,
):
    kinds = [k.strip().lower() for k in variant.split("+")]
    bad = [k for k in kinds if k not in _BACKBONE_KINDS]
    if bad:
        raise ValueError(
            f"build_parallel_backbone: unknown variant part(s) {bad} in {variant!r}, "
            f"expected '+'-joined parts of {_BACKBONE_KINDS}"
        )
    stages, cur = [], in_dim
    for kind in kinds:
        if kind == "cfc":
            stages.append(
                CfCBackboneParallel(
                    cur,
                    d_model,
                    num_blocks=num_blocks,
                    device=device,
                    dtype=dtype,
                    **(cfc_kwargs or {}),
                )
            )
        else:
            stages.append(
                MambaBackboneParallel(
                    in_dim=cur,
                    d_model=d_model,
                    num_blocks=num_blocks,
                    variant=kind,
                    d_state=d_state,
                    d_conv=d_conv,
                    expand=expand,
                    headdim=headdim,
                    device=device,
                    dtype=dtype,
                )
            )
        cur = d_model
    return stages[0] if len(stages) == 1 else nn.Sequential(*stages)


if (
    __name__ == "__main__"
):  # smoke test: python -m LNN_controller.cfc_backbone_parallel (from project root)
    B, L, in_dim, d = 4, 12, 40, 32
    for variant in ("cfc", "cfc+cfc"):
        m = build_parallel_backbone(
            in_dim, d, num_blocks=1, variant=variant, cfc_kwargs=dict(backbone_units=64)
        )
        y = m(torch.randn(B, L, in_dim))
        assert y.shape == (B, L, d), y.shape
        y.pow(2).mean().backward()
        print(f"{variant}: OK")
    print("For mamba1/2/3+cfc stacks, run the pilot below on a GPU (Mamba kernels need CUDA).")
