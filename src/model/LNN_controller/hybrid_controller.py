"""
file: LNN_controller/hybrid_controller.py -- v1

Factory for interleaved, per-timestep controller CHAINS built from any mix of
controller wrappers that follow the controller-wrapper protocol (see
cfc_controller.py's docstring). An rnn_type of the form "<kind>+<kind>[+...]",
kind in {"mamba", "mamba2", "mamba3", "cfc"}, e.g. "mamba+cfc", builds a
ChainedControllerWrapper: stage 0 maps in_dim -> d_model, every later stage
maps d_model -> d_model. Each kind uses its own block count (blocks_per_kind)
and its own hyperparameters (kwargs_per_kind).

Adding a new controller kind = one entry in STAGE_KINDS + one branch in
_make_stage(). Mamba wrappers are imported lazily inside _make_stage() to
avoid a circular import (mamba_controller.py imports this file).

New controller kind: add it to STAGE_KINDS and _make_stage() in hybrid_controller.py, plus one kwargs_per_kind entry in H6.
New parallel backbone kind: add it to _BACKBONE_KINDS and the loop in build_parallel_backbone().

"""

from __future__ import annotations

from LNN_controller.cfc_controller import CfCControllerWrapper
from LNN_controller.chained_controller import ChainedControllerWrapper

STAGE_KINDS = ("mamba", "mamba2", "mamba3", "cfc")


def parse_hybrid_spec(spec: str) -> list[str]:
    return [k.strip().lower() for k in spec.split("+")]


def is_hybrid_rnn_type(rnn_type: str) -> bool:
    kinds = parse_hybrid_spec(rnn_type)
    return len(kinds) >= 2 and all(k in STAGE_KINDS for k in kinds)


def _make_stage(kind, in_dim, d_model, num_blocks, kw, device):
    if kind == "cfc":
        return CfCControllerWrapper(
            in_dim=in_dim, d_model=d_model, num_blocks=num_blocks, device=device, **kw
        )
    if kind == "mamba":
        from mamba_controller.mamba_controller import MambaControllerWrapper as W
    elif kind == "mamba2":
        from mamba_controller.mamba2_controller import Mamba2ControllerWrapper as W
    elif kind == "mamba3":
        from mamba_controller.mamba3_controller import Mamba3ControllerWrapper as W
    else:
        raise ValueError(
            f"hybrid_controller: unknown stage kind {kind!r}, expected one of {STAGE_KINDS}"
        )
    return W(in_dim=in_dim, d_model=d_model, num_blocks=num_blocks, device=device, **kw)


def build_hybrid_controller(spec, in_dim, d_model, blocks_per_kind, kwargs_per_kind, device=None):
    kinds = parse_hybrid_spec(spec)
    assert len(kinds) >= 2 and all(k in STAGE_KINDS for k in kinds), f"bad hybrid spec {spec!r}"
    stages, cur = [], in_dim
    for kind in kinds:
        stages.append(
            _make_stage(
                kind, cur, d_model, blocks_per_kind[kind], kwargs_per_kind.get(kind, {}), device
            )
        )
        cur = d_model
    return ChainedControllerWrapper(stages, stage_kinds=kinds)


if (
    __name__ == "__main__"
):  # smoke test: python -m LNN_controller.hybrid_controller (from project root)
    import torch

    B, T, in_dim, d = 4, 6, 40, 64
    blocks = {"mamba": 1, "mamba2": 1, "mamba3": 1, "cfc": 1}
    kws = {"cfc": dict(backbone_units=64)}
    for spec in ("cfc+cfc", "mamba+cfc", "mamba2+cfc", "mamba3+cfc"):
        try:
            w = build_hybrid_controller(spec, in_dim, d, blocks, kws)
        except ImportError as e:
            print(f"{spec}: SKIPPED ({e})")
            continue
        hx, loss = w.init_state(B), 0.0
        for _ in range(T):
            out, hx = w(torch.randn(B, 1, in_dim), hx)
            assert out.shape == (B, 1, d), out.shape
            loss = loss + out.pow(2).mean()
        loss.backward()
        print(f"{spec}: OK (loss {float(loss):.4f})")
