"""
file: inference/model_loader.py

Rebuild ANY controller variant the training script can produce (lstm, mamba,
mamba2, mamba3, cfc, hybrids "<kind>+<kind>", MoE, link-matrix modes,
split-graph incl. controller combiners) from checkpoint["model_config"], then
load weights. Mirrors the construction logic in core_training.run().

Order matters and matches training: build -> patch link matrix -> create
output_proj -> install stochastic write heads -> load_state_dict (the state
dict contains the heads' mu/logvar params and prior buffers). nr_cells is
taken from the checkpoint (live size at save time), so no resize is needed.

Keys missing from older checkpoints fall back to config/controller_config.py
with a printed warning (a wrong fallback surfaces as a load_state_dict shape
error; fix with --model-config-override).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn

import config.controller_config as cc
from mamba_controller.mamba_controller import MambaDNC
from mamba_controller.split_graph_dnc import SplitGraphDNC
from memory_manipulation.link_matrix_ablation import patch_link_matrix
from memory_manipulation.stochastic_write_head_v2 import (
    install_stochastic_write_heads,
    load_prior_state,
)


@dataclass
class LoadedModel:
    rnn: nn.Module
    output_proj: nn.Module
    heads: list
    config: dict
    step: int
    run_id: str
    beta_target: float
    sampled_writes: bool


class _Cfg:
    """dict wrapper that remembers which keys fell back to defaults."""

    def __init__(self, cfg: dict):
        self.cfg, self.defaulted = cfg, []

    def get(self, key: str, default: Any):
        v = self.cfg.get(key)
        if v is None:
            self.defaulted.append(key)
            return default
        return v


def _moe_kwargs(c: _Cfg) -> dict:
    return dict(
        moe_enabled=bool(c.cfg.get("moe_enabled", False)),
        moe_num_experts=c.get("moe_num_experts", cc.MOE_NUM_EXPERTS),
        moe_expert_dim=c.cfg.get("moe_expert_dim", cc.MOE_EXPERT_DIM),  # None is legitimate
        moe_capacity_factor=c.get("moe_capacity_factor", cc.MOE_CAPACITY_FACTOR),
        moe_load_balance_alpha=c.get("moe_load_balance_alpha", cc.MOE_LOAD_BALANCE_ALPHA),
    )


def _mamba1(c):
    return dict(
        mamba_d_state=c.get("mamba_d_state", cc.MAMBA_D_STATE),
        mamba_d_conv=c.get("mamba_d_conv", cc.MAMBA_D_CONV),
        mamba_expand=c.get("mamba_expand", cc.MAMBA_EXPAND),
    )


def _mamba2(c):
    return dict(
        mamba2_d_state=c.get("mamba2_d_state", cc.MAMBA2_D_STATE),
        mamba2_d_conv=c.get("mamba2_d_conv", cc.MAMBA2_D_CONV),
        mamba2_expand=c.get("mamba2_expand", cc.MAMBA2_EXPAND),
        mamba2_headdim=c.get("mamba2_headdim", cc.MAMBA2_HEADDIM),
        mamba2_ngroups=c.get("mamba2_ngroups", cc.MAMBA2_NGROUPS),
    )


def _mamba3(c):
    return dict(
        mamba3_d_state=c.get("mamba3_d_state", cc.MAMBA3_D_STATE),
        mamba3_expand=c.get("mamba3_expand", cc.MAMBA3_EXPAND),
        mamba3_headdim=c.get("mamba3_headdim", cc.MAMBA3_HEADDIM),
        mamba3_rope_fraction=c.get("mamba3_rope_fraction", cc.MAMBA3_ROPE_FRACTION),
    )


def _cfc(c):
    return dict(
        cfc_mode=c.get("cfc_mode", cc.CFC_MODE),
        cfc_backbone_units=c.get("cfc_backbone_units", cc.CFC_BACKBONE_UNITS),
        cfc_backbone_layers=c.get("cfc_backbone_layers", cc.CFC_BACKBONE_LAYERS),
        cfc_backbone_dropout=c.get("cfc_backbone_dropout", cc.CFC_BACKBONE_DROPOUT),
        cfc_activation=c.get("cfc_activation", cc.CFC_ACTIVATION),
        cfc_mixed_memory=c.get("cfc_mixed_memory", cc.CFC_MIXED_MEMORY),
        cfc_residual=c.get("cfc_residual", cc.CFC_RESIDUAL),
    )


def _controller_kwargs(controller: str, c: _Cfg) -> dict:
    moe = _moe_kwargs(c)
    if controller == "mamba":
        return {**_mamba1(c), **moe}
    if controller == "mamba2":
        return {**_mamba2(c), **moe}
    if controller == "mamba3":
        return {**_mamba3(c), **moe}
    if controller == "cfc":
        return {**_cfc(c), "moe_enabled": moe["moe_enabled"]}
    if "+" in controller:
        return {
            **_mamba1(c),
            **_mamba2(c),
            **_mamba3(c),
            **_cfc(c),
            "hybrid_cfc_num_blocks": c.get("hybrid_cfc_num_blocks", cc.HYBRID_CFC_NUM_BLOCKS),
            "moe_enabled": moe["moe_enabled"],
        }
    return {}  # lstm: identical to training (no extra kwargs)


def _build_split_graph(c: _Cfg, input_dim, hidden, nr_cells, cell_size, read_heads, device):
    variant = c.get("split_graph_variant", cc.SPLIT_GRAPH_MAMBA_VARIANT)
    default_hp = (
        dict(d_state=cc.MAMBA3_D_STATE, d_conv=cc.MAMBA_D_CONV, expand=cc.MAMBA3_EXPAND)
        if variant.startswith("mamba3")
        else dict(d_state=cc.MAMBA2_D_STATE, d_conv=cc.MAMBA2_D_CONV, expand=cc.MAMBA2_EXPAND)
        if variant.startswith("mamba2")
        else dict(d_state=cc.MAMBA_D_STATE, d_conv=cc.MAMBA_D_CONV, expand=cc.MAMBA_EXPAND)
    )
    hp = c.get("split_graph_mamba_hparams", default_hp)
    cfc_kwargs = c.get(
        "split_graph_cfc_kwargs",
        dict(
            mode=cc.CFC_MODE,
            backbone_units=cc.CFC_BACKBONE_UNITS,
            backbone_layers=cc.CFC_BACKBONE_LAYERS,
            backbone_dropout=cc.CFC_BACKBONE_DROPOUT,
            activation=cc.CFC_ACTIVATION,
            mixed_memory=cc.CFC_MIXED_MEMORY,
            residual=cc.CFC_RESIDUAL,
        ),
    )
    return SplitGraphDNC(
        input_size=input_dim,
        hidden_size=hidden,
        nr_cells=nr_cells,
        cell_size=cell_size,
        read_heads=read_heads,
        num_backbone_blocks=c.get("split_graph_num_blocks", cc.SPLIT_GRAPH_NUM_BLOCKS),
        mamba_variant=variant,
        mamba_d_state=hp["d_state"],
        mamba_d_conv=hp["d_conv"],
        mamba_expand=hp["expand"],
        mamba_headdim=c.get("split_graph_headdim", cc.SPLIT_GRAPH_MAMBA_HEADDIM),
        cfc_kwargs=cfc_kwargs,
        combine_reads=c.get("split_graph_combine_reads", cc.SPLIT_GRAPH_COMBINE_READS),
        combiner_mode=c.get("split_graph_combiner_mode", cc.SPLIT_GRAPH_COMBINER_MODE),
        combiner_variant=c.get("split_graph_combiner_variant", cc.SPLIT_GRAPH_COMBINER_VARIANT),
        combiner_num_blocks=c.get(
            "split_graph_combiner_num_blocks", cc.SPLIT_GRAPH_COMBINER_NUM_BLOCKS
        ),
        independent_linears=True,
        device=device,
    ).to(device)


def load_model(ckpt: dict, device: torch.device, deterministic_write: bool = False) -> LoadedModel:
    raw = ckpt["model_config"]
    c = _Cfg(raw)
    controller = raw.get("controller_type", "lstm")
    input_dim = int(raw.get("input_dim", raw.get("input_size")))
    hidden, nr_cells = int(raw["hidden_size"]), int(raw["nr_cells"])
    cell_size, read_heads = int(raw["cell_size"]), int(raw["read_heads"])

    if raw.get("split_graph_enabled", False):
        rnn = _build_split_graph(c, input_dim, hidden, nr_cells, cell_size, read_heads, device)
    else:
        extra = {}
        if raw.get("num_hidden_layers") is not None:
            extra["num_hidden_layers"] = int(raw["num_hidden_layers"])
        rnn = MambaDNC(
            input_size=input_dim,
            hidden_size=hidden,
            rnn_type=controller,
            num_layers=1,
            nr_cells=nr_cells,
            cell_size=cell_size,
            read_heads=read_heads,
            batch_first=True,
            device=device,
            independent_linears=True,
            **_controller_kwargs(controller, c),
            **extra,
        ).to(device)

    link_mode = raw.get("link_matrix_mode", "dense")
    if link_mode != "dense":
        patch_link_matrix(rnn, mode=link_mode, topk=raw.get("link_matrix_topk"))

    w = ckpt["output_proj_state_dict"]["weight"]
    output_proj = nn.Linear(w.shape[1], w.shape[0]).to(device)

    beta = float(ckpt.get("beta_target", 0.0) or 0.0)
    heads = install_stochastic_write_heads(rnn, device=device, sample=(beta > 0.0))

    if c.defaulted:
        print(
            f"[model_loader] WARNING: checkpoint config lacks {sorted(set(c.defaulted))}; "
            f"used config/controller_config.py defaults. If loading fails, pass "
            f"--model-config-override with the right values."
        )
    try:
        rnn.load_state_dict(ckpt["rnn_state_dict"])
    except RuntimeError as e:
        raise RuntimeError(
            "rnn.load_state_dict failed -- the rebuilt architecture does not match the "
            "checkpoint (typically a legacy checkpoint whose config was defaulted). "
            f"Original error:\n{e}"
        ) from e
    output_proj.load_state_dict(ckpt["output_proj_state_dict"])
    if "prior_state" in ckpt:
        load_prior_state(heads, ckpt["prior_state"])

    if deterministic_write:  # use mu only (no write noise) even if trained with beta > 0
        for h in heads:
            h.sample = False

    rnn.eval()
    output_proj.eval()
    return LoadedModel(
        rnn=rnn,
        output_proj=output_proj,
        heads=heads,
        config=raw,
        step=int(ckpt.get("step", -1)),
        run_id=str(ckpt.get("run_id", "?")),
        beta_target=beta,
        sampled_writes=bool(heads and heads[0].sample),
    )
