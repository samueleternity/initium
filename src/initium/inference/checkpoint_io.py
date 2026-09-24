"""
file: inference/checkpoint_io.py

Load a checkpoint written by core_training.save_checkpoint() (periodic or
final - same format) and describe it. Loads to CPU; weights_only=False for the
same reason as core_training.load_checkpoint_for_resume (self-produced file
that also stores python/numpy RNG state).
"""

from __future__ import annotations

import os

import torch

REQUIRED_KEYS = ("rnn_state_dict", "output_proj_state_dict", "model_config")


def load_checkpoint(path: str, model_config_override: dict | None = None) -> dict:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"checkpoint not found: {path}")
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(ckpt, dict):
        raise ValueError(f"{path}: not a core_training checkpoint (expected a dict)")
    missing = [k for k in REQUIRED_KEYS if k not in ckpt]
    if missing:
        raise ValueError(f"{path}: not a usable checkpoint, missing keys {missing}")
    cfg = dict(ckpt["model_config"])
    if model_config_override:
        cfg.update(model_config_override)
        print(f"[checkpoint_io] model_config overridden: {model_config_override}")
    ckpt["model_config"] = cfg
    return ckpt


def describe_checkpoint(ckpt: dict) -> str:
    c = ckpt["model_config"]
    lines = [
        f"run_id={ckpt.get('run_id')} step={ckpt.get('step')} beta_target={ckpt.get('beta_target')}",
        f"controller={c.get('controller_type', 'lstm')} hidden={c.get('hidden_size')} "
        f"nr_cells={c.get('nr_cells')} cell_size={c.get('cell_size')} read_heads={c.get('read_heads')}",
        f"link_matrix={c.get('link_matrix_mode', 'dense')} moe={c.get('moe_enabled', False)} "
        f"split_graph={c.get('split_graph_enabled', False)}"
        + (f"({c.get('split_graph_variant')})" if c.get("split_graph_enabled") else "")
        + f" dynamic_n={c.get('dynamic_n_mode', False)}",
    ]
    return "\n".join(lines)
