"""
file: inference/cache/keys.py

Hashing for cache keys. blake2b (fast, stdlib). A cache key always contains the
MODEL FINGERPRINT (hash of every weight + config + the flags that change the
forward pass), so entries can never leak between different checkpoints.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

import torch


def new_hasher():
    return hashlib.blake2b(digest_size=16)


def update_with_tensor(h, t: torch.Tensor) -> None:
    t = t.detach().cpu().contiguous()
    h.update(str(t.dtype).encode())
    h.update(str(tuple(t.shape)).encode())
    if t.numel():
        h.update(t.reshape(-1).view(torch.uint8).numpy())


def hash_tensor(t: torch.Tensor) -> str:
    h = new_hasher()
    update_with_tensor(h, t)
    return h.hexdigest()


def digest(*parts: Any) -> str:
    h = new_hasher()
    for p in parts:
        h.update(str(p).encode())
        h.update(b"\x1f")
    return h.hexdigest()


def chain_next(chain: str, in_hash: str) -> str:
    """Persistent mode: the state after episode k is a function of (initial state,
    inputs 0..k). Hashing the inputs into a chain therefore identifies the state
    exactly, without hashing the (large) memory tensors every episode."""
    return digest(chain, in_hash)


def _update_obj(h, obj: Any) -> None:
    if isinstance(obj, torch.Tensor):
        update_with_tensor(h, obj)
    elif isinstance(obj, dict):
        for k in sorted(obj, key=str):
            h.update(str(k).encode())
            _update_obj(h, obj[k])
    elif isinstance(obj, list | tuple):
        for v in obj:
            _update_obj(h, v)
    else:
        h.update(repr(obj).encode())


def model_fingerprint(ckpt: dict, extras: dict) -> str:
    """Hash of weights (rnn + output_proj + prior buffers) + model_config + `extras`
    (flags that alter the forward pass: deterministic_write, ablate_memory, ...).
    Takes a couple of seconds for a large model; only computed when caching is on.
    Deliberately NOT included: device / torch version (tiny numeric differences are
    what --cache-verify is for)."""
    h = new_hasher()
    _update_obj(h, ckpt["rnn_state_dict"])
    _update_obj(h, ckpt["output_proj_state_dict"])
    _update_obj(h, ckpt.get("prior_state"))
    h.update(json.dumps(ckpt["model_config"], sort_keys=True, default=str).encode())
    h.update(json.dumps(extras, sort_keys=True, default=str).encode())
    return h.hexdigest()
