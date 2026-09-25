"""
file: inference/cache/state_utils.py

Helpers for the nested model state (chx, mhx, last_read): controller states
(tuples/lists), the Memory dict, tensors, None. Snapshots are stored on CPU and
ALWAYS deep-copied on both sides: Memory.reset(..., hidden) rebinds entries of the
dict it is given and the write path rebinds them again, so a cached dict passed
straight into forward() would silently change under us.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any

import torch


def tree_map(fn: Callable[[torch.Tensor], Any], obj: Any) -> Any:
    if isinstance(obj, torch.Tensor):
        return fn(obj)
    if isinstance(obj, dict):
        return {k: tree_map(fn, v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [tree_map(fn, v) for v in obj]
    if isinstance(obj, tuple):
        items = [tree_map(fn, v) for v in obj]
        return type(obj)(*items) if hasattr(obj, "_fields") else tuple(items)
    return obj  # None, numbers, strings


def snapshot(tree: Any) -> Any:  # -> independent CPU copy
    return tree_map(lambda t: t.detach().to("cpu", copy=True), tree)


def restore(tree: Any, device: torch.device) -> Any:  # -> independent copy on `device`
    return tree_map(lambda t: t.detach().to(device, copy=True), tree)


def clone_tree(tree: Any) -> Any:
    return tree_map(lambda t: t.detach().clone(), tree)


def tree_nbytes(obj: Any) -> int:
    if isinstance(obj, torch.Tensor):
        return obj.numel() * obj.element_size()
    if isinstance(obj, dict):
        return sum(tree_nbytes(v) for v in obj.values())
    if isinstance(obj, list | tuple):
        return sum(tree_nbytes(v) for v in obj)
    return 0


def tree_max_abs_diff(a: Any, b: Any) -> float:
    """Max |a-b| over all tensors; inf on any structural mismatch; NaN propagates
    (callers must test `diff <= tol`, never `diff > tol`)."""
    if isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor):
        if a.shape != b.shape:
            return math.inf
        if a.numel() == 0:
            return 0.0
        return (a.detach().float() - b.detach().float().to(a.device)).abs().max().item()
    if isinstance(a, dict) and isinstance(b, dict):
        if a.keys() != b.keys():
            return math.inf
        return max([0.0] + [tree_max_abs_diff(a[k], b[k]) for k in a])
    if isinstance(a, list | tuple) and isinstance(b, list | tuple):
        if len(a) != len(b):
            return math.inf
        return max([0.0] + [tree_max_abs_diff(x, y) for x, y in zip(a, b)])
    if a is None and b is None:
        return 0.0
    return 0.0 if a == b else math.inf
