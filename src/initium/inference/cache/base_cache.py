"""
file: inference/cache/base_cache.py

Interface the cached engine talks to (counterpart of data/base_dataset.py). A new
cache type subclasses BaseCache, overrides only the hooks it needs, and registers
itself in cache_registry.py; the engine never imports anything cache-specific.

Two hook families:
  episode-level : lookup_episode() / store_episode()   (wrap the whole episode)
  prefix-level  : find_resume() / store_boundary()     (accelerate the computation;
                  only used when the caches' `wants_boundaries` is True and the
                  episode declares cache_boundaries)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch

from src.initium.inference.cache.keys import digest, hash_tensor
from src.initium.inference.cache.state_utils import tree_nbytes
from src.initium.inference.cache.stores import CacheStats, TieredStore


@dataclass
class EpisodeCtx:
    index: int
    episode: Any  # inference.tasks.base_task.Episode
    reset_experience: bool
    in_hash: str  # hash of the episode's full input sequence
    chain: str  # "root" in fresh mode; history hash in persistent mode
    boundaries: list[int] = field(default_factory=list)  # validated cut points
    _prefix_hashes: dict[int, str] = field(default_factory=dict, repr=False)

    def prefix_hash(self, boundary: int) -> str:
        h = self._prefix_hashes.get(boundary)
        if h is None:
            h = self._prefix_hashes[boundary] = hash_tensor(self.episode.input_seq[:boundary])
        return h


@dataclass
class EpisodeHit:
    output: torch.Tensor  # (T, output_dim) CPU float
    hidden: Any  # restored state (persistent mode) or None
    saved_ms: float


@dataclass
class ResumePoint:
    position: int  # steps [0, position) are already reflected in `hidden`
    hidden: Any
    saved_ms: float
    cache_name: str


class BaseCache:
    name: str = "base"
    wants_boundaries: bool = False

    def __init__(self, store: TieredStore, model_fp: str, device: torch.device):
        self.store, self.model_fp, self.device = store, model_fp, device
        self.stats = CacheStats()

    # ---- hooks ----------------------------------------------------------------
    def applicable(self, *, reset_experience: bool, resumable: bool) -> tuple[bool, str]:
        return True, ""

    def lookup_episode(self, ctx: EpisodeCtx) -> EpisodeHit | None:
        return None

    def store_episode(
        self, ctx: EpisodeCtx, output: torch.Tensor, hidden_after: Any, compute_ms: float
    ) -> None:
        pass

    def find_resume(self, ctx: EpisodeCtx) -> ResumePoint | None:
        return None

    def store_boundary(
        self, ctx: EpisodeCtx, boundary: int, hidden: Any, compute_ms: float
    ) -> None:
        pass

    # ---- helpers --------------------------------------------------------------
    def report(self, reset: bool = False) -> dict:
        d = self.stats.to_dict()
        if reset:
            self.stats = CacheStats()
        return d

    def _key(self, *parts: Any) -> str:
        return f"{self.name}-{digest(self.model_fp, self.name, *parts)}"

    def _get(self, key: str):
        return self.store.get(key)

    def _put(self, key: str, payload: Any, compute_ms: float) -> None:
        nbytes = tree_nbytes(payload)
        self.store.put(
            key,
            {
                "payload": payload,
                "compute_ms": float(compute_ms),
                "nbytes": nbytes,
                "kind": self.name,
            },
        )
        self.stats.stores += 1
        self.stats.bytes_stored += nbytes
