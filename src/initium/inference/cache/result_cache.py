"""
file: inference/cache/result_cache.py

Result / lookup cache (GATI-style): memoizes the model output of a whole episode.

  fresh mode      : key = (model fp, "F", hash of the episode inputs). A hit skips the
                    forward pass entirely.
  persistent mode : the output also depends on the carried-over state, so the key adds
                    a history CHAIN hash (see keys.chain_next) and the entry also stores
                    the resulting state, so the run can continue after a hit.

Exact-match only, deterministic models only (the setup gate refuses stochastic write
heads unless overridden).
"""

from __future__ import annotations

import torch

from initium.inference.cache.base_cache import BaseCache, EpisodeCtx, EpisodeHit
from initium.inference.cache.state_utils import restore, snapshot


class ResultCache(BaseCache):
    name = "result"

    def applicable(self, *, reset_experience: bool, resumable: bool) -> tuple[bool, str]:
        return True, ""

    def _key_for(self, ctx: EpisodeCtx) -> str:
        return self._key("F" if ctx.reset_experience else "P", ctx.chain, ctx.in_hash)

    def lookup_episode(self, ctx: EpisodeCtx) -> EpisodeHit | None:
        self.stats.lookups += 1
        got = self._get(self._key_for(ctx))
        if got is None:
            self.stats.misses += 1
            return None
        entry, tier = got
        payload = entry["payload"]
        if not ctx.reset_experience and payload.get("hidden") is None:
            self.stats.misses += 1
            return None
        self.stats.hit(tier, entry["compute_ms"])
        hidden = (
            restore(payload["hidden"], self.device) if payload.get("hidden") is not None else None
        )
        return EpisodeHit(payload["output"].clone(), hidden, entry["compute_ms"])

    def store_episode(
        self, ctx: EpisodeCtx, output: torch.Tensor, hidden_after, compute_ms: float
    ) -> None:
        payload = {
            "output": output.detach().cpu().clone(),
            "hidden": None if ctx.reset_experience else snapshot(hidden_after),
        }
        self._put(self._key_for(ctx), payload, compute_ms)
