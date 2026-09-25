"""
file: inference/cache/prefix_cache.py

Prefix STATE cache (the DNC analogue of a KV / context cache). After the static
prefix of an episode (e.g. the whole edge listing) has been processed, the complete
state (controller state, Memory dict incl. link matrix / usage / weights, last read
vector) is snapshotted, keyed by (model fingerprint, hash of the prefix tokens).
Later episodes with the SAME prefix restore it and process only their suffix.

Valid because the model is causal and deterministic: state after step p depends only
on inputs [0, p). Requires reset_experience=True (identical start state) and a task
that declares cache_boundaries. If several boundaries are declared, the LONGEST cached
one is used (longest-prefix match) and the remaining ones are stored on the way.
"""

from __future__ import annotations

from initium.inference.cache.base_cache import BaseCache, EpisodeCtx, ResumePoint
from initium.inference.cache.state_utils import restore, snapshot


class PrefixStateCache(BaseCache):
    name = "prefix"
    wants_boundaries = True

    def applicable(self, *, reset_experience: bool, resumable: bool) -> tuple[bool, str]:
        if not reset_experience:
            return False, (
                "needs reset_experience=True: in persistent mode every episode starts "
                "from a different state, so a prefix snapshot is never reusable"
            )
        if not resumable:
            return False, (
                "this model cannot resume mid-sequence (SplitGraphDNC needs the "
                "start_step edit in split_graph_dnc.py)"
            )
        return True, ""

    def _key_for(self, ctx: EpisodeCtx, boundary: int) -> str:
        return self._key(boundary, ctx.prefix_hash(boundary))

    def find_resume(self, ctx: EpisodeCtx) -> ResumePoint | None:
        if not ctx.boundaries:
            return None
        self.stats.lookups += 1
        for b in reversed(ctx.boundaries):  # longest prefix first
            got = self._get(self._key_for(ctx, b))
            if got is None:
                continue
            entry, tier = got
            self.stats.hit(tier, entry["compute_ms"])
            return ResumePoint(
                b, restore(entry["payload"], self.device), entry["compute_ms"], self.name
            )
        self.stats.misses += 1
        return None

    def store_boundary(self, ctx: EpisodeCtx, boundary: int, hidden, compute_ms: float) -> None:
        self._put(self._key_for(ctx, boundary), snapshot(hidden), compute_ms)
