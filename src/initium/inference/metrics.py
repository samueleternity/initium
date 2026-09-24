"""
file: inference/metrics.py

Task-agnostic scoring containers and aggregation. A task's score_episode()
returns an EpisodeScore; everything downstream (breakdowns, windows,
adaptation trend, logging) only ever sees EpisodeScore, so a new task type
needs no changes here.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple

import numpy as np


@dataclass
class EpisodeScore:
    n_items: int                     # scored items in the episode (graph: answer triples)
    n_correct: int
    perfect: bool                    # every item correct
    group: Any = None                # bucket key for breakdowns (graph: walk length / hop count)
    fields: Dict[str, Tuple[int, int]] = field(default_factory=dict)  # name -> (correct, total)

    @property
    def item_acc(self) -> float:
        return 100.0 * self.n_correct / max(self.n_items, 1)


@dataclass
class EpisodeResult:
    index: int
    score: EpisodeScore
    elapsed_ms: float
    reset_experience: bool
    cache_event: str = ""            # "", "miss", "hit:result", "hit:prefix@22", ...

def aggregate(scores: List[EpisodeScore]) -> dict:
    n_eps = len(scores)
    n_items = sum(s.n_items for s in scores)
    n_correct = sum(s.n_correct for s in scores)
    n_perfect = sum(int(s.perfect) for s in scores)

    by_group: Dict[Any, List[int]] = {}
    field_tot: Dict[str, List[int]] = {}
    for s in scores:
        if s.group is not None:
            g = by_group.setdefault(s.group, [0, 0, 0, 0])
            g[0] += s.n_items
            g[1] += s.n_correct
            g[2] += 1
            g[3] += int(s.perfect)
        for name, (c, t) in s.fields.items():
            f = field_tot.setdefault(name, [0, 0])
            f[0] += c
            f[1] += t

    return {
        "n_episodes": n_eps,
        "item_acc": 100.0 * n_correct / max(n_items, 1),
        "perfect_frac": 100.0 * n_perfect / max(n_eps, 1),
        "by_group": {
            g: {"item_acc": 100.0 * v[1] / max(v[0], 1),
                "perfect_frac": 100.0 * v[3] / max(v[2], 1),
                "n_episodes": v[2]}
            for g, v in sorted(by_group.items())
        },
        "field_acc": {n: 100.0 * c / max(t, 1) for n, (c, t) in field_tot.items()},
    }


def windowed(scores: List[EpisodeScore], window: int) -> List[dict]:
    """Aggregate consecutive chunks of `window` episodes (persistent-mode adaptation view)."""
    window = max(1, window)
    out = []
    for w, start in enumerate(range(0, len(scores), window)):
        chunk = scores[start:start + window]
        agg = aggregate(chunk)
        agg.update(window_index=w, start_episode=start, end_episode=start + len(chunk) - 1)
        out.append(agg)
    return out


def adaptation_trend(scores: List[EpisodeScore], window: int) -> dict:
    """First-window vs last-window accuracy and a least-squares slope of
    per-episode accuracy -- a compact 'did it adapt mid-inference' read."""
    if len(scores) < 2:
        return {}
    w = max(1, min(window, len(scores) // 2))
    first, last = aggregate(scores[:w]), aggregate(scores[-w:])
    accs = np.array([s.item_acc for s in scores], dtype=np.float64)
    slope = float(np.polyfit(np.arange(len(accs)), accs, 1)[0])
    return {
        "window": w,
        "first_window_item_acc": first["item_acc"],
        "last_window_item_acc": last["item_acc"],
        "delta_item_acc": last["item_acc"] - first["item_acc"],
        "slope_acc_per_episode": slope,
    }