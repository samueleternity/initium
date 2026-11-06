"""Small helpers shared by the synthetic workspace task generators."""

from __future__ import annotations

from typing import Any


def common_metadata(
    *,
    track: str,
    sequence_length: int,
    query_type: str,
    relevant_updates: int = 0,
    distractor_count: int = 0,
    hop_count: int = 0,
    source_count: int = 0,
    candidate_baseline_pct: float = 50.0,
    **extra: Any,
) -> dict[str, Any]:
    """Return the stable diagnostic fields consumed by the shared evaluator."""
    return {
        "track": track,
        "sequence_length": int(sequence_length),
        "query_type": str(query_type),
        "relevant_updates": int(relevant_updates),
        "distractor_count": int(distractor_count),
        "hop_count": int(hop_count),
        "correction_relevant": False,
        "novelty_distractor": False,
        "source_count": int(source_count),
        "candidate_baseline_pct": float(candidate_baseline_pct),
        **extra,
    }
