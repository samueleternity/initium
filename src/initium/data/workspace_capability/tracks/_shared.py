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

EXT_OP_NAMES = ("sub", "mul", "neg", "inc", "dec", "eq", "and", "or", "xor", "not")
EXT_OP_IDS = {"sub": 8, "and": 9, "or": 10, "xor": 11, "not": 12,
              "mul": 13, "neg": 14, "inc": 15, "dec": 16, "eq": 17}
EXT_UNARY = {"neg", "inc", "dec", "not"}


def ext_op_result(name: str, a: int, b: int) -> int:
    return {
        "sub": (a - b) % 16, "mul": a * b % 16, "neg": -a % 16,
        "inc": (a + 1) % 16, "dec": (a - 1) % 16, "eq": int(a == b),
        "and": int(a == 1 and b == 1), "or": int(a == 1 or b == 1),
        "xor": int((a == 1) != (b == 1)), "not": int(a != 1),
    }[name]
