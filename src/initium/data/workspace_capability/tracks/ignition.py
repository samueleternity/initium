"""Graded-evidence detection probe for all-or-none workspace access."""

from __future__ import annotations

import random

from initium.data.workspace_capability.episode import (
    BOS, BOOL_FALSE, BOOL_TRUE, EOS, EV_INIT, QUERY, SEP, TaskEpisode,
    entity_token, scalar_token,
)
from initium.data.workspace_capability.tracks._shared import common_metadata


def generate(difficulty: dict, rng: random.Random, *, ood: bool = False) -> TaskEpisode:
    maximum = min(8, int(difficulty.get("maximum_strength", 6)) + int(ood))
    threshold = min(4 + int(ood), maximum)
    strength = rng.randint(0, maximum)
    # Repetition controls sensory evidence; the report boundary is a fixed
    # threshold, so the trial family exposes a measurable psychometric curve.
    stream = [BOS, EV_INIT, entity_token(0)]
    stream.extend([scalar_token(1)] * strength)
    stream.extend((SEP, QUERY, entity_token(0), SEP))
    detected = strength >= threshold
    return TaskEpisode(
        "ignition", tuple(stream), (BOOL_TRUE if detected else BOOL_FALSE, EOS),
        common_metadata(
            track="ignition", sequence_length=len(stream),
            query_type=f"strength_{strength}", relevant_updates=1,
            candidate_baseline_pct=50.0, stimulus_strength=strength,
            report_threshold=threshold, report=detected,
            progression_group="all",
            workspace_broadcast_target=detected,
        ),
    )
