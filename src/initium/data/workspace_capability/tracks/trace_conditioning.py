"""Delay-versus-trace associative prediction with a controlled cue gap."""

from __future__ import annotations

import random

from initium.data.workspace_capability.episode import (
    BOS, BOOL_FALSE, BOOL_TRUE, EOS, EV_INIT, EV_UPDATE, QUERY, SEP,
    SCALAR_COUNT, TaskEpisode, entity_token, scalar_token,
)
from initium.data.workspace_capability.tracks._shared import common_metadata


def generate(difficulty: dict, rng: random.Random, *, ood: bool = False) -> TaskEpisode:
    trace = bool(rng.getrandbits(1))
    gap = rng.randint(1, min(6, int(difficulty.get("maximum_gap", 4)) + int(ood))) if trace else 0
    cue = rng.randrange(8)
    outcome = bool(rng.getrandbits(1))
    stream = [BOS, EV_INIT, entity_token(cue), scalar_token(cue), SEP]
    for _ in range(gap):
        distractor_key = 8 + rng.randrange(8)
        stream.extend((
            EV_UPDATE, entity_token(distractor_key),
            scalar_token(rng.randrange(SCALAR_COUNT)), SEP,
        ))
    stream.extend((EV_UPDATE, entity_token(cue), BOOL_TRUE if outcome else BOOL_FALSE, SEP))
    stream.extend((QUERY, entity_token(cue), SEP))
    return TaskEpisode(
        "trace_conditioning", tuple(stream), (BOOL_TRUE if outcome else BOOL_FALSE, EOS),
        common_metadata(
            track="trace_conditioning", sequence_length=len(stream),
            query_type=f"{'trace' if trace else 'delay'}_gap_{gap}",
            relevant_updates=2, distractor_count=gap, candidate_baseline_pct=50.0,
            trace_condition=trace, cue_outcome_pair=(cue, outcome), gap_events=gap,
            progression_group="all",
        ),
    )
