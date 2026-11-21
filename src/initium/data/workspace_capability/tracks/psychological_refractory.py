"""Paired-response task with a controlled stimulus-onset asynchrony."""

from __future__ import annotations

import random

from initium.data.workspace_capability.episode import (
    BOS, EOS, EV_INIT, EV_UPDATE, QUERY, SEP, SCALAR_COUNT, TaskEpisode,
    entity_token, scalar_token,
)
from initium.data.workspace_capability.tracks._shared import common_metadata


def generate(difficulty: dict, rng: random.Random, *, ood: bool = False) -> TaskEpisode:
    maximum_soa = min(8, int(difficulty.get("maximum_soa", 5)) + int(ood))
    soa = rng.randint(0, maximum_soa)
    first = rng.randrange(SCALAR_COUNT)
    second = rng.randrange(SCALAR_COUNT)
    stream = [BOS, EV_INIT, entity_token(0), scalar_token(first), SEP]
    # Neutral, query-irrelevant events occupy the interval between responses.
    for _ in range(soa):
        stream.extend((EV_UPDATE, entity_token(2 + rng.randrange(8)), scalar_token(rng.randrange(SCALAR_COUNT)), SEP))
    stream.extend((EV_INIT, entity_token(1), scalar_token(second), SEP, QUERY, entity_token(0), entity_token(1), SEP))
    return TaskEpisode(
        "psychological_refractory", tuple(stream),
        (scalar_token(first), scalar_token(second), EOS),
        common_metadata(
            track="psychological_refractory", sequence_length=len(stream),
            query_type=f"soa_{soa}", relevant_updates=2, distractor_count=soa,
            candidate_baseline_pct=100.0 / SCALAR_COUNT, soa=soa,
            progression_group="all",
            response_order=(0, 1), second_task_requires_serial_access=True,
        ),
    )
