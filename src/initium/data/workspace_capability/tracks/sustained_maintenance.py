"""Recall an item after a parametrically controlled distractor delay."""

from __future__ import annotations

import random

from initium.data.workspace_capability.episode import (
    BOS, EOS, EV_INIT, EV_UPDATE, QUERY, SEP, SCALAR_COUNT, TaskEpisode,
    entity_token, scalar_token,
)
from initium.data.workspace_capability.tracks._shared import common_metadata


def generate(difficulty: dict, rng: random.Random, *, ood: bool = False) -> TaskEpisode:
    delay = rng.randint(0, min(12, int(difficulty.get("maximum_delay", 8)) + 2 * int(ood)))
    item = rng.randrange(SCALAR_COUNT)
    stream = [BOS, EV_INIT, entity_token(0), scalar_token(item), SEP]
    for _ in range(delay):
        distractor = 1 + rng.randrange(8)
        stream.extend((EV_UPDATE, entity_token(distractor), scalar_token(rng.randrange(SCALAR_COUNT)), SEP))
    stream.extend((QUERY, entity_token(0), SEP))
    return TaskEpisode(
        "sustained_maintenance", tuple(stream), (scalar_token(item), EOS),
        common_metadata(
            track="sustained_maintenance", sequence_length=len(stream),
            query_type=f"delay_{delay}", relevant_updates=1, distractor_count=delay,
            candidate_baseline_pct=100.0 / SCALAR_COUNT, delay_events=delay,
            progression_group="all",
            target_value=item, distractors_do_not_update_target=True,
        ),
    )
