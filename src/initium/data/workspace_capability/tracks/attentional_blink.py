"""RSVP-style dual-target identification with lag and symbolic mask controls."""

from __future__ import annotations

import random

from initium.data.workspace_capability.episode import (
    BOS,
    EOS,
    QUERY,
    RELATION_COUNT,
    SCALAR_COUNT,
    SEP,
    TaskEpisode,
    entity_token,
    relation_token,
    scalar_token,
)
from initium.data.workspace_capability.tracks._shared import common_metadata


def generate(difficulty: dict, rng: random.Random, *, ood: bool = False) -> TaskEpisode:
    maximum_lag = min(max(1, int(difficulty["maximum_lag"])), 9 if ood else 8)
    lag = rng.randint(1, maximum_lag)
    mask_applied = bool(rng.getrandbits(1)) and lag > 1
    target_one = rng.randrange(SCALAR_COUNT)
    target_two = rng.randrange(SCALAR_COUNT)
    if target_two == target_one:
        target_two = (target_two + rng.randrange(1, SCALAR_COUNT)) % SCALAR_COUNT
    stream = [BOS]
    stream.extend(relation_token(rng.randrange(RELATION_COUNT)) for _ in range(rng.randint(2, 4)))
    stream.append(scalar_token(target_one))
    # Lag counts temporal items between T1 and T2. The symbolic mask occupies
    # an ordinary RSVP slot and is varied independently of the target answer.
    for gap_index in range(lag - 1):
        if mask_applied and gap_index == 0:
            stream.append(entity_token(0))
        else:
            stream.append(relation_token(rng.randrange(RELATION_COUNT)))
    stream.append(scalar_token(target_two))
    stream.extend(relation_token(rng.randrange(RELATION_COUNT)) for _ in range(2))
    stream.extend((QUERY, SEP))
    return TaskEpisode(
        "attentional_blink",
        tuple(stream),
        (scalar_token(target_one), scalar_token(target_two), EOS),
        common_metadata(
            track="attentional_blink",
            sequence_length=len(stream),
            query_type=f"lag_{lag}_{'masked' if mask_applied else 'unmasked'}",
            relevant_updates=2,
            distractor_count=len(stream) - 5,
            candidate_baseline_pct=100.0 / SCALAR_COUNT,
            target_lag=lag,
            mask_condition=mask_applied,
            target_one=target_one,
            target_two=target_two,
            ood=bool(ood),
            symbolic_mask=True,
        ),
    )
