"""Noisy graded-evidence detection probe for report/access curves.

Strength changes the reliability of each sensory sample; it does not directly
specify the correct answer. This lets evaluation estimate hits and false
alarms at every strength instead of fitting a curve to a deterministic label
threshold.
"""

from __future__ import annotations

import random

from initium.data.workspace_capability.episode import (
    BOS,
    BOOL_FALSE,
    BOOL_TRUE,
    EOS,
    EV_INIT,
    EV_UPDATE,
    QUERY,
    SEP,
    TaskEpisode,
    entity_token,
    scalar_token,
)
from initium.data.workspace_capability.tracks._shared import common_metadata


def generate(difficulty: dict, rng: random.Random, *, ood: bool = False) -> TaskEpisode:
    maximum = min(8, int(difficulty.get("maximum_strength", 6)) + int(ood))
    strength = rng.randint(0, maximum)
    target_present = bool(rng.getrandbits(1))
    sample_count = 5

    # Symmetric signal-detection channel. At zero strength the evidence is
    # uninformative; at maximum strength, present and absent trials are still
    # noisy rather than deterministic.
    signal_probability = 0.5 + 0.05 * strength
    evidence_probability = (
        signal_probability if target_present else 1.0 - signal_probability
    )
    observations = tuple(
        int(rng.random() < evidence_probability) for _ in range(sample_count)
    )

    stream = [BOS]
    for index, observation in enumerate(observations):
        stream.extend(
            (
                EV_INIT if index == 0 else EV_UPDATE,
                entity_token(index + 1),
                scalar_token(observation),
                SEP,
            )
        )
    stream.extend((QUERY, *(entity_token(index + 1) for index in range(sample_count)), SEP))
    answer = BOOL_TRUE if target_present else BOOL_FALSE

    return TaskEpisode(
        "ignition",
        tuple(stream),
        (answer, EOS),
        common_metadata(
            track="ignition",
            sequence_length=len(stream),
            query_type=f"strength_{strength}_{'present' if target_present else 'absent'}",
            relevant_updates=sample_count,
            candidate_baseline_pct=50.0,
            stimulus_strength=strength,
            target_present=target_present,
            signal_probability=signal_probability,
            sensory_observations=observations,
            report=target_present,
            progression_group="all",
            workspace_broadcast_target=target_present,
        ),
    )
