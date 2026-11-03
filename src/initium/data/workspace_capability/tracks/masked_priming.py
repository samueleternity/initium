"""Masked versus visible prime with a downstream target response."""

from __future__ import annotations

import random

from initium.data.workspace_capability.episode import (
    BOS, EOS, EV_INIT, EV_NOISE, QUERY, SEP,
    TaskEpisode, entity_token, scalar_token,
)
from initium.data.workspace_capability.tracks._shared import common_metadata


def generate(difficulty: dict, rng: random.Random, *, ood: bool = False) -> TaskEpisode:
    masked = rng.random() < float(difficulty.get("masked_probability", 0.5))
    target = rng.randrange(2)
    congruent = bool(rng.getrandbits(1))
    prime = target if congruent else 1 - target
    target_strength = rng.randrange(3)
    target_observation = target if rng.random() < (0.55 + 0.2 * target_strength) else 1 - target
    stream = [BOS]
    if masked:
        # A masked prime remains available to local sequence processing but is
        # not marked as a workspace-addressed event.
        stream.extend((EV_NOISE, scalar_token(prime), SEP))
    else:
        stream.extend((EV_INIT, entity_token(0), scalar_token(prime), SEP))
    stream.extend((
        EV_INIT, entity_token(1), scalar_token(target_observation), SEP,
        EV_NOISE, scalar_token(target_strength), SEP, QUERY, entity_token(1), SEP,
    ))
    # Score the perceptual response supported by the presented target evidence.
    # The latent target is retained for perceptual-error diagnostics only.
    response = target_observation
    return TaskEpisode(
        "masked_priming", tuple(stream), (scalar_token(response), EOS),
        common_metadata(
            track="masked_priming", sequence_length=len(stream),
            query_type=f"{'masked' if masked else 'visible'}_{'congruent' if congruent else 'incongruent'}",
            relevant_updates=2, distractor_count=int(masked),
            candidate_baseline_pct=50.0, mask_condition=masked,
            prime_value=prime, target_value=response, latent_target=target,
            prime_congruent=congruent,
            target_strength=target_strength, target_observation=target_observation,
            progression_group="all",
            target_only_report=True, masked_prime_is_not_workspace_addressed=masked,
        ),
    )
