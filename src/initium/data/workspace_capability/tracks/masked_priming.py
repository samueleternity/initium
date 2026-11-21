"""Masked/visible priming with balanced, separately noisy evidence channels.

The response is the latent target category inferred from several imperfect
target observations. Prime congruency is defined against that latent target,
so incongruent trials are genuine conflict trials rather than accidental
disagreements with a single noisy target token.
"""

from __future__ import annotations

import random

from initium.data.workspace_capability.episode import (
    BOS,
    EOS,
    EV_INIT,
    EV_NOISE,
    EV_UPDATE,
    QUERY,
    SEP,
    TaskEpisode,
    entity_token,
    scalar_token,
)
from initium.data.workspace_capability.tracks._shared import common_metadata


def generate(difficulty: dict, rng: random.Random, *, ood: bool = False) -> TaskEpisode:
    masked = rng.random() < float(difficulty.get("masked_probability", 0.5))
    target = rng.randrange(2)
    congruent = bool(rng.getrandbits(1))
    latent_prime = target if congruent else 1 - target

    # Visibility affects prime identification but does not erase all local
    # information. The encoded observation, rather than the latent prime, is
    # available to the model and to the auxiliary workspace trace probe.
    prime_accuracy = 0.68 if masked else 0.96
    observed_prime = (
        latent_prime
        if rng.random() < prime_accuracy
        else 1 - latent_prime
    )

    target_strength = rng.randrange(3)
    target_accuracy = (0.62, 0.76, 0.90)[target_strength]
    evidence_count = 5
    observations = tuple(
        target if rng.random() < target_accuracy else 1 - target
        for _ in range(evidence_count)
    )

    stream = [BOS]
    stream.extend(
        (
            EV_NOISE if masked else EV_INIT,
            *( () if masked else (entity_token(0),) ),
            scalar_token(observed_prime),
            SEP,
            EV_NOISE,
            scalar_token(target_strength),
            SEP,
        )
    )
    for index, observation in enumerate(observations):
        stream.extend(
            (
                EV_INIT if index == 0 else EV_UPDATE,
                entity_token(index + 1),
                scalar_token(observation),
                SEP,
            )
        )
    stream.extend((QUERY, *(entity_token(index + 1) for index in range(evidence_count)), SEP))

    return TaskEpisode(
        "masked_priming",
        tuple(stream),
        (scalar_token(target), EOS),
        common_metadata(
            track="masked_priming",
            sequence_length=len(stream),
            query_type=f"{'masked' if masked else 'visible'}_{'congruent' if congruent else 'incongruent'}",
            relevant_updates=evidence_count + 1,
            distractor_count=int(masked),
            candidate_baseline_pct=50.0,
            mask_condition=masked,
            prime_value=observed_prime,
            latent_prime=latent_prime,
            target_value=target,
            latent_target=target,
            target_strength=target_strength,
            target_accuracy=target_accuracy,
            target_observations=observations,
            prime_observation_accuracy=prime_accuracy,
            prime_congruent=congruent,
            progression_group="all",
            target_only_report=True,
            masked_prime_is_not_workspace_addressed=masked,
            congruency_defined_against_latent_target=True,
        ),
    )
