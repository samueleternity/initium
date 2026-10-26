"""Quantized-vector Nth-farthest relational comparison episodes."""

from __future__ import annotations

import random

from initium.data.workspace_capability.episode import (
    BOS,
    ENTITY_COUNT,
    EV_VECTOR,
    EV_NOISE,
    EOS,
    QUERY,
    SEP,
    TaskEpisode,
    entity_token,
    scalar_token,
)


def generate(difficulty: dict, rng: random.Random, *, ood: bool = False) -> TaskEpisode:
    count = int(difficulty["candidates"])
    dimensions = int(difficulty.get("dimensions", 3))
    rank = min(int(difficulty.get("rank", count)), count)
    control_probability = float(difficulty.get("control_probability", 0.0))
    distractors = int(difficulty.get("distractors", 0))
    if count < 2 or count > 16 or not 1 <= dimensions <= 6:
        raise ValueError("Nth-farthest requires 2..16 candidates and 1..6 dimensions")
    if not 0.0 <= control_probability <= 1.0:
        raise ValueError("control_probability must be in [0,1]")
    draw = rng.random()
    if draw < control_probability / 2:
        rank = 1
        query_type = "farthest"
    elif draw < control_probability:
        rank = count
        query_type = "nearest"
    else:
        query_type = "nth_farthest"
    start = ENTITY_COUNT - count if ood else 0
    candidates = list(range(start, start + count))
    rng.shuffle(candidates)
    query = [rng.randrange(16) for _ in range(dimensions)]
    vectors = {entity: [rng.randrange(16) for _ in range(dimensions)] for entity in candidates}
    # Regenerate rare exact ties to keep the target unique and the oracle crisp.
    distances = {entity: sum((a - b) ** 2 for a, b in zip(vectors[entity], query))
                 for entity in candidates}
    if len(set(distances.values())) != count:
        # Deterministic no-tie fallback; extra dimensions remain represented,
        # while the first coordinate provides an unambiguous distance order.
        query = [0] * dimensions
        vectors = {
            entity: [entity - start] + [0] * (dimensions - 1)
            for entity in candidates
        }
        distances = {entity: sum(value * value for value in vectors[entity])
                     for entity in candidates}
    ordered = sorted(candidates, key=lambda entity: (-distances[entity], entity))
    answer_entity = ordered[rank - 1]

    stream = [BOS]
    for entity in candidates:
        stream.extend((EV_VECTOR, entity_token(entity)))
        stream.extend(scalar_token(value) for value in vectors[entity])
        stream.append(SEP)
    for _ in range(distractors):
        stream.extend((EV_NOISE, scalar_token(rng.randrange(16)), scalar_token(rng.randrange(16))))
        stream.append(SEP)
    stream.extend((QUERY, scalar_token(rank - 1)))
    stream.extend(scalar_token(value) for value in query)
    stream.append(SEP)
    return TaskEpisode(
        "nth_farthest",
        tuple(stream),
        (entity_token(answer_entity), EOS),
        {
            "query_type": query_type,
            "sequence_length": len(stream),
            "relevant_updates": count,
            "distractor_count": distractors,
            "entity_count": count,
            "hop_count": 1,
            "source_count": 1,
            "correction_relevant": False,
            "novelty_distractor": distractors > 0,
            "answer_count": 1,
            "candidate_count": count,
            "candidate_baseline_pct": 100.0 / count,
            "rank": rank,
            "difficulty": count + dimensions,
        },
    )
