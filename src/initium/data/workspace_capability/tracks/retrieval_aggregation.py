"""Delayed associative recall, multi-query, and bounded aggregation tasks."""

from __future__ import annotations

import random

from initium.data.workspace_capability.episode import (
    BOS,
    ENTITY_COUNT,
    COUNT_COUNT,
    EV_KV,
    EV_UPDATE,
    EV_COUNT_QUERY,
    EV_SUM_QUERY,
    EV_PARITY_QUERY,
    EOS,
    QUERY,
    SEP,
    TaskEpisode,
    count_token,
    entity_token,
    scalar_token,
)


def generate(difficulty: dict, rng: random.Random, *, ood: bool = False) -> TaskEpisode:
    num_keys = int(difficulty["keys"])
    updates = int(difficulty.get("updates", 0))
    distractors = int(difficulty.get("distractors", 0))
    query_count = min(int(difficulty.get("queries", 1)), num_keys)
    aggregate = difficulty.get("aggregate", "recall")
    if not 1 <= num_keys <= 32 or not 1 <= query_count <= num_keys:
        raise ValueError("retrieval task requires 1..32 keys and valid query count")
    key_start = 32 if ood else 0
    keys = list(range(key_start, key_start + num_keys))
    state = {key: rng.randrange(16) for key in keys}
    stream = [BOS]
    for key in keys:
        stream.extend((EV_KV, entity_token(key), scalar_token(state[key]), SEP))
    for _ in range(updates):
        key = rng.choice(keys)
        value = rng.randrange(16)
        state[key] = value
        stream.extend((EV_UPDATE, entity_token(key), scalar_token(value), SEP))
    for _ in range(distractors):
        # Out-of-query vocabulary writes create irrelevant but plausible facts.
        unrelated_keys = [key for key in range(ENTITY_COUNT) if key not in keys]
        if not unrelated_keys:
            break
        key = rng.choice(unrelated_keys)
        stream.extend((EV_KV, entity_token(key), scalar_token(rng.randrange(16)), SEP))
    query_keys = rng.sample(keys, query_count)
    query_op = {
        "recall": None,
        "sum_mod16": EV_SUM_QUERY,
        "count_even": EV_COUNT_QUERY,
        "parity": EV_PARITY_QUERY,
    }.get(aggregate)
    if aggregate not in {"recall", "sum_mod16", "count_even", "parity"}:
        raise ValueError(f"unknown aggregation task {aggregate!r}")
    stream.append(QUERY)
    if query_op is not None:
        stream.append(query_op)
    stream.extend(entity_token(key) for key in query_keys)
    stream.append(SEP)

    values = [state[key] for key in query_keys]
    if aggregate == "recall":
        answer = tuple(scalar_token(value) for value in values) + (EOS,)
        candidate_baseline = 100.0 / 16.0
    elif aggregate == "sum_mod16":
        answer = (scalar_token(sum(values) % 16), EOS)
        candidate_baseline = 100.0 / 16.0
    elif aggregate == "count_even":
        answer = (count_token(sum(value % 2 == 0 for value in values)), EOS)
        candidate_baseline = 100.0 / (query_count + 1)
    elif aggregate == "parity":
        parity = sum(values) % 2
        from initium.data.workspace_capability.episode import BOOL_FALSE, BOOL_TRUE

        answer = (BOOL_TRUE if parity else BOOL_FALSE, EOS)
        candidate_baseline = 50.0
    else:
        raise ValueError(f"unknown aggregation task {aggregate!r}")
    return TaskEpisode(
        "retrieval_aggregation",
        tuple(stream),
        answer,
        {
            "query_type": aggregate,
            "sequence_length": len(stream),
            "relevant_updates": updates,
            "distractor_count": distractors,
            "entity_count": num_keys,
            "hop_count": 1,
            "source_count": 1,
            "correction_relevant": updates > 0,
            "novelty_distractor": distractors > 0,
            "answer_count": len(answer) - 1,
            "candidate_count": max(2, round(100.0 / candidate_baseline)),
            "candidate_baseline_pct": candidate_baseline,
            "difficulty": num_keys + updates + distractors,
        },
    )
