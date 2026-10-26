"""Permutation-composition episodes with exact causal state oracle."""

from __future__ import annotations

import random

from initium.data.workspace_capability.episode import (
    BOS,
    ENTITY_COUNT,
    EV_INIT,
    EV_SWAP,
    EOS,
    QUERY,
    SEP,
    TaskEpisode,
    entity_token,
    location_token,
)


def generate(difficulty: dict, rng: random.Random, *, ood: bool = False) -> TaskEpisode:
    n = int(difficulty["entities"])
    max_swaps = int(difficulty["swaps"])
    distractors = int(difficulty.get("distractors", 0))
    if n < 2 or n > 16 or max_swaps < 0:
        raise ValueError("permutation difficulty requires 2<=entities<=16 and swaps>=0")

    labels = list(range(ENTITY_COUNT))
    rng.shuffle(labels)
    object_ids = labels[:n]
    # OOD uses a label band never used by the in-distribution sampler.
    if ood:
        object_ids = list(range(ENTITY_COUNT - n, ENTITY_COUNT))
    positions = list(range(n))
    rng.shuffle(positions)
    state = dict(zip(object_ids, positions))
    position_to_object = {position: obj for obj, position in state.items()}
    stream = [BOS]
    for obj, position in state.items():
        stream.extend((EV_INIT, entity_token(obj), location_token(position), SEP))

    query_by_object = rng.random() < 0.5
    query_obj = object_ids[0]
    query_pos = state[query_obj] if query_by_object else rng.randrange(n)
    swap_count = rng.randint(0, max_swaps)
    relevant = 0
    for _ in range(swap_count):
        left, right = rng.sample(object_ids, 2)
        occupant = position_to_object[query_pos]
        relevant += int(
            (left == query_obj or right == query_obj)
            if query_by_object
            else (left == occupant or right == occupant)
        )
        stream.extend((EV_SWAP, entity_token(left), entity_token(right), SEP))
        p_left, p_right = state[left], state[right]
        state[left], state[right] = p_right, p_left
        position_to_object[p_left], position_to_object[p_right] = right, left
    for _ in range(distractors):
        # Distractors leave the queried entity or position untouched.
        protected = query_obj if query_by_object else position_to_object[query_pos]
        candidates = [obj for obj in object_ids if obj != protected]
        if len(candidates) < 2:
            continue
        left, right = rng.sample(candidates, 2)
        stream.extend((EV_SWAP, entity_token(left), entity_token(right), SEP))
        p_left, p_right = state[left], state[right]
        state[left], state[right] = p_right, p_left
        position_to_object[p_left], position_to_object[p_right] = right, left

    if query_by_object:
        answer = (location_token(state[query_obj]), EOS)
        query_type = "object_to_position"
        query_tokens = (entity_token(query_obj),)
    else:
        query_obj = position_to_object[query_pos]
        answer = (entity_token(query_obj), EOS)
        query_type = "position_to_object"
        query_tokens = (location_token(query_pos),)
    stream.extend((QUERY, *query_tokens, SEP))
    return TaskEpisode(
        "permutation",
        tuple(stream),
        answer,
        {
            "query_type": query_type,
            "sequence_length": len(stream),
            "relevant_updates": relevant,
            "distractor_count": distractors,
            "entity_count": n,
            "hop_count": 1,
            "source_count": 1,
            "correction_relevant": False,
            "novelty_distractor": False,
            "answer_count": 1,
            "candidate_count": n,
            "candidate_baseline_pct": 100.0 / n,
            "difficulty": max_swaps,
        },
    )
