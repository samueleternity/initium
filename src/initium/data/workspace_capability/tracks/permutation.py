"""Permutation-composition episodes with exact causal state oracle."""

from __future__ import annotations

import random
from collections.abc import Iterable

from initium.data.workspace_capability.episode import (
    BOS,
    ENTITY_COUNT,
    EV_COMMIT,
    EV_INIT,
    EV_SWAP,
    EOS,
    QUERY,
    SEP,
    TaskEpisode,
    entity_token,
    location_token,
)

ID_ENTITY_IDS = range(ENTITY_COUNT - 16)
OOD_ENTITY_IDS = range(ENTITY_COUNT - 16, ENTITY_COUNT)
_LABEL_SEED_XOR = 0x6A09E667F3BCC909


def generate(
    difficulty: dict,
    rng: random.Random,
    *,
    ood: bool = False,
    ood_axes: Iterable[str] | None = None,
) -> TaskEpisode:
    n = int(difficulty["entities"])
    max_swaps = int(difficulty["swaps"])
    distractors = int(difficulty.get("distractors", 0))
    if n < 2 or n > 16 or max_swaps < 0:
        raise ValueError("permutation difficulty requires 2<=entities<=16 and swaps>=0")

    axes = (
        {str(axis).strip().lower() for axis in ood_axes}
        if ood_axes is not None
        else ({"labels", "length"} if ood else set())
    )
    unknown_axes = axes - {"labels", "length"}
    if unknown_axes:
        raise ValueError(f"unknown permutation OOD axes: {sorted(unknown_axes)}")

    # Keep structure and label sampling independent. This makes label-only
    # evaluations structurally paired with ID episodes when seeded equally.
    structure_seed = rng.getrandbits(64)
    structure_rng = random.Random(structure_seed)
    label_rng = random.Random(structure_seed ^ _LABEL_SEED_XOR)
    label_pool = OOD_ENTITY_IDS if "labels" in axes else ID_ENTITY_IDS
    object_ids = label_rng.sample(tuple(label_pool), n)
    positions = list(range(n))
    structure_rng.shuffle(positions)
    state = dict(zip(object_ids, positions))
    position_to_object = {position: obj for obj, position in state.items()}
    stream = [BOS]
    state_trace = []
    causal_state = {}
    for obj, position in state.items():
        stream.extend((EV_INIT, entity_token(obj), location_token(position), SEP))
        causal_state[obj] = position
        state_trace.append(
            {
                "step": len(stream) - 1,
                "event": "init",
                "event_index": len(causal_state) - 1,
                "positions": tuple(causal_state.get(entity, -1) for entity in range(ENTITY_COUNT)),
            }
        )

    query_by_object = structure_rng.random() < 0.5
    query_obj = object_ids[0]
    query_pos = state[query_obj] if query_by_object else structure_rng.randrange(n)
    swap_count = structure_rng.randint(0, max_swaps)
    relevant = 0
    for swap_index in range(swap_count):
        left, right = structure_rng.sample(object_ids, 2)
        occupant = position_to_object[query_pos]
        relevant += int(
            (left == query_obj or right == query_obj)
            if query_by_object
            else (left == occupant or right == occupant)
        )
        stream.extend(
            (
                EV_SWAP,
                entity_token(left),
                entity_token(right),
                EV_COMMIT,
                entity_token(left),
                entity_token(right),
                SEP,
            )
        )
        p_left, p_right = state[left], state[right]
        state[left], state[right] = p_right, p_left
        position_to_object[p_left], position_to_object[p_right] = right, left
        causal_state[left], causal_state[right] = p_right, p_left
        state_trace.append(
            {
                "step": len(stream) - 1,
                "event": "swap",
                "event_index": swap_index,
                "positions": tuple(causal_state.get(entity, -1) for entity in range(ENTITY_COUNT)),
            }
        )
    for _ in range(distractors):
        # Distractors leave the queried entity or position untouched.
        protected = query_obj if query_by_object else position_to_object[query_pos]
        candidates = [obj for obj in object_ids if obj != protected]
        if len(candidates) < 2:
            continue
        left, right = structure_rng.sample(candidates, 2)
        stream.extend(
            (
                EV_SWAP,
                entity_token(left),
                entity_token(right),
                EV_COMMIT,
                entity_token(left),
                entity_token(right),
                SEP,
            )
        )
        p_left, p_right = state[left], state[right]
        state[left], state[right] = p_right, p_left
        position_to_object[p_left], position_to_object[p_right] = right, left
        causal_state[left], causal_state[right] = p_right, p_left
        state_trace.append(
            {
                "step": len(stream) - 1,
                "event": "distractor",
                "event_index": len(state_trace) - n - swap_count,
                "positions": tuple(causal_state.get(entity, -1) for entity in range(ENTITY_COUNT)),
            }
        )

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
            "generalization_axis": str(
                difficulty.get(
                    "generalization_axis",
                    "combined" if axes else "in_distribution",
                )
            ),
            "report_generalization_axis": bool(difficulty.get("report_generalization_axis", False)),
            "label_split": "ood" if "labels" in axes else "id",
            "entity_label_min": min(object_ids),
            "entity_label_max": max(object_ids),
            "causal_swap_count": swap_count,
            "state_trace": tuple(state_trace),
        },
    )
