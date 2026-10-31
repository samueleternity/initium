"""Multi-operation episodes whose final answer concatenates skill outputs."""

from __future__ import annotations

import random

from initium.data.workspace_capability.episode import (
    BOS,
    ENTITY_COUNT,
    EOS,
    EV_KV,
    EV_VECTOR,
    QUERY,
    SCALAR_COUNT,
    SEP,
    TaskEpisode,
    entity_token,
    scalar_token,
    source_token,
)
from initium.data.workspace_capability.tracks._shared import common_metadata

_SKILLS = ("lookup", "arithmetic", "copy", "sort")


def generate(difficulty: dict, rng: random.Random, *, ood: bool = False) -> TaskEpisode:
    operation_count = min(max(2, int(difficulty["operations"])), 4)
    keys = min(max(4, int(difficulty["keys"])), 10 if ood else 8)
    table = {key: rng.randrange(SCALAR_COUNT) for key in range(keys)}
    skills = list(_SKILLS)
    rng.shuffle(skills)
    skills = skills[:operation_count]
    table_values = list(table.values())

    stream = [BOS]
    for key, value in table.items():
        stream.extend((EV_KV, entity_token(key), scalar_token(value), SEP))

    outputs: list[int] = []
    stage_ids = [ENTITY_COUNT - 4 + index for index in range(operation_count)]
    for stage_id, skill in zip(stage_ids, skills):
        if skill == "lookup":
            key = rng.randrange(keys)
            result = (table[key],)
            args = (entity_token(key),)
            skill_id = 0
        elif skill == "arithmetic":
            left, right = rng.randrange(SCALAR_COUNT), rng.randrange(SCALAR_COUNT)
            result = ((left + right) % SCALAR_COUNT,)
            args = (scalar_token(left), scalar_token(right))
            skill_id = 1
        elif skill == "copy":
            value = rng.choice(table_values)
            result = (value,)
            args = (scalar_token(value),)
            skill_id = 2
        else:
            left, right = rng.sample(range(SCALAR_COUNT), 2)
            result = tuple(sorted((left, right)))
            args = (scalar_token(left), scalar_token(right))
            skill_id = 3
        # Stage IDs are the semantic workspace addresses. The skill token
        # selects the operation, and the output itself is absent from input.
        stream.extend((EV_VECTOR, entity_token(stage_id), source_token(skill_id), *args, SEP))
        outputs.extend(scalar_token(value) for value in result)

    stream.extend((QUERY, *(entity_token(stage_id) for stage_id in stage_ids), SEP))
    answer = (*outputs, EOS)
    return TaskEpisode(
        "mixed_skill",
        tuple(stream),
        answer,
        common_metadata(
            track="mixed_skill",
            sequence_length=len(stream),
            query_type=f"ops{operation_count}_" + ";".join(skills),
            relevant_updates=operation_count,
            distractor_count=keys,
            source_count=len(_SKILLS),
            candidate_baseline_pct=100.0 / (SCALAR_COUNT ** max(1, len(outputs))),
            operation_count=operation_count,
            skill_sequence=tuple(skills),
            output_count=len(outputs),
            ood=bool(ood),
        ),
    )
