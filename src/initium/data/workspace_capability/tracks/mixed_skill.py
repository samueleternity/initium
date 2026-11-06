"""Dependent pipelines that exercise the workspace's six learned operations."""

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

_SKILL_IDS = {
    "lookup": 0,
    "arithmetic": 1,
    "copy": 2,
    "sort": 3,
    "compare": 4,
    "select": 5,
    "rank": 6,
}


def generate(difficulty: dict, rng: random.Random, *, ood: bool = False) -> TaskEpisode:
    operation_count = min(max(3, int(difficulty["operations"])), 8)
    keys = min(max(4, int(difficulty["keys"])), 12 if ood else 10)
    table = {key: rng.randrange(SCALAR_COUNT) for key in range(keys)}
    stream = [BOS]
    for key, value in table.items():
        stream.extend((EV_KV, entity_token(key), scalar_token(value), SEP))

    outputs: list[tuple[int, ...]] = []
    # Reserve two semantic addresses per stage so a sort operation can commit
    # both ordered outputs without aliasing the next stage's result.
    stage_ids = [ENTITY_COUNT - 16 + 2 * index for index in range(operation_count)]
    skills = ["lookup"] + [
        ("arithmetic", "copy", "sort", "compare", "select", "rank")[(index - 1) % 6]
        for index in range(1, operation_count)
    ]
    for index, (stage_id, skill) in enumerate(zip(stage_ids, skills)):
        previous = outputs[index - 1][0] if index else None
        previous_stage = stage_ids[index - 1] if index else None
        if skill == "lookup":
            # The initial lookup seeds the chain. Later lookup stages are not
            # used so every downstream stage can depend on a prior result.
            key = rng.randrange(keys)
            result = (table[key],)
            args = (entity_token(key),)
        elif skill == "arithmetic":
            right = rng.randrange(SCALAR_COUNT)
            result = ((int(previous) + right) % SCALAR_COUNT,)
            args = (entity_token(int(previous_stage)), scalar_token(right))
        elif skill == "copy":
            result = (int(previous),)
            args = (entity_token(int(previous_stage)),)
        elif skill == "sort":
            right = rng.randrange(SCALAR_COUNT)
            result = tuple(sorted((int(previous), right)))
            args = (entity_token(int(previous_stage)), scalar_token(right))
        elif skill == "compare":
            right = rng.randrange(SCALAR_COUNT)
            result = (int(int(previous) > right),)
            args = (entity_token(int(previous_stage)), scalar_token(right))
        elif skill == "select":
            right = rng.randrange(SCALAR_COUNT)
            result = (max(int(previous), right),)
            args = (entity_token(int(previous_stage)), scalar_token(right))
        else:  # rank: zero is the higher-ranked operand; one is the lower.
            right = rng.randrange(SCALAR_COUNT)
            result = (0 if int(previous) >= right else 1,)
            args = (entity_token(int(previous_stage)), scalar_token(right))
        stream.extend(
            (EV_VECTOR, entity_token(stage_id), source_token(_SKILL_IDS[skill]), *args, SEP)
        )
        outputs.append(result)

    stream.extend((QUERY, *(entity_token(stage_id) for stage_id in stage_ids), SEP))
    answer = tuple(scalar_token(value) for output in outputs for value in output) + (EOS,)
    output_keys: list[int] = []
    operation_output_keys: list[int] = []
    for stage_id, skill, stage_output in zip(stage_ids, skills, outputs):
        if skill == "sort":
            output_keys.extend((-1, -1))
            operation_output_keys.extend((stage_id, stage_id + 1))
        else:
            output_keys.append(stage_id)
            operation_output_keys.append(stage_id)
    output_keys.append(-1)
    operation_output_keys.append(-1)
    return TaskEpisode(
        "mixed_skill",
        tuple(stream),
        answer,
        common_metadata(
            track="mixed_skill",
            sequence_length=len(stream),
            query_type=f"dependent_ops{operation_count}_" + ";".join(skills),
            relevant_updates=operation_count,
            distractor_count=keys,
            source_count=len(_SKILL_IDS),
            candidate_baseline_pct=100.0 / (SCALAR_COUNT ** max(1, len(answer) - 1)),
            operation_count=operation_count,
            skill_sequence=tuple(skills),
            output_count=len(answer) - 1,
            dependency_depth=max(0, operation_count - 1),
            outputs_are_not_in_input=True,
            ood=bool(ood),
            progression_group=f"dependent_ops_{operation_count}",
            workspace_output_keys=tuple(output_keys),
            workspace_operation_output_keys=tuple(operation_output_keys),
        ),
    )
