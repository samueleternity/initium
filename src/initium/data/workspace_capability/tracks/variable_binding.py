"""Reassignment chains with copy-by-value and computed-value queries."""

from __future__ import annotations

import random

from initium.data.workspace_capability.episode import (
    BOS,
    EOS,
    EV_INIT,
    EV_KV,
    EV_UPDATE,
    EV_VECTOR,
    QUERY,
    SCALAR_COUNT,
    SEP,
    TaskEpisode,
    entity_token,
    scalar_token,
    source_token,
)
from initium.data.workspace_capability.tracks._shared import (
    EXT_OP_NAMES, EXT_UNARY, common_metadata, ext_op_result,
)

_ASSIGN_ADD = 4


def _initial_plan(
    rng: random.Random, variable_count: int
) -> tuple[dict[int, int], int, int, int, int]:
    """Choose a source/update chain whose computed answer is not a literal."""
    for _ in range(128):
        values = {variable: rng.randrange(4, 8) for variable in range(variable_count)}
        first_update = rng.randrange(8, 12)
        second_update = rng.randrange(12, 16)
        for third_update in rng.sample(range(SCALAR_COUNT), SCALAR_COUNT):
            result = (values[0] + third_update) % SCALAR_COUNT
            if result not in {
                *values.values(), first_update, second_update, third_update
            }:
                return values, first_update, second_update, third_update, result

    # 7 + 4 = 11, absent from the 6/7 initialization values and the updates.
    return {variable: 7 for variable in range(variable_count)}, 8, 12, 4, 11


def generate(difficulty: dict, rng: random.Random, *, ood: bool = False) -> TaskEpisode:
    variable_count = min(max(5, int(difficulty["variables"])), 12)
    assignment_count = min(
        max(7, int(difficulty["assignments"])), 24 if ood else 21
    )
    variable_ids = list(range(variable_count))
    values, first_update, second_update, third_update, planned_result = _initial_plan(
        rng, variable_count
    )
    query_variable = variable_ids[-1]
    stream = [BOS]
    literal_values = set(values.values())
    for variable in variable_ids:
        stream.extend((EV_INIT, entity_token(variable), scalar_token(values[variable]), SEP))

    # Three copies propagate a value through variables whose sources are each
    # reassigned. The query combines the preserved value with a later literal.
    # The final modular sum is selected to be absent from every
    # literal value in the stream, so a lookup of one earlier scalar cannot
    # solve the episode.
    copy_source, first_copy, second_copy, third_copy = variable_ids[:4]
    values[first_copy] = values[copy_source]
    stream.extend((EV_KV, entity_token(first_copy), entity_token(copy_source), SEP))

    def fresh_literal(exclude: set[int]) -> int:
        candidates = [value for value in range(SCALAR_COUNT) if value not in exclude]
        return rng.choice(candidates) if candidates else rng.choice(
            [value for value in range(SCALAR_COUNT) if value != planned_result]
        )

    literal_values.add(first_update)
    values[copy_source] = first_update
    stream.extend((EV_UPDATE, entity_token(copy_source), scalar_token(first_update), SEP))

    values[second_copy] = values[first_copy]
    stream.extend((EV_KV, entity_token(second_copy), entity_token(first_copy), SEP))
    literal_values.add(second_update)
    values[first_copy] = second_update
    stream.extend((EV_UPDATE, entity_token(first_copy), scalar_token(second_update), SEP))

    values[third_copy] = values[second_copy]
    stream.extend((EV_KV, entity_token(third_copy), entity_token(second_copy), SEP))
    literal_values.add(third_update)
    values[second_copy] = third_update
    stream.extend((EV_UPDATE, entity_token(second_copy), scalar_token(third_update), SEP))

    result = (values[third_copy] + values[second_copy]) % SCALAR_COUNT
    if result != planned_result:
        raise ValueError("generated variable-binding chain disagrees with its planned result")

    stream.extend(
        (
            EV_VECTOR,
            entity_token(query_variable),
            source_token(_ASSIGN_ADD),
            entity_token(third_copy),
            entity_token(second_copy),
            SEP,
        )
    )
    values[query_variable] = result
    operations = 7
    copies = 3
    reassignments = 3

    while operations < assignment_count:
        destination = rng.choice(variable_ids[:-1])
        action = rng.randrange(5 if difficulty.get("extended_ops") else 3)
        if action == 0:
            source = rng.choice([variable for variable in variable_ids if variable != destination])
            values[destination] = values[source]
            stream.extend((EV_KV, entity_token(destination), entity_token(source), SEP))
            copies += 1
        elif action == 1:
            value = fresh_literal(literal_values | {result})
            literal_values.add(value)
            values[destination] = value
            stream.extend((EV_UPDATE, entity_token(destination), scalar_token(value), SEP))
            reassignments += 1
        elif action >= 3:
            name = rng.choice([n for n in EXT_OP_NAMES if (n in EXT_UNARY) == (action == 4)])
            i = EXT_OP_NAMES.index(name)
            srcs = [v for v in variable_ids if v != destination]
            left, right = rng.sample(srcs, 2) if action == 3 else (rng.choice(srcs), None)
            values[destination] = ext_op_result(
                name, values[left], values[right] if right is not None else 0
            )
            stream.extend(
                (EV_VECTOR, entity_token(destination), source_token(i // 4), source_token(i % 4),
                 entity_token(left), *((entity_token(right),) if right is not None else ()), SEP)
            )
            reassignments += 1
        else:
            left, right = rng.sample(
                [variable for variable in variable_ids if variable != destination], 2
            )
            values[destination] = (values[left] + values[right]) % SCALAR_COUNT
            stream.extend(
                (
                    EV_VECTOR,
                    entity_token(destination),
                    source_token(_ASSIGN_ADD),
                    entity_token(left),
                    entity_token(right),
                    SEP,
                )
            )
            reassignments += 1
        operations += 1

    stream.extend((QUERY, entity_token(query_variable), SEP))
    return TaskEpisode(
        "variable_binding",
        tuple(stream),
        (scalar_token(values[query_variable]), EOS),
        common_metadata(
            track="variable_binding",
            sequence_length=len(stream),
            query_type=f"computed_chain_v{variable_count}_a{assignment_count}",
            relevant_updates=assignment_count,
            distractor_count=assignment_count - 5,
            candidate_baseline_pct=100.0 / SCALAR_COUNT,
            variable_count=variable_count,
            assignment_count=assignment_count,
            copy_count=copies,
            reassignment_count=reassignments,
            chain_hops=4,
            query_variable=query_variable,
            workspace_output_keys=(query_variable, -1),
            answer_is_literal=False,
            ood=bool(ood),
            progression_group=f"computed_chain_{'long' if assignment_count >= 12 else 'short'}",
        ),
    )
