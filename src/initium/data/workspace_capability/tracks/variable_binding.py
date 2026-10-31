"""Online assignment evaluation with copy-by-value and reassignment semantics."""

from __future__ import annotations

import random

from initium.data.workspace_capability.episode import (
    BOS,
    EOS,
    EV_INIT,
    EV_KV,
    EV_UPDATE,
    QUERY,
    SCALAR_COUNT,
    SEP,
    TaskEpisode,
    entity_token,
    scalar_token,
)
from initium.data.workspace_capability.tracks._shared import common_metadata


def generate(difficulty: dict, rng: random.Random, *, ood: bool = False) -> TaskEpisode:
    variable_count = min(max(2, int(difficulty["variables"])), 12)
    assignment_count = min(
        max(variable_count + 2, int(difficulty["assignments"])), 24
    )
    variable_ids = list(range(variable_count))
    values = {variable: rng.randrange(SCALAR_COUNT) for variable in variable_ids}
    stream = [BOS]
    for variable in variable_ids:
        stream.extend((EV_INIT, entity_token(variable), scalar_token(values[variable]), SEP))

    # Guarantee the defining binding behavior even at the first curriculum
    # level: copy a value, then reassign its source, and ask for the copy.
    copy_source, copy_destination = variable_ids[:2]
    values[copy_destination] = values[copy_source]
    stream.extend(
        (EV_KV, entity_token(copy_destination), entity_token(copy_source), SEP)
    )
    copied_value = values[copy_destination]
    reassigned_value = (
        values[copy_source] + rng.randrange(1, SCALAR_COUNT)
    ) % SCALAR_COUNT
    values[copy_source] = reassigned_value
    source_value_after_reassignment = reassigned_value
    stream.extend(
        (EV_UPDATE, entity_token(copy_source), scalar_token(reassigned_value), SEP)
    )
    copies = 1
    reassignments = 1
    for step in range(assignment_count - variable_count - 2):
        destination = rng.choice(variable_ids)
        if destination == copy_destination:
            destination = copy_source
        if step % 3 == 1 and variable_count > 1:
            source = rng.choice([item for item in variable_ids if item != destination])
            # y=x captures x's value now; later writes to x do not alter y.
            values[destination] = values[source]
            stream.extend((EV_KV, entity_token(destination), entity_token(source), SEP))
            copies += 1
        else:
            new_value = rng.randrange(SCALAR_COUNT)
            if new_value == values[destination]:
                new_value = (new_value + rng.randrange(1, SCALAR_COUNT)) % SCALAR_COUNT
            values[destination] = new_value
            stream.extend(
                (EV_UPDATE, entity_token(destination), scalar_token(new_value), SEP)
            )
            reassignments += 1

    query_variable = copy_destination
    stream.extend((QUERY, entity_token(query_variable), SEP))
    return TaskEpisode(
        "variable_binding",
        tuple(stream),
        (scalar_token(values[query_variable]), EOS),
        common_metadata(
            track="variable_binding",
            sequence_length=len(stream),
            query_type=f"final_value_v{variable_count}_a{assignment_count}",
            relevant_updates=assignment_count - variable_count,
            distractor_count=0,
            candidate_baseline_pct=100.0 / SCALAR_COUNT,
            variable_count=variable_count,
            assignment_count=assignment_count,
            copy_count=copies,
            reassignment_count=reassignments,
            copy_source=copy_source,
            copy_destination=copy_destination,
            copied_value=copied_value,
            source_value_after_reassignment=source_value_after_reassignment,
            query_variable=query_variable,
            ood=bool(ood),
        ),
    )
