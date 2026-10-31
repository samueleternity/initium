"""Multi-agent belief updates with world-state and partial-observation controls."""

from __future__ import annotations

import random

from initium.data.workspace_capability.episode import (
    BOS,
    EOS,
    EV_ASSERT,
    EV_INIT,
    EV_MOVE,
    LOCATION_COUNT,
    QUERY,
    SEP,
    TaskEpisode,
    entity_token,
    location_token,
    source_token,
)
from initium.data.workspace_capability.tracks._shared import common_metadata


def generate(difficulty: dict, rng: random.Random, *, ood: bool = False) -> TaskEpisode:
    agents = min(max(2, int(difficulty["agents"])), 4)
    objects = min(max(1, int(difficulty["objects"])), 8)
    moves = min(max(1, int(difficulty["moves"])), 10)
    object_ids = list(range(objects))
    world = {obj: rng.randrange(LOCATION_COUNT) for obj in object_ids}
    beliefs = {(agent, obj): world[obj] for agent in range(agents) for obj in object_ids}
    stream = [BOS]
    query_condition = rng.choice(("false_belief", "true_belief", "reality"))
    query_agent = rng.randrange(agents)
    query_object = rng.choice(object_ids)

    # Everyone observes the initial arrangement. Subsequent moves update only
    # those agents who observe that move, leaving other beliefs stale.
    for object_id in object_ids:
        stream.extend(
            (EV_INIT, entity_token(object_id), location_token(world[object_id]), SEP)
        )
        for agent_id in range(agents):
            stream.extend(
                (
                    EV_ASSERT,
                    source_token(agent_id),
                    entity_token(object_id),
                    location_token(world[object_id]),
                    SEP,
                )
            )

    observation_count = 0
    query_observed_updates = 0
    for move_index in range(moves):
        moved_object = (
            query_object
            if (query_condition != "reality" and move_index == moves - 1)
            else rng.choice(object_ids)
        )
        old = world[moved_object]
        new = (old + rng.randrange(1, LOCATION_COUNT)) % LOCATION_COUNT
        world[moved_object] = new
        stream.extend((EV_MOVE, entity_token(moved_object), location_token(new), SEP))
        observers = [agent for agent in range(agents) if rng.random() < 0.5]
        # Keep some moves unobserved and some multiply observed so a single
        # global state cannot answer all belief queries.
        if not observers:
            observers = [rng.randrange(agents)] if rng.random() < 0.35 else []
        if query_condition == "false_belief" and move_index == moves - 1:
            observers = [observer for observer in observers if observer != query_agent]
        elif query_condition == "true_belief" and move_index == moves - 1:
            observers = sorted(set((*observers, query_agent)))
        if moved_object == query_object and query_agent in observers:
            query_observed_updates += 1
        for observer in observers:
            beliefs[(observer, moved_object)] = new
            stream.extend(
                (
                    EV_ASSERT,
                    source_token(observer),
                    entity_token(moved_object),
                    location_token(new),
                    SEP,
                )
            )
            observation_count += 1

    if query_condition == "reality":
        stream.extend((QUERY, entity_token(query_object), SEP))
    else:
        stream.extend((QUERY, source_token(query_agent), entity_token(query_object), SEP))
    answer_value = (
        world[query_object]
        if query_condition == "reality"
        else beliefs[(query_agent, query_object)]
    )
    relevant_updates = moves if query_condition == "reality" else query_observed_updates
    return TaskEpisode(
        "belief_tracking",
        tuple(stream),
        (location_token(answer_value), EOS),
        common_metadata(
            track="belief_tracking",
            sequence_length=len(stream),
            query_type=f"{query_condition}_agents{agents}_objects{objects}",
            relevant_updates=relevant_updates,
            distractor_count=moves - relevant_updates,
            source_count=agents,
            candidate_baseline_pct=100.0 / LOCATION_COUNT,
            agent_count=agents,
            object_count=objects,
            world_move_count=moves,
            observation_count=observation_count,
            false_belief=(
                query_condition == "false_belief" and answer_value != world[query_object]
            ),
            true_belief=(
                query_condition == "true_belief" and answer_value == world[query_object]
            ),
            ood=bool(ood),
        ),
    )
