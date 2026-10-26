"""Multi-source entity state with source priority, corrections, and retractions."""

from __future__ import annotations

import random

from initium.data.workspace_capability.episode import (
    BOS,
    ENTITY_COUNT,
    EV_ASSERT,
    EV_MOVE,
    EV_PUT,
    EV_REMOVE,
    EV_RETRACT,
    EV_SUPERSEDE,
    EOS,
    FACT_COUNT,
    LOCATION_COUNT,
    NONE_TOKEN,
    QUERY,
    RELATION_COUNT,
    SEP,
    SOURCE_COUNT,
    SOURCE_PRIORITY_POLICY,
    TaskEpisode,
    fact_token as encode_fact_token,
    entity_token,
    location_token,
    relation_token,
    source_token,
)

Fact = tuple[int, int, int, int]  # fact-id, source-id, value, logical timestamp


def fact_token(fact_id: int) -> int:
    return encode_fact_token(int(fact_id))


def _winner(facts: list[Fact]) -> Fact:
    """Highest source priority, then latest time, then lowest source ID wins."""
    if not facts:
        raise ValueError("cannot select a winner from an empty fact set")
    return max(facts, key=lambda fact: (fact[1] % 3, fact[3], -fact[1]))


def generate(difficulty: dict, rng: random.Random, *, ood: bool = False) -> TaskEpisode:
    n_entities = min(int(difficulty["entities"]), ENTITY_COUNT)
    n_sources = min(int(difficulty.get("sources", 2)), SOURCE_COUNT)
    updates = int(difficulty["updates"])
    max_noise = int(difficulty.get("distractors", 0))
    correction_relevant = (
        bool(difficulty.get("correction", True))
        and updates >= 1
        and rng.random() < float(difficulty.get("correction_probability", 0.5))
    )
    novelty_distractor = (
        bool(difficulty.get("novelty", True))
        and max_noise >= 1
        and rng.random() < float(difficulty.get("novelty_probability", 0.5))
    )
    noise = max_noise if novelty_distractor else 0
    entity_ids = list(range(ENTITY_COUNT - n_entities, ENTITY_COUNT)) if ood else list(
        range(n_entities)
    )
    sources = list(range(SOURCE_COUNT)) if ood else list(range(n_sources))
    if ood:
        rng.shuffle(sources)
        sources = sources[:n_sources]
    query_entity = entity_ids[0]
    if ood:
        target_relation = RELATION_COUNT // 2 + rng.randrange(RELATION_COUNT // 2)
    else:
        target_relation = rng.randrange(RELATION_COUNT // 2)

    stream = [BOS]
    active: dict[int, list[Fact]] = {entity: [] for entity in entity_ids}
    used_fact_ids: set[int] = set()
    query_candidates: set[int] = set()
    next_fact_id = 0
    timestamp = 0

    def allocate_fact_id():
        nonlocal next_fact_id
        for offset in range(FACT_COUNT):
            candidate = (next_fact_id + offset) % FACT_COUNT
            if candidate not in used_fact_ids:
                used_fact_ids.add(candidate)
                next_fact_id = (candidate + 1) % FACT_COUNT
                return candidate
        raise RuntimeError("generated entity-state episode exhausted fact identifiers")

    def add_put(
        entity: int,
        source: int,
        value: int,
        *,
        fact_id=None,
        logical_time=None,
        event_type=EV_PUT,
    ):
        nonlocal timestamp
        if fact_id is None:
            fact_id = allocate_fact_id()
        if logical_time is None:
            timestamp += 1
            logical_time = timestamp
        active[entity].append((fact_id, source, value, logical_time))
        stream.extend(
            (
                event_type,
                source_token(source),
                fact_token(fact_id),
                entity_token(entity),
                relation_token(target_relation),
                location_token(value),
                SEP,
            )
        )
        if entity == query_entity:
            query_candidates.add(value)
        return fact_id, logical_time

    # Every entity begins with one source assertion. The queried entity gets
    # a second source assertion so priority conflicts occur before any updates.
    for entity in entity_ids:
        add_put(entity, rng.choice(sources), rng.randrange(LOCATION_COUNT))
    equal_priority_conflict = False
    if n_sources >= 2:
        query_facts = active[query_entity]
        existing_source = query_facts[0][1]
        alternatives = [source for source in sources if source != existing_source]
        tie_pairs = [
            source for source in alternatives
            if source % 3 == existing_source % 3
        ]
        if tie_pairs and rng.random() < 0.5:
            second_source = rng.choice(tie_pairs)
            # Same logical timestamp deliberately exercises the source-ID tie break.
            tie_time = query_facts[0][3]
            equal_priority_conflict = True
            add_put(
                query_entity,
                second_source,
                rng.randrange(LOCATION_COUNT),
                logical_time=tie_time,
            )
        else:
            add_put(
                query_entity,
                rng.choice(alternatives),
                rng.randrange(LOCATION_COUNT),
            )

    relevant_updates = 0
    query_change_kind = None
    for index in range(updates):
        entity = query_entity if index == 0 and correction_relevant else rng.choice(entity_ids)
        current = active[entity]
        if entity == query_entity and correction_relevant:
            # Correct the currently authoritative source. Retracting it exposes
            # the next-best source, then SUPERSEDE installs a newer value.
            old_fact, old_source, old_value, _old_time = _winner(current)
            replacement = (old_value + rng.randint(1, LOCATION_COUNT - 1)) % LOCATION_COUNT
            query_candidates.add(replacement)
            new_fact = allocate_fact_id()
            stream.extend((EV_RETRACT, source_token(old_source), fact_token(old_fact), SEP))
            current.remove((old_fact, old_source, old_value, _old_time))
            timestamp += 1
            stream.extend(
                (
                    EV_SUPERSEDE,
                    source_token(old_source),
                    fact_token(old_fact),
                    fact_token(new_fact),
                    entity_token(entity),
                    relation_token(target_relation),
                    location_token(replacement),
                    SEP,
                )
            )
            current.append((new_fact, old_source, replacement, timestamp))
            relevant_updates += 1
            query_change_kind = "supersede"
            continue

        source = rng.choice(sources)
        source_fact = next((fact for fact in current if fact[1] == source), None)
        replacement_value = rng.randrange(LOCATION_COUNT)
        if entity == query_entity:
            query_candidates.add(replacement_value)
        replacement_fact = allocate_fact_id()
        if source_fact is not None and rng.random() < 0.4:
            old_fact, old_source, old_value, _old_time = source_fact
            current.remove(source_fact)
            timestamp += 1
            stream.extend(
                (
                    EV_MOVE,
                    source_token(old_source),
                    fact_token(old_fact),
                    entity_token(entity),
                    relation_token(target_relation),
                    location_token(old_value),
                    source_token(source),
                    fact_token(replacement_fact),
                    location_token(replacement_value),
                    SEP,
                )
            )
            current.append((replacement_fact, source, replacement_value, timestamp))
        else:
            if source_fact is not None:
                old_fact = source_fact[0]
                current.remove(source_fact)
                stream.extend((EV_RETRACT, source_token(source), fact_token(old_fact), SEP))
            add_put(
                entity,
                source,
                replacement_value,
                fact_id=replacement_fact,
                event_type=EV_ASSERT,
            )
        relevant_updates += int(entity == query_entity)
        query_change_kind = "move" if source_fact is not None else "assert"

    for _ in range(noise):
        # High-surprise distractors use unrelated relations and cannot change the
        # queried relation's source-priority resolution.
        candidates = [entity for entity in entity_ids if entity != query_entity]
        if not candidates:
            break
        entity = rng.choice(candidates)
        source = rng.choice(sources)
        relation = (target_relation + rng.randint(1, RELATION_COUNT - 1)) % RELATION_COUNT
        value = rng.randrange(LOCATION_COUNT)
        distractor_fact_id = allocate_fact_id()
        stream.extend(
            (
                EV_ASSERT,
                source_token(source),
                fact_token(distractor_fact_id),
                entity_token(entity),
                relation_token(relation),
                location_token(value),
                SEP,
            )
        )

    valid = active[query_entity]
    if valid and not correction_relevant and updates > 0 and rng.random() < 0.2:
        old_fact, old_source, old_value, _old_time = _winner(valid)
        valid.remove((old_fact, old_source, old_value, _old_time))
        stream.extend(
            (
                EV_REMOVE,
                source_token(old_source),
                fact_token(old_fact),
                entity_token(query_entity),
                relation_token(target_relation),
                location_token(old_value),
                SEP,
            )
        )
        relevant_updates += 1
        query_change_kind = "remove"

    if valid:
        _fact, _source, answer_value, _time = _winner(valid)
        answer = (location_token(answer_value), EOS)
    else:
        answer = (NONE_TOKEN, EOS)
    stream.extend(
        (QUERY, SOURCE_PRIORITY_POLICY, entity_token(query_entity), relation_token(target_relation), SEP)
    )
    return TaskEpisode(
        "entity_state",
        tuple(stream),
        answer,
        {
            "query_type": "priority_then_latest_entity_value",
            "sequence_length": len(stream),
            "relevant_updates": relevant_updates,
            "distractor_count": noise,
            "entity_count": n_entities,
            "hop_count": 1,
            "source_count": n_sources,
            "correction_relevant": correction_relevant and relevant_updates > 0,
            "novelty_distractor": novelty_distractor,
            "correction_kind": query_change_kind or "none",
            "answer_count": 1,
            "candidate_count": max(1, len(query_candidates)),
            "candidate_baseline_pct": 100.0 / max(1, len(query_candidates)),
            "priority_rule": "source_mod3_then_latest_then_low_source_id",
            "equal_priority_conflict": equal_priority_conflict,
            "difficulty": updates,
        },
    )
