"""Interleaved facts and multi-hop cross-source relational joins."""

from __future__ import annotations

import random

from initium.data.workspace_capability.episode import (
    BOS,
    ENTITY_COUNT,
    EV_FACT,
    EOS,
    QUERY,
    RELATION_COUNT,
    SEP,
    SOURCE_COUNT,
    TaskEpisode,
    entity_token,
    relation_token,
    source_token,
)


def generate(difficulty: dict, rng: random.Random, *, ood: bool = False) -> TaskEpisode:
    hops = int(difficulty["hops"])
    distractors = int(difficulty.get("distractors", 0))
    if not 1 <= hops <= 3:
        raise ValueError("relational join hop count must be in [1,3]")
    entity_ids = list(range(ENTITY_COUNT - hops - 1, ENTITY_COUNT)) if ood else list(
        range(hops + 1)
    )
    relations = list(range(RELATION_COUNT - hops, RELATION_COUNT)) if ood else list(
        range(hops)
    )
    sources = list(range(SOURCE_COUNT))
    chain = [(entity_ids[i], relations[i], entity_ids[i + 1]) for i in range(hops)]
    facts = [(rng.randrange(SOURCE_COUNT), *triple) for triple in chain]
    decoy_ids = [entity for entity in range(ENTITY_COUNT) if entity not in entity_ids]
    rng.shuffle(decoy_ids)
    irrelevant_relations = [relation for relation in range(RELATION_COUNT) if relation not in relations]
    for index in range(min(distractors, len(decoy_ids))):
        # One disconnected, near-miss fact per decoy prevents accidental
        # alternative paths from making the oracle answer non-unique.
        subject = entity_ids[index % len(entity_ids)]
        relation = irrelevant_relations[index % len(irrelevant_relations)]
        obj = decoy_ids[index]
        if index % 2:
            subject, obj = obj, subject
        facts.append((rng.randrange(SOURCE_COUNT), subject, relation, obj))
    rng.shuffle(facts)
    stream = [BOS]
    for src, subject, relation, obj in facts:
        stream.extend(
            (EV_FACT, source_token(src), entity_token(subject), relation_token(relation),
             entity_token(obj), SEP)
        )
    stream.extend(
        (QUERY, entity_token(entity_ids[0]), *(relation_token(r) for r in relations), SEP)
    )
    return TaskEpisode(
        "relational_join",
        tuple(stream),
        (entity_token(entity_ids[-1]), EOS),
        {
            "query_type": f"{hops}_hop_join",
            "sequence_length": len(stream),
            "relevant_updates": hops,
            "distractor_count": min(distractors, len(decoy_ids)),
            "entity_count": len(entity_ids),
            "hop_count": hops,
            "source_count": len(set(f[0] for f in facts)),
            "correction_relevant": False,
            "novelty_distractor": min(distractors, len(decoy_ids)) > 0,
            "answer_count": 1,
            "candidate_count": len(set(entity_ids) | set(decoy_ids[:min(distractors, len(decoy_ids))])),
            "candidate_baseline_pct": 100.0 / max(
                1, len(set(entity_ids) | set(decoy_ids[:min(distractors, len(decoy_ids))]))
            ),
            "difficulty": hops + distractors,
        },
    )
