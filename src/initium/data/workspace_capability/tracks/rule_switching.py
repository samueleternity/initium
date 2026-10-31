"""Feedback-inferred and cued task-set changes inspired by card sorting."""

from __future__ import annotations

import random

from initium.data.workspace_capability.episode import (
    BOS,
    BOOL_FALSE,
    BOOL_TRUE,
    EOS,
    EV_ASSERT,
    EV_UPDATE,
    EV_VECTOR,
    LOCATION_COUNT,
    QUERY,
    RELATION_COUNT,
    SCALAR_COUNT,
    SEP,
    TaskEpisode,
    entity_token,
    location_token,
    relation_token,
    scalar_token,
    source_token,
)
from initium.data.workspace_capability.tracks._shared import common_metadata


def _card(rng: random.Random):
    # Attribute token ranges are disjoint, making the response dimension
    # recoverable without leaking which dimension is currently correct.
    return (
        rng.randrange(min(4, LOCATION_COUNT)),
        rng.randrange(min(4, RELATION_COUNT)),
        rng.randrange(min(4, SCALAR_COUNT)),
    )


def _card_tokens(card):
    return (location_token(card[0]), relation_token(card[1]), scalar_token(card[2]))


def generate(difficulty: dict, rng: random.Random, *, ood: bool = False) -> TaskEpisode:
    evidence_trials = min(max(2, int(difficulty["evidence_trials"])), 9 if ood else 8)
    switch = bool(difficulty.get("switch", True)) and evidence_trials >= 3
    switch_at = rng.randrange(1, evidence_trials - 1) if switch else evidence_trials
    initial_rule = rng.randrange(3)
    current_rule = initial_rule
    cue = switch and bool(rng.getrandbits(1))
    stream = [BOS]

    for trial in range(evidence_trials):
        if switch and trial == switch_at:
            current_rule = (initial_rule + rng.randrange(1, 3)) % 3
            if cue:
                stream.extend((EV_UPDATE, source_token(current_rule), SEP))
        card = _card(rng)
        stream.extend((EV_VECTOR, entity_token(0), *_card_tokens(card), SEP))
        # A simulated sorter makes a feature-value choice; feedback is exact
        # and depends on the currently active hidden rule.
        if rng.random() < 0.75:
            choice_dimension = rng.randrange(3)
            choice = _card_tokens(card)[choice_dimension]
        else:
            choice_dimension = rng.randrange(3)
            choice = (location_token, relation_token, scalar_token)[choice_dimension](
                (card[choice_dimension] + rng.randrange(1, 4)) % 4
            )
        correct = choice == _card_tokens(card)[current_rule]
        stream.extend(
            (EV_ASSERT, entity_token(0), choice, BOOL_TRUE if correct else BOOL_FALSE, SEP)
        )

    final_card = _card(rng)
    stream.extend((EV_VECTOR, entity_token(0), *_card_tokens(final_card), QUERY, SEP))
    answer = _card_tokens(final_card)[current_rule]
    return TaskEpisode(
        "rule_switching",
        tuple(stream),
        (answer, EOS),
        common_metadata(
            track="rule_switching",
            sequence_length=len(stream),
            query_type=(
                f"{'cued_switch' if cue else 'feedback_inferred_switch' if switch else 'stable_rule'}"
                f"_trials{evidence_trials}"
            ),
            relevant_updates=evidence_trials,
            distractor_count=evidence_trials - 1,
            source_count=3,
            candidate_baseline_pct=100.0 / 12.0,
            evidence_trials=evidence_trials,
            switched=switch,
            switch_at=switch_at,
            cue_provided=cue,
            initial_rule=initial_rule,
            final_rule=current_rule,
            ood=bool(ood),
        ),
    )
