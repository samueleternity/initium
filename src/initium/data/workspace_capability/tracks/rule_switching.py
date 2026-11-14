"""Noisy feedback-based task-set inference with repeated rule changes."""

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
    QUERY,
    SCALAR_COUNT,
    SEP,
    TaskEpisode,
    entity_token,
    scalar_token,
    source_token,
)
from initium.data.workspace_capability.tracks._shared import common_metadata


def _card(rng: random.Random) -> tuple[int, int, int]:
    # All dimensions share one value vocabulary; feature position, not token
    # family, defines the candidate rules.
    return tuple(rng.randrange(SCALAR_COUNT) for _ in range(3))


def generate(difficulty: dict, rng: random.Random, *, ood: bool = False) -> TaskEpisode:
    evidence_trials = min(max(4, int(difficulty["evidence_trials"])), 16 if ood else 14)
    maximum_switches = max(0, (evidence_trials // 2) - 1)
    switch_count = min(max(0, int(difficulty.get("switches", 0))), maximum_switches)
    feedback_noise = min(max(float(difficulty.get("feedback_noise", 0.2)), 0.05), 0.35)
    initial_rule = rng.randrange(3)

    segment_count = switch_count + 1
    base_length, remainder = divmod(evidence_trials, segment_count)
    segment_lengths = [base_length + int(index < remainder) for index in range(segment_count)]
    boundaries = []
    running = 0
    for length in segment_lengths[:-1]:
        running += length
        boundaries.append(running)

    rules = [initial_rule]
    for _ in range(switch_count):
        previous_rule = rules[-1]
        rules.append((previous_rule + rng.randrange(1, 3)) % 3)
    switch_cues = [bool(rng.getrandbits(1)) for _ in range(switch_count)]
    cue_by_boundary = dict(zip(boundaries, switch_cues))

    stream = [BOS, EV_UPDATE, source_token(initial_rule), SEP]
    trial_rules = []
    for trial in range(evidence_trials):
        segment = sum(trial >= boundary for boundary in boundaries)
        current_rule = rules[segment]
        if trial in cue_by_boundary and cue_by_boundary[trial]:
            stream.extend((EV_UPDATE, source_token(current_rule), SEP))

        card = _card(rng)
        stream.extend((EV_VECTOR, entity_token(0), *(scalar_token(value) for value in card), SEP))

        # The chosen feature is observable, but feedback is noisy. A single
        # success/failure therefore cannot identify the active rule reliably.
        choice_dimension = rng.randrange(3)
        feedback_correct = rng.random() >= feedback_noise if choice_dimension == current_rule else (
            rng.random() < feedback_noise
        )
        stream.extend(
            (
                EV_ASSERT,
                entity_token(0),
                source_token(choice_dimension),
                scalar_token(card[choice_dimension]),
                BOOL_TRUE if feedback_correct else BOOL_FALSE,
                SEP,
            )
        )
        trial_rules.append(current_rule)

    final_rule = rules[-1]
    final_card = _card(rng)
    stream.extend(
        (
            EV_VECTOR,
            entity_token(0),
            *(scalar_token(value) for value in final_card),
            QUERY,
            SEP,
        )
    )
    cue_count = sum(switch_cues)
    if not switch_count:
        progression_group = "stable_rule"
    elif cue_count == switch_count:
        progression_group = "cued_switch"
    elif cue_count == 0:
        progression_group = "feedback_inferred_switch"
    else:
        progression_group = "mixed_cue_switch"

    return TaskEpisode(
        "rule_switching",
        tuple(stream),
        (scalar_token(final_card[final_rule]), EOS),
        common_metadata(
            track="rule_switching",
            sequence_length=len(stream),
            query_type=(
                f"{progression_group}_trials{evidence_trials}_switches{switch_count}"
            ),
            relevant_updates=evidence_trials,
            distractor_count=evidence_trials - switch_count,
            source_count=3,
            candidate_baseline_pct=100.0 / SCALAR_COUNT,
            evidence_trials=evidence_trials,
            switch_count=switch_count,
            switch_boundaries=tuple(boundaries),
            cue_count=cue_count,
            feedback_noise=feedback_noise,
            initial_rule=initial_rule,
            final_rule=final_rule,
            trial_rules=tuple(trial_rules),
            workspace_query_reads=((len(stream) - 1, 63, source_token(final_rule)),),
            workspace_output_programs=(
                (
                    0,
                    "select",
                    (63, 0, 1, 2),
                    (source_token(final_rule), *(scalar_token(value) for value in final_card)),
                ),
            ),
            ood=bool(ood),
            progression_group=progression_group,
        ),
    )
