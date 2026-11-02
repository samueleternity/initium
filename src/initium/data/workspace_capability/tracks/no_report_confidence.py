"""Indirect category selection with an optional confidence-bin probe."""

from __future__ import annotations

import random

from initium.data.workspace_capability.episode import (
    BOS, BOOL_FALSE, BOOL_TRUE, EOS, EV_INIT, EV_UPDATE, QUERY, SEP,
    TaskEpisode, entity_token, scalar_token,
)
from initium.data.workspace_capability.tracks._shared import common_metadata


def generate(difficulty: dict, rng: random.Random, *, ood: bool = False) -> TaskEpisode:
    evidence = rng.randint(3, min(9, int(difficulty.get("maximum_evidence", 6)) + int(ood)))
    truth = bool(rng.getrandbits(1))
    cue_error_rate = rng.choice((0.1, 0.25, 0.4))
    observations = [truth ^ (rng.random() < cue_error_rate) for _ in range(evidence)]
    positive_count = sum(observations)
    action = positive_count * 2 >= evidence
    margin = abs(2 * positive_count - evidence)
    report_condition = bool(rng.getrandbits(1))
    confidence = min(3, margin * 4 // evidence)
    stream = [BOS, EV_INIT, entity_token(0), scalar_token(evidence), SEP]
    for index, observation in enumerate(observations):
        stream.extend((EV_UPDATE, entity_token(index + 1), scalar_token(int(observation)), SEP))
    stream.extend((QUERY, *(entity_token(index + 1) for index in range(evidence)), SEP))
    answer = (BOOL_TRUE if truth else BOOL_FALSE,)
    if report_condition:
        answer += (scalar_token(confidence),)
    answer += (EOS,)
    return TaskEpisode(
        "no_report_confidence", tuple(stream), answer,
        common_metadata(
            track="no_report_confidence", sequence_length=len(stream),
            query_type="confidence_probe" if report_condition else "no_report_action",
            relevant_updates=evidence, candidate_baseline_pct=50.0, evidence_strength=evidence,
            confidence_bin=confidence, action=truth, majority_action=action,
            action_correct=action == truth, cue_error_rate=cue_error_rate,
            vote_margin=margin, explicit_report=report_condition,
            progression_group="all",
        ),
    )
