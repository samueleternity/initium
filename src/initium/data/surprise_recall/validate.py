"""Run deterministic generator-level checks for the Phase 0 SGR task.

Run with ``python -m initium.data.surprise_recall.validate`` from the repo root.
The report describes task validity only; it does not train or modify a model.
"""

from __future__ import annotations

import random

from initium.data.surprise_recall.surprise_recall import (
    ALERT_TOKENS,
    LESSONS,
    VOCAB_SIZE,
    build_episode,
    oracle_answer,
    verify_labels,
)


def run_validity_gate(episodes_per_lesson: int = 1000, seed: int = 20261018):
    rng = random.Random(seed)
    checked = {
        idx: [build_episode(idx, rng) for _ in range(episodes_per_lesson)]
        for idx in range(len(LESSONS))
    }
    labels_valid = all(verify_labels(ep) for episodes in checked.values() for ep in episodes)

    a2 = [ep for ep in checked[2] if "A2" in ep.anomaly_types]
    a2_flags = [
        token in ALERT_TOKENS
        for ep in a2
        for token, kind in zip(ep.observed_tokens, ep.anomaly_types)
        if kind == "A2"
    ]
    a3 = [ep for ep in checked[3] if "A3" in ep.anomaly_types]
    a3_flags = [
        token in ALERT_TOKENS
        for ep in a3
        for token, kind in zip(ep.observed_tokens, ep.anomaly_types)
        if kind == "A3"
    ]
    content_miss_rate = 1.0 - sum(a2_flags) / max(len(a2_flags), 1)
    content_false_alarm_rate = sum(a3_flags) / max(len(a3_flags), 1)

    # A stateless uniform guess has per-token chance 1/V. This is a chance
    # reference, not a trained no-memory model baseline.
    answer_lengths = [len(oracle_answer(ep)) for episodes in checked.values() for ep in episodes]
    uniform_token_chance = 1.0 / VOCAB_SIZE
    mean_exact_chance = sum(VOCAB_SIZE ** (-n) for n in answer_lengths) / len(answer_lengths)
    stateless_rng = random.Random(seed + 1)
    exact_stateless = 0
    for episodes in checked.values():
        for ep in episodes:
            expected = oracle_answer(ep)
            guess = tuple(stateless_rng.randrange(VOCAB_SIZE) for _ in expected)
            exact_stateless += int(guess == expected)
    stateless_exact_rate = exact_stateless / sum(len(eps) for eps in checked.values())
    oracle_exact = sum(
        tuple(ep.targets[ep.answer_start :]) == oracle_answer(ep)
        for eps in checked.values()
        for ep in eps
    )
    oracle_exact_rate = oracle_exact / sum(len(eps) for eps in checked.values())

    lesson_steps = {
        lesson.name: {
            "episodes": len(episodes),
            "mean_tokens_per_episode": sum(len(ep.inputs) for ep in episodes) / len(episodes),
            "total_episode_tokens": sum(len(ep.inputs) for ep in episodes),
            "mean_answer_tokens": sum(len(oracle_answer(ep)) for ep in episodes) / len(episodes),
        }
        for lesson, episodes in zip(LESSONS, checked.values())
    }
    result = {
        "scope": "generator-level checks; no model is trained by this command",
        "labels_match_generator_oracle": labels_valid,
        "content_only_a2_miss_rate": content_miss_rate,
        "content_only_a3_false_alarm_rate": content_false_alarm_rate,
        "stateless_uniform_token_chance": uniform_token_chance,
        "stateless_uniform_exact_sequence_chance_mean": mean_exact_chance,
        "stateless_random_exact_match_rate": stateless_exact_rate,
        "oracle_answer_upper_bound_exact_match_rate": oracle_exact_rate,
        "episode_token_counts_by_lesson": lesson_steps,
    }
    if not labels_valid:
        raise AssertionError("SGR labels diverged from generator oracle")
    if content_miss_rate < 0.99 or content_false_alarm_rate < 0.99:
        raise AssertionError("content-only baseline did not fail both A2 and A3 checks")
    if mean_exact_chance >= 0.01 or stateless_exact_rate >= 0.01:
        raise AssertionError("stateless random-guess reference is not sufficiently close to chance")
    if oracle_exact_rate != 1.0:
        raise AssertionError("oracle-flag upper bound did not solve the generated task")
    return result


if __name__ == "__main__":
    import json

    print(json.dumps(run_validity_gate(), indent=2))
