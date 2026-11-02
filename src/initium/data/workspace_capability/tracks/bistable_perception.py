"""Two-interpretation rivalry probe with stochastic evidence reversals."""

from __future__ import annotations

import random

from initium.data.workspace_capability.episode import (
    BOS, BOOL_FALSE, BOOL_TRUE, EOS, EV_INIT, EV_UPDATE, QUERY, SEP,
    TaskEpisode, entity_token, scalar_token,
)
from initium.data.workspace_capability.tracks._shared import common_metadata


def generate(difficulty: dict, rng: random.Random, *, ood: bool = False) -> TaskEpisode:
    frames = rng.randint(4, min(10, int(difficulty.get("maximum_frames", 8)) + int(ood)))
    initial = rng.randrange(2)
    state = initial
    switches = 0
    integrated_evidence = 0
    stream = [BOS, EV_INIT, entity_token(0), scalar_token(initial), SEP]
    evidence: list[int] = []
    for _ in range(frames):
        # A symmetric evidence stream drives a leaky hysteresis process.
        cue = rng.choice((-1, 0, 1))
        evidence.append(cue)
        integrated_evidence += cue
        opposing = integrated_evidence < 0 if state == 1 else integrated_evidence > 0
        if opposing and abs(integrated_evidence) >= 2:
            state = 1 - state
            switches += 1
            integrated_evidence = 0
        stream.extend((EV_UPDATE, entity_token(1 + len(evidence) % 8), scalar_token(cue + 1), SEP))
    stream.extend((QUERY, entity_token(0), SEP))
    return TaskEpisode(
        "bistable_perception", tuple(stream), (BOOL_TRUE if state else BOOL_FALSE, EOS),
        common_metadata(
            track="bistable_perception", sequence_length=len(stream),
            query_type=f"frames_{frames}_switches_{min(switches, 4)}",
            relevant_updates=frames, distractor_count=frames, candidate_baseline_pct=50.0,
            frame_count=frames, initial_dominance=initial, dominance_switches=switches,
            evidence_trace=tuple(evidence), final_dominance=state,
            progression_group="all",
        ),
    )
