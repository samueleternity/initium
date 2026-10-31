"""Capacity tasks: change detection, symbolic multi-object tracking, and n-back."""

from __future__ import annotations

import random

from initium.data.workspace_capability.episode import (
    BOS,
    BOOL_FALSE,
    BOOL_TRUE,
    EOS,
    EV_INIT,
    EV_MOVE,
    EV_UPDATE,
    LOCATION_COUNT,
    QUERY,
    SEP,
    TaskEpisode,
    count_token,
    entity_token,
    location_token,
)
from initium.data.workspace_capability.tracks._shared import common_metadata


def _change_detection(difficulty: dict, rng: random.Random, *, ood: bool) -> TaskEpisode:
    k = min(max(1, int(difficulty["set_size"])), 9 if ood else 8)
    items = list(range(k))
    values = [rng.randrange(LOCATION_COUNT) for _ in items]
    probe = rng.randrange(k)
    changed = bool(rng.getrandbits(1))
    current = list(values)
    if changed:
        current[probe] = (current[probe] + rng.randrange(1, LOCATION_COUNT)) % LOCATION_COUNT
    stream = [BOS]
    for item, value in zip(items, values):
        stream.extend((EV_INIT, entity_token(item), location_token(value), SEP))
    for item, value in zip(items, current):
        stream.extend((EV_UPDATE, entity_token(item), location_token(value), SEP))
    stream.extend((QUERY, entity_token(probe), SEP))
    answer = (BOOL_TRUE if changed else BOOL_FALSE, EOS)
    return TaskEpisode(
        "capacity_bottleneck",
        tuple(stream),
        answer,
        common_metadata(
            track="capacity_bottleneck",
            sequence_length=len(stream),
            query_type=f"change_detection_k{k}",
            relevant_updates=int(changed),
            candidate_baseline_pct=50.0,
            set_size=k,
            probe_item=probe,
            changed=changed,
            ood=bool(ood),
            symbolic=True,
        ),
    )


def _tracking(difficulty: dict, rng: random.Random, *, ood: bool) -> TaskEpisode:
    k = min(max(1, int(difficulty["target_count"])), 9 if ood else 8)
    object_count = min(12, max(k + 2, 4))
    frames = min(6 if ood else 5, max(2, int(difficulty["tracking_frames"])))
    positions = rng.sample(range(LOCATION_COUNT), object_count)
    targets = set(rng.sample(range(object_count), k))
    probe = rng.choice(tuple(sorted(targets)))
    stream = [BOS]
    for obj, pos in enumerate(positions):
        stream.extend(
            (
                EV_INIT,
                entity_token(obj),
                location_token(pos),
                BOOL_TRUE if obj in targets else BOOL_FALSE,
                SEP,
            )
        )
    relevant_updates = 0
    for _ in range(frames):
        order = list(range(object_count))
        rng.shuffle(order)
        for obj in order:
            old = positions[obj]
            pos = (old + rng.randrange(1, LOCATION_COUNT)) % LOCATION_COUNT
            positions[obj] = pos
            relevant_updates += int(obj in targets)
            stream.extend((EV_MOVE, entity_token(obj), location_token(pos), SEP))
    stream.extend((QUERY, entity_token(probe), SEP))
    return TaskEpisode(
        "capacity_bottleneck",
        tuple(stream),
        (location_token(positions[probe]), EOS),
        common_metadata(
            track="capacity_bottleneck",
            sequence_length=len(stream),
            query_type=f"symbolic_multiple_object_tracking_k{k}",
            relevant_updates=relevant_updates,
            distractor_count=(object_count - k) * frames,
            candidate_baseline_pct=100.0 / LOCATION_COUNT,
            target_count=k,
            object_count=object_count,
            tracking_frames=frames,
            probe_item=probe,
            ood=bool(ood),
            symbolic_identity_tokens=True,
        ),
    )


def _n_back(difficulty: dict, rng: random.Random, *, ood: bool) -> TaskEpisode:
    n = min(max(1, int(difficulty["n_back"])), 5 if ood else 4)
    length = min(16, n + 3)
    symbols = [rng.randrange(LOCATION_COUNT) for _ in range(length)]
    is_match = bool(rng.getrandbits(1))
    if is_match:
        symbols[-1] = symbols[-1 - n]
    elif symbols[-1] == symbols[-1 - n]:
        symbols[-1] = (symbols[-1] + rng.randrange(1, LOCATION_COUNT)) % LOCATION_COUNT
    stream = [BOS, EV_INIT, count_token(n), SEP]
    stream.extend(location_token(value) for value in symbols)
    stream.extend((QUERY, SEP))
    return TaskEpisode(
        "capacity_bottleneck",
        tuple(stream),
        (BOOL_TRUE if is_match else BOOL_FALSE, EOS),
        common_metadata(
            track="capacity_bottleneck",
            sequence_length=len(stream),
            query_type=f"n_back_n{n}",
            relevant_updates=int(is_match),
            candidate_baseline_pct=50.0,
            n_back=n,
            stream_length=length,
            ood=bool(ood),
        ),
    )


def generate(difficulty: dict, rng: random.Random, *, ood: bool = False) -> TaskEpisode:
    """Sample one of three distinct memory paradigms, tagged in episode metadata."""
    mode = rng.choice(("change_detection", "symbolic_multiple_object_tracking", "n_back"))
    if mode == "change_detection":
        return _change_detection(difficulty, rng, ood=ood)
    if mode == "symbolic_multiple_object_tracking":
        return _tracking(difficulty, rng, ood=ood)
    return _n_back(difficulty, rng, ood=ood)
