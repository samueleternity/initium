"""Cue-conditioned RSVP target identification with lag and mask controls."""

from __future__ import annotations

import random

from initium.data.workspace_capability.episode import (
    BOS,
    EV_NOISE,
    EOS,
    QUERY,
    SCALAR_COUNT,
    SEP,
    TaskEpisode,
    scalar_token,
    source_token,
)
from initium.data.workspace_capability.tracks._shared import common_metadata


def _distractor(
    rng: random.Random,
    *,
    target_color: int,
    target_shape: int,
    lure_probability: float,
) -> tuple[int, int]:
    """Sample a non-target item, often sharing one cued feature."""
    if rng.random() < lure_probability:
        if rng.getrandbits(1):
            color = target_color
            shape = rng.choice(tuple(value for value in range(4, 8) if value != target_shape))
        else:
            color = rng.choice(tuple(value for value in range(4) if value != target_color))
            shape = target_shape
    else:
        color = rng.choice(tuple(value for value in range(4) if value != target_color))
        shape = rng.choice(tuple(value for value in range(4, 8) if value != target_shape))
    return color, shape


def _append_item(
    stream: list[int], color: int, shape: int, value: int, *, masked: bool = False
) -> None:
    """Append one uniformly encoded RSVP item frame."""
    if masked:
        # The mask is attached to T2 so the frame-level workspace encoder can
        # distinguish an item that was locally processed from one eligible for
        # stable item storage. A trailing noise frame after T2 did not mask T2.
        stream.append(EV_NOISE)
    stream.extend((source_token(color), source_token(shape), scalar_token(value), SEP))


def generate(difficulty: dict, rng: random.Random, *, ood: bool = False) -> TaskEpisode:
    maximum_lag = min(max(2, int(difficulty["maximum_lag"])), 9 if ood else 8)
    lag = rng.randint(1, maximum_lag)
    mask_applied = bool(rng.getrandbits(1))
    distractor_count = min(max(2, int(difficulty.get("distractors", 4))), 16 if ood else 14)
    lure_probability = min(max(float(difficulty.get("lure_probability", 0.5)), 0.0), 0.95)

    target_color = rng.randrange(4)
    target_shape = rng.randrange(4, 8)
    target_one = rng.randrange(SCALAR_COUNT)
    target_two = rng.choice(tuple(value for value in range(SCALAR_COUNT) if value != target_one))

    # The cue defines a conjunction over two ordinary item features. Distractors
    # often match one feature, making a single-feature/parity filter insufficient.
    stream = [BOS, source_token(target_color), source_token(target_shape), SEP]
    before_count = distractor_count // 2
    after_count = distractor_count - before_count
    lure_count = 0
    for _ in range(before_count):
        color, shape = _distractor(
            rng,
            target_color=target_color,
            target_shape=target_shape,
            lure_probability=lure_probability,
        )
        lure_count += int(color == target_color or shape == target_shape)
        value = rng.randrange(SCALAR_COUNT)
        _append_item(stream, color, shape, value)

    _append_item(stream, target_color, target_shape, target_one)

    # T2 lag is measured in RSVP frames; each intervening frame is an ordinary
    # distractor, including at lag > 1.
    for gap_index in range(lag - 1):
        color, shape = _distractor(
            rng,
            target_color=target_color,
            target_shape=target_shape,
            lure_probability=lure_probability,
        )
        lure_count += int(color == target_color or shape == target_shape)
        value = rng.randrange(SCALAR_COUNT)
        _append_item(stream, color, shape, value)

    _append_item(stream, target_color, target_shape, target_two, masked=mask_applied)

    for _ in range(after_count):
        color, shape = _distractor(
            rng,
            target_color=target_color,
            target_shape=target_shape,
            lure_probability=lure_probability,
        )
        lure_count += int(color == target_color or shape == target_shape)
        value = rng.randrange(SCALAR_COUNT)
        _append_item(stream, color, shape, value)
    stream.extend((QUERY, SEP))

    return TaskEpisode(
        "attentional_blink",
        tuple(stream),
        (scalar_token(target_one), scalar_token(target_two), EOS),
        common_metadata(
            track="attentional_blink",
            sequence_length=len(stream),
            query_type=f"lag_{lag}_{'masked' if mask_applied else 'unmasked'}",
            relevant_updates=2,
            distractor_count=distractor_count + lag - 1,
            candidate_baseline_pct=100.0 / SCALAR_COUNT,
            target_lag=lag,
            lag_band="short" if lag <= 2 else "long",
            mask_condition=mask_applied,
            target_color=target_color,
            target_shape=target_shape,
            partial_match_distractors=lure_count,
            lure_probability=lure_probability,
            target_one=target_one,
            target_two=target_two,
            ood=bool(ood),
            progression_group=f"lag_{lag}_{'masked' if mask_applied else 'unmasked'}",
            target_and_distractors_share_token_type=True,
            cue_requires_feature_conjunction=True,
            mask_marks_t2_frame_before_workspace_encoding=mask_applied,
        ),
    )
