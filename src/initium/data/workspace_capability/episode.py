"""Typed token vocabulary and immutable episode records for workspace tasks."""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import chain
from typing import Any

VOCAB_SIZE = 320
PAD = 0
BOS = 1
EOS = 2
SEP = 3
QUERY = 4
ANSWER = 5
TRACK_BASE = 24

# Event tokens occupy a dedicated range; fields use disjoint typed ranges.
EV_INIT = 8
EV_SWAP = 9
EV_PUT = 10
EV_MOVE = 11
EV_REMOVE = 12
EV_ASSERT = 13
EV_RETRACT = 14
EV_SUPERSEDE = 15
EV_FACT = 16
EV_VECTOR = 17
EV_KV = 18
EV_UPDATE = 19
EV_COUNT_QUERY = 20
EV_SUM_QUERY = 21
EV_PARITY_QUERY = 22
EV_NOISE = 23
EV_COMMIT = 188

ENTITY_BASE, ENTITY_COUNT = 32, 64
SOURCE_BASE, SOURCE_COUNT = 96, 8
RELATION_BASE, RELATION_COUNT = 104, 16
LOCATION_BASE, LOCATION_COUNT = 120, 16
SCALAR_BASE, SCALAR_COUNT = 136, 16
COUNT_BASE, COUNT_COUNT = 152, 32
BOOL_FALSE, BOOL_TRUE = 184, 185
NONE_TOKEN = 186
SOURCE_PRIORITY_POLICY = 187
FACT_BASE, FACT_COUNT = 192, 128


def entity_token(index: int) -> int:
    if not 0 <= int(index) < ENTITY_COUNT:
        raise ValueError(f"entity index must be in [0,{ENTITY_COUNT}), got {index}")
    return ENTITY_BASE + int(index)


def source_token(index: int) -> int:
    if not 0 <= int(index) < SOURCE_COUNT:
        raise ValueError(f"source index must be in [0,{SOURCE_COUNT}), got {index}")
    return SOURCE_BASE + int(index)


def relation_token(index: int) -> int:
    if not 0 <= int(index) < RELATION_COUNT:
        raise ValueError(f"relation index must be in [0,{RELATION_COUNT}), got {index}")
    return RELATION_BASE + int(index)


def location_token(index: int) -> int:
    if not 0 <= int(index) < LOCATION_COUNT:
        raise ValueError(f"location index must be in [0,{LOCATION_COUNT}), got {index}")
    return LOCATION_BASE + int(index)


def scalar_token(value: int) -> int:
    if not 0 <= int(value) < SCALAR_COUNT:
        raise ValueError(f"scalar must be in [0,{SCALAR_COUNT}), got {value}")
    return SCALAR_BASE + int(value)


def count_token(value: int) -> int:
    if not 0 <= int(value) < COUNT_COUNT:
        raise ValueError(f"count must be in [0,{COUNT_COUNT}), got {value}")
    return COUNT_BASE + int(value)


def fact_token(index: int) -> int:
    if not 0 <= int(index) < FACT_COUNT:
        raise ValueError(f"fact index must be in [0,{FACT_COUNT}), got {index}")
    return FACT_BASE + int(index)


@dataclass(frozen=True, slots=True)
class TaskEpisode:
    """One serialized stream, causal answer sequence, and oracle-only metadata."""

    track: str
    inputs: tuple[int, ...]
    answer: tuple[int, ...]
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.track:
            raise ValueError("episode track must be non-empty")
        if not self.inputs or self.inputs[0] != BOS:
            raise ValueError("episode input must begin with BOS")
        if not self.answer or self.answer[-1] != EOS:
            raise ValueError("episode answer must end with EOS")
        if any(token < 0 or token >= VOCAB_SIZE for token in chain(self.inputs, self.answer)):
            raise ValueError("episode contains a token outside the configured vocabulary")
        if not self.metadata.get("query_type"):
            raise ValueError("episode metadata must include query_type")

    @property
    def answer_start(self) -> int:
        return len(self.inputs)

    @property
    def sequence(self) -> tuple[int, ...]:
        # ANSWER is an input-side placeholder: it cannot leak the oracle label.
        return self.inputs + (ANSWER,) * len(self.answer)

    @property
    def targets(self) -> tuple[int, ...]:
        return (PAD,) * len(self.inputs) + self.answer

    @property
    def answer_mask(self) -> tuple[float, ...]:
        return (0.0,) * len(self.inputs) + (1.0,) * len(self.answer)


def validate_episode(episode: TaskEpisode) -> TaskEpisode:
    """Cheap generator-boundary validation; the track oracle owns semantics."""
    expected_length = len(episode.inputs) + len(episode.answer)
    if expected_length != len(episode.targets) or expected_length != len(episode.answer_mask):
        raise ValueError("episode input/target/mask lengths do not match")
    if episode.metadata.get("answer_count", len(episode.answer) - 1) != len(episode.answer) - 1:
        raise ValueError("episode answer_count metadata disagrees with oracle answer")
    return episode
