"""Typed records shared by state generation, training, and memory."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Fact:
    subject: str
    relation: str
    object: str
    valid_from: int = 0
    valid_to: int | None = None
    source: str = "simulator"
    confidence: float = 1.0


@dataclass(frozen=True)
class WorldState:
    state_id: str
    entities: tuple[str, ...]
    relations: tuple[Fact, ...]
    rules: tuple[str, ...] = ()
    time: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class StateTransition:
    transition_id: str
    before: WorldState
    action: str
    after: WorldState
    valid: bool = True


@dataclass(frozen=True)
class TextViewPair:
    state_id: str
    first: str
    second: str
    split: str = "train"


@dataclass(frozen=True)
class DialogueTurn:
    role: str
    content: str


@dataclass(frozen=True)
class Dialogue:
    conversation_id: str
    turns: tuple[DialogueTurn, ...]
    split: str = "train"
    metadata: dict[str, Any] = field(default_factory=dict)
