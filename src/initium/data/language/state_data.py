"""Small symbolic simulator that emits exact states, paraphrase views, and transitions."""

from __future__ import annotations

import random
import hashlib

from initium.data.language.schema import Fact, StateTransition, TextViewPair, WorldState

PEOPLE = (
    "Mira", "Jon", "Asha", "Lena", "Omar", "Nia", "Pavel", "Ruth", "Tariq", "Uma",
    "Vera", "Wes", "Xia", "Yara", "Zane", "Ari", "Bea", "Cato", "Dara", "Ezra",
    "Faye", "Gus", "Hana", "Ivo", "Jules", "Kian", "Luz", "Milo", "Noor", "Orin",
    "Pia", "Quin",
)
OBJECTS = (
    "the key", "the book", "the cup", "the lamp", "the bag", "the phone", "the map",
    "the coat", "the bowl", "the toy", "the box", "the letter", "the bottle", "the tool",
    "the card", "the folder", "the brush", "the clock", "the plate", "the camera",
)
PLACES = (
    "the kitchen", "the garden", "the office", "the hallway", "the studio", "the library",
    "the workshop", "the balcony", "the attic", "the cellar", "the lobby", "the garage",
)
ENTITIES = PEOPLE + OBJECTS + PLACES


def _stable_id(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:24]


def render_state(state: WorldState, rng: random.Random, *, style_offset: int = 0) -> str:
    """Select a wording independently while preserving the same underlying facts."""
    templates = (
        "At time {time}, {subject} {relation} {object}.",
        "The record for time {time} says that {subject} {relation} {object}.",
        "As of {time}: {subject} {relation} {object}.",
    )
    facts = state.relations
    if not facts:
        return f"At time {state.time}, the world contains {', '.join(state.entities)}."
    style = rng.randrange(len(templates))
    return " ".join(templates[(style + style_offset + index) % len(templates)].format(
        time=state.time, subject=f.subject, relation=f.relation, object=f.object
    ) for index, f in enumerate(facts))


def sample_world(rng: random.Random, *, time: int = 0) -> WorldState:
    person = rng.choice(PEOPLE)
    object_name = rng.choice(OBJECTS)
    place = rng.choice(PLACES)
    facts = (
        Fact(person, "is in", place, valid_from=time),
        Fact(object_name, "is in", place, valid_from=time),
    )
    state_id = _stable_id(f"{time}|{person}|{object_name}|{place}")
    return WorldState(state_id, tuple(sorted({person, object_name, place})), facts,
                      rules=("move(entity, destination) changes only that entity's location",), time=time)


def apply_action(state: WorldState, action: str) -> StateTransition:
    """Apply ``move ENTITY to PLACE``; malformed or unknown moves are invalid no-ops."""
    parts = action.split(" to ", 1)
    valid = len(parts) == 2 and parts[0].startswith("move ")
    entity = parts[0][5:] if valid else ""
    destination = parts[1] if valid else ""
    facts = list(state.relations)
    current = next((f for f in reversed(facts) if f.subject == entity and f.relation == "is in"), None)
    valid = bool(valid and current and destination in PLACES)
    next_time = state.time + 1
    if valid:
        facts = [Fact(f.subject, f.relation, f.object, f.valid_from, next_time, f.source, f.confidence)
                 if f is current else f for f in facts]
        facts.append(Fact(entity, "is in", destination, next_time))
    after_id = _stable_id(f"{state.state_id}|{action}|{next_time}")
    after = WorldState(after_id, state.entities, tuple(facts), state.rules, next_time,
                       {"parent_state_id": state.state_id, "action": action})
    transition_id = _stable_id(f"{state.state_id}|{action}|{after_id}")
    return StateTransition(transition_id, state, action, after, valid)


def make_view_pair(state: WorldState, seed: int = 0, split: str = "train") -> TextViewPair:
    first = render_state(state, random.Random(seed))
    second = render_state(state, random.Random(seed + 1), style_offset=1)
    if first == second:
        second = render_state(state, random.Random(seed + 1), style_offset=2)
    return TextViewPair(state.state_id, first, second, split)
