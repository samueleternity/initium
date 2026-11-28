"""Deterministic synthetic state curriculum writer for JEPA and logic data."""

from __future__ import annotations

import json
from pathlib import Path
import random

from initium.data.language.schema import StateTransition
from initium.data.language.corpus import stable_split
from initium.data.language.state_data import PLACES, apply_action, make_view_pair, sample_world


def world_to_dict(state: WorldState):
    return {"state_id": state.state_id, "entities": list(state.entities),
            "relations": [fact.__dict__ for fact in state.relations], "rules": list(state.rules),
            "time": state.time, "metadata": state.metadata}


def transition_to_dict(record: StateTransition):
    return {"transition_id": record.transition_id, "before": world_to_dict(record.before),
            "action": record.action, "after": world_to_dict(record.after), "valid": record.valid}


def write_synthetic_state_data(output_dir: str | Path, *, count: int = 10_000, seed: int = 0):
    """Write separate paired-view, transition, and logic JSONL training files."""
    if count < 1:
        raise ValueError("count must be positive")
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    view_path, transition_path, logic_path = (out / "state_views.jsonl", out / "transitions.jsonl",
                                               out / "logic.jsonl")
    with view_path.open("w", encoding="utf-8") as views, \
            transition_path.open("w", encoding="utf-8") as transitions, \
            logic_path.open("w", encoding="utf-8") as logic:
        for index in range(count):
            state = sample_world(rng)
            split = stable_split(state.state_id)
            view_pair = make_view_pair(state, seed=seed + index * 2, split=split)
            views.write(json.dumps({"state_id": view_pair.state_id, "first": view_pair.first,
                                    "second": view_pair.second, "split": split}, ensure_ascii=False) + "\n")
            movable = state.relations[0].subject
            current_place = state.relations[0].object
            destination = rng.choice(tuple(place for place in PLACES if place != current_place))
            action = f"move {movable} to {destination}"
            valid_action = apply_action(state, action)
            invalid_action = apply_action(state, f"move {movable} to the moon")
            transition = valid_action if index % 4 else invalid_action
            transition_record = transition_to_dict(transition) | {"split": split}
            transitions.write(json.dumps(transition_record, ensure_ascii=False) + "\n")
            person, current_place = state.relations[0].subject, state.relations[0].object
            object_name = state.relations[1].subject
            other_place = destination
            move_action = f"move {person} to {other_place}"
            unchanged_action = f"move {object_name} to {other_place}"
            examples = (
                (f"State: {person} is in {current_place}. {object_name} is in {current_place}. "
                 f"Can action '{move_action}' be performed?", "valid_transition"),
                (f"State: {person} is in {current_place}. "
                 f"Can action 'move {person} to the moon' be performed?", "invalid_transition"),
                (f"State: {person} is in {current_place}. {object_name} is in {current_place}. "
                 f"Are {person} and {object_name} in the same place?", "relation_composition_true"),
                (f"State: {person} is in {current_place}. {object_name} is in {current_place}. "
                 f"Is {person} in {other_place}?", "relation_composition_false"),
                (f"State: {person} is in {current_place}. Does the claim '{person} is in {other_place}' "
                 "contradict the state?", "contradiction"),
                (f"State: {person} is in {current_place}. Does the claim '{person} is in {current_place}' "
                 "contradict the state?", "consistent"),
                (f"State: {person} is in {current_place}. After '{move_action}', is {person} still in "
                 f"{current_place}?", "counterfactual_changed"),
                (f"State: {person} is in {current_place}. After '{unchanged_action}', is {person} still in "
                 f"{current_place}?", "counterfactual_unchanged"),
            )
            for query, label in examples:
                logic.write(json.dumps({"state_id": state.state_id, "query": query,
                                        "label": label,
                                        "target_state": world_to_dict(valid_action.after),
                                        "split": split}, ensure_ascii=False) + "\n")
    return {"state_views": str(view_path), "transitions": str(transition_path),
            "logic": str(logic_path), "count": count, "seed": seed}
