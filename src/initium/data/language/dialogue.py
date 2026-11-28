"""Conversation parsing and runtime-policy-consistent causal examples."""

from __future__ import annotations

import json
from pathlib import Path

from initium.data.language.schema import Dialogue, DialogueTurn

ALLOWED_ROLES = {"system", "user", "assistant", "tool"}


def read_dialogues(path: str | Path):
    with Path(path).open(encoding="utf-8") as stream:
        for line_no, line in enumerate(stream, 1):
            if not line.strip():
                continue
            raw = json.loads(line)
            turns = tuple(DialogueTurn(str(t["role"]), str(t["content"])) for t in raw["turns"])
            if not turns or any(t.role not in ALLOWED_ROLES for t in turns):
                raise ValueError(f"invalid roles or empty conversation at {path}:{line_no}")
            if any(not t.content.strip() for t in turns):
                raise ValueError(f"empty turn at {path}:{line_no}")
            yield Dialogue(str(raw["conversation_id"]), turns, raw.get("split", "train"),
                           raw.get("metadata", {}))


def encode_dialogue(dialogue: Dialogue, tokenizer, *, include_prior_assistant: bool = True):
    """Encode the full prefix and train assistant spans only, matching chat runtime history."""
    ids, labels = [], []
    for turn in dialogue.turns:
        role_id = tokenizer.token_to_id(f"<{turn.role}>")
        end_id = tokenizer.token_to_id("<turn_end>")
        if role_id is None or end_id is None:
            raise ValueError("tokenizer must define role markers and <turn_end>")
        content = tokenizer.encode(turn.content, add_special_tokens=False).ids
        segment = [role_id, *content, end_id]
        ids.extend(segment)
        labels.extend(segment if turn.role == "assistant" else [-100] * len(segment))
        if turn.role == "assistant" and not include_prior_assistant:
            # Runtime policy option for stateless response training. Keep the
            # visible history in inputs; callers can reset state per example.
            pass
    return {"input_ids": ids, "labels": labels, "conversation_id": dialogue.conversation_id}
