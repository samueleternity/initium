"""Objective-batch scheduling helpers for interleaved training loops."""

from __future__ import annotations

import random


OBJECTIVES = ("text", "same_state", "transition", "logic", "dialogue")


def interleave_objectives(iterators: dict, weights: dict[str, float], *, seed: int = 0):
    """Yield objective-tagged batches using weighted sampling until all streams end."""
    active = {name: iter(stream) for name, stream in iterators.items() if weights.get(name, 0) > 0}
    if not active:
        raise ValueError("at least one objective stream must have positive weight")
    rng = random.Random(seed)
    while active:
        names = list(active)
        chosen = rng.choices(names, weights=[weights[name] for name in names], k=1)[0]
        try:
            yield chosen, next(active[chosen])
        except StopIteration:
            del active[chosen]


def pack_documents(tokenized_documents, *, eos_id: int, max_tokens: int):
    """Pack ordinary document tokens continuously, retaining document EOS markers."""
    buffer = []
    for document in tokenized_documents:
        buffer.extend(document)
        buffer.append(eos_id)
        while len(buffer) >= max_tokens:
            yield buffer[:max_tokens]
            buffer = buffer[max_tokens:]
    if len(buffer) > 1:
        yield buffer
