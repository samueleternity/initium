"""Teacher-supervised arithmetic/logic/conditional probe for rarely used V4 operators."""

from __future__ import annotations

import random

from initium.data.workspace_capability.episode import (
    BOS, ENTITY_BASE, ENTITY_COUNT, EOS, EV_INIT, EV_VECTOR, QUERY, RELATION_BASE,
    RELATION_COUNT, SCALAR_BASE, SCALAR_COUNT, SEP, TaskEpisode, entity_token,
    relation_token, scalar_token,
)
from initium.data.workspace_capability.tracks._shared import common_metadata

OPERATIONS = ("sub", "mul", "neg", "inc", "dec", "eq", "member", "cond_copy",
              "and", "or", "xor", "not", "max")
_BIT_RESULT = ("eq", "member", "and", "or", "xor", "not")
_HELD_OUT = {"mul": "dec", "sub": "max", "inc": "eq", "neg": "member"}
_HELD_OUT_PAIRS = set(_HELD_OUT.items())


def generate(difficulty: dict, rng: random.Random, *, ood: bool = False) -> TaskEpisode:
    n_ops = min(max(2, int(difficulty["operations"])), 8)
    keys = min(max(4, int(difficulty["keys"])), 12)
    values = {k: rng.randrange(SCALAR_COUNT) for k in range(keys)}
    stream = [BOS]
    for k, v in values.items():
        stream.extend((EV_INIT, entity_token(k), scalar_token(v), SEP))
    bits: list[int] = []
    chosen: list[str] = []
    final = 0
    for i in range(n_ops):
        allowed = ["sub", "mul", "neg", "inc", "dec", "eq", "member", "max"]
        if bits:
            allowed += ["cond_copy", "not"]
        if len(bits) >= 2:
            allowed += ["and", "or", "xor"]
        prev = chosen[-1] if chosen else None
        if ood:
            forced = _HELD_OUT.get(prev)
            op = forced if forced and rng.random() < 0.4 else rng.choice(allowed)
        else:
            op = rng.choice([a for a in allowed if (prev, a) not in _HELD_OUT_PAIRS])
        pool = list(values)
        dest = 32 + i
        if op in ("sub", "mul", "eq"):
            a = rng.choice(pool)
            va = values[a]
            if rng.random() < 0.5:
                vb = rng.randrange(SCALAR_COUNT)
                args = (entity_token(a), scalar_token(vb))
            else:
                e = rng.choice([x for x in pool if x != a])
                vb = values[e]
                args = (entity_token(a), entity_token(e))
            out = {"sub": (va - vb) % 16, "mul": va * vb % 16, "eq": int(va == vb)}[op]
        elif op in ("neg", "inc", "dec"):
            a = rng.choice(pool)
            va = values[a]
            args = (entity_token(a),)
            out = {"neg": (-va) % 16, "inc": (va + 1) % 16, "dec": (va - 1) % 16}[op]
        elif op == "member":
            s = rng.sample(pool, 2)
            x = values[s[0]] if rng.random() < 0.5 else rng.randrange(SCALAR_COUNT)
            args = (entity_token(s[0]), entity_token(s[1]), scalar_token(x))
            out = int(x in (values[s[0]], values[s[1]]))
        elif op == "max":
            s = rng.sample(pool, 3)
            args = tuple(entity_token(e) for e in s)
            out = max(values[e] for e in s)
        elif op == "cond_copy":
            dest = rng.randrange(keys)
            flag = rng.choice(bits)
            src = rng.choice([x for x in pool if x not in (dest, flag)])
            args = (entity_token(src), entity_token(flag))
            out = values[src] if values[flag] else values[dest]
        elif op == "not":
            args = (entity_token(rng.choice(bits)),)
            out = 1 - values[int(args[0]) - ENTITY_BASE]
        else:  # and / or / xor
            f = rng.sample(bits, 2)
            args = tuple(entity_token(e) for e in f)
            x, y = values[f[0]], values[f[1]]
            out = {"and": x & y, "or": x | y, "xor": x ^ y}[op]
        stream.extend((EV_VECTOR, entity_token(dest), relation_token(OPERATIONS.index(op)),
                       *args, SEP))
        values[dest] = out
        if op in _BIT_RESULT:
            bits.append(dest)
        chosen.append(op)
        final = dest
    stream.extend((QUERY, entity_token(final), SEP))
    return TaskEpisode(
        "conditional_ops", tuple(stream), (scalar_token(values[final]), EOS),
        common_metadata(
            track="conditional_ops", sequence_length=len(stream), query_type=f"ops{n_ops}",
            relevant_updates=n_ops, distractor_count=keys,
            candidate_baseline_pct=100.0 / SCALAR_COUNT, operations=tuple(chosen),
            progression_group="all", workspace_output_keys=(final, -1), ood=bool(ood),
        ),
    )


def parse_records(tokens, n_inputs):
    """Input-only oracle replay -> records in `_workspace_operation_events` format.

    Set operands for member/max: operand0/operand1 hold the first two keys and
    destination1 carries the third (it is unused by these single-output ops).
    """
    tok = scalar_token
    values: dict[int, int] = {}
    records = []
    i = 0
    while i < n_inputs:
        ev = int(tokens[i])
        if ev not in (EV_INIT, EV_VECTOR):
            i += 1
            continue
        end = i + 1
        while end < n_inputs and int(tokens[end]) != SEP:
            end += 1
        if end >= n_inputs:
            break
        f = [int(t) for t in tokens[i + 1 : end]]
        ents = [t - ENTITY_BASE for t in f if ENTITY_BASE <= t < ENTITY_BASE + ENTITY_COUNT]
        scal = [t for t in f if SCALAR_BASE <= t < SCALAR_BASE + SCALAR_COUNT]
        i = end + 1
        if ev == EV_INIT:
            values[ents[0]] = scal[0] - SCALAR_BASE
            continue
        op = OPERATIONS[next(t - RELATION_BASE for t in f
                             if RELATION_BASE <= t < RELATION_BASE + RELATION_COUNT)]
        dest, d1 = ents[0], -1
        o0, o1 = (1, ents[1]) if len(ents) > 1 else (0, -1), (0, -1)
        ea, eb = -100, -100
        if op in ("sub", "mul", "eq"):
            va = values[ents[1]]
            if len(ents) >= 3:
                vb, o1 = values[ents[2]], (1, ents[2])
            else:
                vb, o1 = scal[0] - SCALAR_BASE, (2, scal[0])
            out = {"sub": (va - vb) % 16, "mul": va * vb % 16, "eq": int(va == vb)}[op]
            ea, eb = tok(va), tok(vb)
        elif op in ("neg", "inc", "dec"):
            va = values[ents[1]]
            out = {"neg": (-va) % 16, "inc": (va + 1) % 16, "dec": (va - 1) % 16}[op]
            ea = tok(va)
        elif op == "member":
            x = scal[0] - SCALAR_BASE
            o0, o1, d1 = (2, scal[0]), (1, ents[1]), ents[2]
            out = int(x in (values[ents[1]], values[ents[2]]))
            ea = tok(x)
        elif op == "max":
            o0, o1, d1 = (1, ents[1]), (1, ents[2]), ents[3]
            out = max(values[e] for e in ents[1:4])
        elif op == "cond_copy":
            o1 = (1, ents[2])
            flag = values[ents[2]]
            out = values[ents[1]] if flag else values[dest]
            ea, eb = tok(values[ents[1]]), tok(flag)
        elif op == "not":
            out = 1 - values[ents[1]]
            ea = tok(values[ents[1]])
        else:
            x, y = values[ents[1]], values[ents[2]]
            o1 = (1, ents[2])
            out = {"and": x & y, "or": x | y, "xor": x ^ y}[op]
            ea, eb = tok(x), tok(y)
        values[dest] = out
        records.append((end, op, dest, d1, o0, o1, (tok(out),), (ea, eb)))
    return tuple(records)