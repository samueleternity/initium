"""Developmental word-binding episodes: in-episode word<->concept binding.

Concept keys are entity-token indices, re-used across families (one family per episode):
objects 0-23, location-now 24-31 (D5) / fact slots 24-31 (D6), properties 32-47,
initial-location 48-55 (D5), relations 56-63. Word-forms are re-drawn every episode;
train words are FACT indices [0,64), held-out words [64,128) (OOD only).
"""

from __future__ import annotations

import random

from initium.data.workspace_capability.episode import (
    BOS, EOS, EV_ASSERT, EV_FACT, EV_INIT, EV_KV, EV_MOVE, EV_UPDATE, LOCATION_COUNT,
    NONE_TOKEN, QUERY, SEP, TaskEpisode, entity_token, fact_token, location_token,
    scalar_token, source_token,
)
from initium.data.workspace_capability.tracks._shared import common_metadata

_OBJ, _LOC, _PROP, _INIT, _REL, _WORDS = 0, 24, 32, 48, 56, 64
_BUDGET = 280  # token cap for teaching scenes (T4 memory)



def _pair_ok(s, r, combos):
    """Held-out (subject, relation) slot combos are used only on the `combos` OOD axis."""
    return ((s * 7 + r) % 4 == 0) == bool(combos)


def _lexicon(rng, keys, held):
    words = rng.sample(range(_WORDS, 2 * _WORDS) if held else range(_WORDS), len(keys) + 1)
    return dict(zip(keys, words[:-1])), words[-1]  # spare word is never uttered


def _cross(rng, stream, groups, W, scenes):
    """Cross-situational scenes: one EV_UPDATE per present concept carrying the utterance."""
    cand: dict[int, set] = {}
    for _ in range(scenes):
        if len(stream) > _BUDGET:
            break
        chosen = [rng.sample(g, rng.randint(lo, min(len(g), max(lo, hi)))) for g, lo, hi in groups]
        present = [k for c in chosen for k in c]
        must = chosen[-1] if len(groups) > 1 else []
        rest = [k for k in present if k not in must]
        said = must + rng.sample(rest, rng.randint(0 if must else 1, min(2 if must else 3, len(rest))))
        rng.shuffle(said)
        for k in said:
            cand[W[k]] = cand[W[k]] & set(present) if W[k] in cand else set(present)
        rng.shuffle(present)
        for k in present:
            stream += [EV_UPDATE, entity_token(k), *(fact_token(W[s]) for s in said), SEP]
    resolved = {k: w for k, w in W.items() if cand.get(w) == {k}}
    if not resolved:  # guarantee a unique, answerable query
        k = rng.choice(list(W))
        stream += [EV_UPDATE, entity_token(k), fact_token(W[k]), SEP]
        resolved = {k: W[k]}
    return resolved


def _teach(rng, stream, mode, W, noise):
    keys = list(W)
    rng.shuffle(keys)
    for k in keys:
        if mode == "correction":  # noisy feedback; true word always has the unique plurality
            others = [W[j] for j in W if j != k]
            trials = [W[k]] * 3 + rng.sample(others, min(2, len(others)))
            rng.shuffle(trials)
            for _ in range(6):
                fb = [(w == W[k]) != (rng.random() < noise) for w in trials]
                if sum(f for w, f in zip(trials, fb) if w == W[k]) >= 2:
                    break
            else:
                fb = [w == W[k] for w in trials]
            for w, f in zip(trials, fb):
                stream += [EV_ASSERT, entity_token(k), fact_token(w), source_token(int(f)), SEP]
        else:  # direct: supervised EV_KV; demo: same pairing via unsupervised EV_UPDATE
            stream += [EV_KV if mode == "direct" else EV_UPDATE, entity_token(k), fact_token(W[k]), SEP]
    return dict(W)


def _episode(stream, answer, family, qtype, baseline, held, ood, updates, **extra):
    return TaskEpisode(
        "developmental", tuple(stream), answer,
        common_metadata(
            track="developmental", sequence_length=len(stream), query_type=qtype,
            relevant_updates=updates, candidate_baseline_pct=baseline,
            progression_group=family, heldout_words=held, ood=bool(ood), **extra,
        ),
    )


def generate(difficulty: dict, rng: random.Random, *, ood: bool = False) -> TaskEpisode:
    family = rng.choice(tuple(difficulty["families"]))
    n = min(max(3, int(difficulty["concepts"])), 12)
    scenes = max(2, int(difficulty["scenes"]))
    amb = max(2, int(difficulty["ambiguity"]))
    noise = float(difficulty["noise"])
    held = rng.random() < float(difficulty["heldout"])
    stream = [BOS]

    if family == "d5_temporal":
        m = min(n, 8)
        loc = [rng.randrange(LOCATION_COUNT) for _ in range(m)]
        init = list(loc)
        for o in range(m):
            stream += [EV_INIT, entity_token(_LOC + o), location_token(loc[o]), SEP,
                       EV_INIT, entity_token(_INIT + o), location_token(loc[o]), SEP]
        moves = rng.randint(1, scenes)
        for _ in range(moves):
            o = rng.randrange(m)
            loc[o] = (loc[o] + rng.randrange(1, LOCATION_COUNT)) % LOCATION_COUNT
            stream += [EV_MOVE, entity_token(_LOC + o), location_token(loc[o]), SEP]
        q, kind = rng.randrange(m), rng.randrange(2)
        stream += [QUERY, entity_token(q), scalar_token(kind), SEP]
        return _episode(
            stream, (location_token((init if kind else loc)[q]), EOS), family,
            f"{family}_{'before' if kind else 'now'}", 100.0 / LOCATION_COUNT, False, ood, moves,
            workspace_output_keys=((_INIT if kind else _LOC) + q, -1),
        )

    if family == "d5_action":  # verb words: "go" -> current location, "back" -> initial location
        m = min(n, 6)
        W, _ = _lexicon(rng, list(range(m)) + [_REL, _REL + 1], held)
        _teach(rng, stream, "direct", W, 0.0)
        loc = [rng.randrange(LOCATION_COUNT) for _ in range(m)]
        init = list(loc)
        for i in range(m):
            stream += [EV_INIT, entity_token(_LOC + i), location_token(loc[i]), SEP,
                       EV_INIT, entity_token(_INIT + i), location_token(loc[i]), SEP]
        moves = rng.randint(1, scenes)
        for _ in range(moves):
            i = rng.randrange(m)
            loc[i] = (loc[i] + rng.randrange(1, LOCATION_COUNT)) % LOCATION_COUNT
            stream += [EV_MOVE, entity_token(_LOC + i), location_token(loc[i]), SEP]
        q, back = rng.randrange(m), rng.randrange(2)
        stream += [QUERY, fact_token(W[q]), fact_token(W[_REL + back]), SEP]
        return _episode(
            stream, (location_token((init if back else loc)[q]), EOS), family,
            f"{family}_{'back' if back else 'go'}", 100.0 / LOCATION_COUNT, held, ood, moves,
            workspace_query_reads=tuple((len(stream) - 1, c, fact_token(w)) for c, w in W.items()),
        )

    objs = rng.sample(range(_OBJ, _LOC), n)
    if family in ("d1_naming", "d2_cross"):
        keys, groups = objs, [(objs, 2, amb)]
    elif family == "d3_attribute":
        objs = objs[: max(2, n // 2)]
        feats = [_PROP + v for v in rng.sample(range(16), max(2, n - len(objs)))]
        keys, groups = objs + feats, [(objs, 1, max(1, amb - 1)), (feats, 1, 1)]
    else:  # d4_relation, d6_compose
        rels = [_REL + r for r in rng.sample(range(8), 1 + (n >= 6))]
        objs = objs[: n - len(rels)]
        keys, groups = objs + rels, [(objs, 1, amb), (rels, 1, 1)]

    if family == "d1_naming":
        mode = rng.choice(("direct", "demo") + (("correction",) if noise > 0 else ()))
    else:
        mode = "direct" if family == "d6_compose" else "ambiguous"
    W, spare = _lexicon(rng, keys, held)
    R = _cross(rng, stream, groups, W, scenes) if mode == "ambiguous" else _teach(rng, stream, mode, W, noise)

    if family == "d6_compose":
        pairs = [(s, r) for s in objs for r in rels]
        ok = [p for p in pairs if _pair_ok(*p, difficulty.get("combos", 0))]
        pairs = ok if len(ok) >= 2 else pairs
        picks = rng.sample(pairs, min(len(pairs), 8, max(2, len(objs))))
        facts = {}
        for s, r in picks:
            facts[(s, r)] = rng.choice([x for x in objs if x != s])
            stream += [EV_FACT, entity_token(s), entity_token(r), entity_token(facts[(s, r)]), SEP]
        s, r = rng.choice(picks)
        stream += [QUERY, fact_token(W[s]), fact_token(W[r]), SEP]
        answer, direction, extra, baseline = (entity_token(facts[(s, r)]), EOS), "compose", {}, 100.0 / len(objs)
    else:
        withheld = rng.random() < 0.15  # queried word never taught -> NONE
        qk = rng.choice(list(R))
        if not withheld and rng.random() < 0.5:
            stream += [QUERY, entity_token(qk), SEP]
            answer, direction, baseline = (fact_token(R[qk]), EOS), "c2w", 100.0 / len(R)
            extra = {"workspace_output_keys": (qk, -1)}
        else:
            stream += [QUERY, fact_token(spare if withheld else R[qk]), SEP]
            answer = (NONE_TOKEN if withheld else entity_token(qk), EOS)
            direction, baseline = ("none" if withheld else "w2c"), 100.0 / (len(R) + 1)
            extra = {"workspace_output_programs": (
                (0, "argmatch", tuple(R), tuple(fact_token(w) for w in R.values())),)}
    # Auxiliary slot supervision (training loss only): each resolved concept slot decodes its word.
    extra["workspace_query_reads"] = tuple((len(stream) - 1, k, fact_token(w)) for k, w in R.items())
    extra["lexicon"] = tuple((k, fact_token(w)) for k, w in R.items())
    return _episode(
        stream, answer, family, f"{family}_{mode}_{direction}_{'heldout' if held else 'train'}",
        baseline, held, ood, len(R), teacher=mode, resolved_count=len(R), **extra,
    )