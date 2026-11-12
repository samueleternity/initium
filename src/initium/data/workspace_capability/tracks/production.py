"""Stage 2 production episodes: concept->word description (2a single fact, 2b multi-fact)."""

from __future__ import annotations

import random

from initium.data.workspace_capability.episode import (
    BOS, EOS, EV_FACT, EV_KV, EV_RETRACT, EV_SUPERSEDE, QUERY, SEP, TaskEpisode, entity_token as e, fact_token, scalar_token,
)
from initium.data.workspace_capability.tracks._shared import common_metadata
from initium.data.workspace_capability.tracks.developmental import (
    _LOC, _OBJ, _PROP, _REL, _cross, _lexicon, _pair_ok, _teach,
)
_ORDERS = ((0, 1, 2), (0, 2, 1), (1, 0, 2), (2, 1, 0), (1, 2, 0), (2, 0, 1))  # role (S,R,O) per output slot; 0-2 train, 3-5 held-out

def _top_up(stream, W, R, pool, need):
    """Directly teach just enough unresolved concepts so the answer is determined."""
    for x in [x for x in pool if x not in R][: max(0, need - sum(x in R for x in pool))]:
        stream += [EV_KV, e(x), fact_token(W[x]), SEP]
        R[x] = W[x]

def generate(difficulty, rng, *, ood=False, keyed=True, track="developmental_production"):
    grp = rng.choice(tuple(difficulty["groups"]))
    multi = grp.startswith("p_all")
    k = (2 if grp == "p_all2" else rng.choice((3, 4))) if multi else 0
    d = rng.randint(0, int(difficulty["distractors"])) if multi else 0
    held = rng.random() < float(difficulty["heldout"])
    combos = int(difficulty.get("combos", 0))
    scenes, amb = max(2, int(difficulty["scenes"])), max(2, int(difficulty["ambiguity"]))
    nf = k + d if multi else min(int(difficulty["facts_a"]), 3)
    objs = rng.sample(range(_OBJ, _LOC), min(10, max(int(difficulty["concepts"]) - 2, nf + 2)))
    rels = [_REL + r for r in rng.sample(range(8), 3)]
    props = [_PROP + v for v in rng.sample(range(16), 3)] if grp == "p_property" else []
    W, _ = _lexicon(rng, objs + rels + props, held)
    stream = [BOS]
    if rng.random() < float(difficulty["ambiguity_prob"]):
        gs = [(objs, 1, amb), (rels, 1, 1)] + ([(props, 1, 1)] if props else [])
        R, mode = _cross(rng, stream, gs, W, scenes), "ambiguous"
    else:
        R, mode = _teach(rng, stream, "direct", W, 0.0), "direct"
    _top_up(stream, W, R, objs, nf + 1)
    _top_up(stream, W, R, rels, nf if grp == "p_relation" else (2 if d else 1))
    _top_up(stream, W, R, props, 1)
    ro, rr, rp = ([x for x in p if x in R] for p in (objs, rels, props))

    if multi:
        rq = rng.choice(rr)
        others = [x for x in rr if x != rq]
        subs = rng.sample(ro, k + d)
        tgt = subs[:k]
        cur = {s: (rq if s in tgt else rng.choice(others), rng.choice([x for x in ro if x != s])) for s in subs}
        order = subs[:]
        rng.shuffle(order)
        for s in order:
            stream += [EV_FACT, e(s), e(cur[s][0]), e(cur[s][1]), SEP]
        if rng.random() < 0.6:  # state change: describe CURRENT state only
            s0 = rng.choice(tgt)
            cur[s0] = (rq, rng.choice([x for x in ro if x not in (s0, cur[s0][1])] or ro))
            stream += [EV_FACT, e(s0), e(rq), e(cur[s0][1]), SEP]
            order.remove(s0)
            order.append(s0)
        by_id = bool(difficulty["ordering"]) and rng.random() < 0.5
        seq = sorted(tgt) if by_id else [s for s in order if s in tgt]
        stream += [QUERY, EV_SUPERSEDE if by_id else EV_RETRACT, e(rq), SEP]
        ans, keys_out = [], []
        for s in seq:
            ans += [fact_token(R[s]), fact_token(R[rq]), fact_token(R[cur[s][1]])]
            keys_out += [s, rq, cur[s][1]]
        stride, qtype, base = 3, f"describe_all_k{k}_{'id' if by_id else 'time'}", 100.0 / len(R)
    else:
        subs = rng.sample(ro, nf)
        facts, used = {}, []
        for s in subs:
            pool = [x for x in rr if x not in used] if grp == "p_relation" else rr
            r = rng.choice([x for x in pool if _pair_ok(s, x, combos)] or pool)
            used.append(r)
            facts[s] = (r, rng.choice([x for x in ro if x != s]), rng.choice(rp) if rp else None)
        order = subs[:]
        rng.shuffle(order)
        for s in order:
            r, o, p = facts[s]
            stream += [EV_FACT, e(s), e(r), e(o), *([e(p)] if p is not None else []), SEP]
        s = rng.choice(subs)
        r, o, p = facts[s]
        oi = rng.randrange(3) + 3 * int(bool(combos)) if grp == "p_order" else -1
        stream += [QUERY, e(r) if grp == "p_relation" else fact_token(R[s]) if grp == "p_comp" else e(s),
                   *(scalar_token(x) for x in _ORDERS[oi] if oi >= 0), SEP]
        if grp == "p_comp":
            ans, keys_out, stride, qtype = [e(s)], [], 0, "comprehend_entity"
        else:
            keys_out = [s] + ([p] if p is not None else []) + [r, o]
            if oi >= 0:
                keys_out = [(s, r, o)[i] for i in _ORDERS[oi]]
            ans = [fact_token(R[x]) for x in keys_out]
            stride, qtype = len(ans), {"p_entity": "describe_entity", "p_relation": "describe_relation", "p_order": "describe_ordered",
                                       "p_property": "describe_property"}[grp]
        base = 100.0 / len(R)

    extra = {"workspace_query_reads": tuple((len(stream) - 1, c, fact_token(w)) for c, w in R.items())}
    extra["lexicon"] = tuple((c, fact_token(w)) for c, w in R.items())
    extra["answer_concepts"] = tuple(keys_out)
    if grp == "p_comp":
        extra["workspace_output_programs"] = ((0, "argmatch", tuple(R), tuple(fact_token(w) for w in R.values())),)
    elif keyed:
        extra["workspace_output_keys"] = tuple(keys_out) + (-1,)
    return TaskEpisode(
        track, tuple(stream), tuple(ans) + (EOS,),
        common_metadata(
            track=track, sequence_length=len(stream),
            query_type=f"{qtype}_{mode}_{'heldout' if held else 'train'}",
            relevant_updates=nf, distractor_count=d, candidate_baseline_pct=base,
            progression_group=grp, answer_stride=stride, heldout_words=held, ood=bool(ood),
            teacher=mode, resolved_count=len(R), **extra,
        ),
    )


def generate_unkeyed(difficulty, rng, *, ood=False):
    return generate(difficulty, rng, ood=ood, keyed=False, track="developmental_production_unkeyed")