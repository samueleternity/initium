"""Stage 3 sampling: band-graph facts, counterfactual tails, word teaching, metadata."""

from __future__ import annotations

from initium.data.workspace_capability.episode import EV_KV, SEP, entity_token as e, fact_token
from initium.data.workspace_capability.tracks import _dev_lexicon as dl
from initium.data.workspace_capability.tracks._shared import common_metadata
from initium.data.workspace_capability.tracks.developmental import _cross, _pair_ok, _teach


def pick_fact(rng, band, combos):
    L = dl.load()
    keys = list(L.pairs[band])
    h, r = rng.choice([p for p in keys if _pair_ok(p[0], p[1], combos)] or keys)
    return h, r, rng.choice(L.pairs[band][(h, r)])


def pick_chain(rng, band, combos):
    ch = dl.load().chains[band]
    return rng.choice([c for c in ch if _pair_ok(c[1], c[3], combos)] or ch)


def other_facts(rng, band, n, avoid, near=None):
    """n facts with unique (head, relation) not in `avoid`; `near` biases toward near-misses."""
    L = dl.load()
    pool = [p for p in L.pairs[band] if p not in avoid]
    rng.shuffle(pool)
    if near is not None and rng.random() < 0.7:
        pool.sort(key=lambda p: not (p[0] == near[0] or p[1] == near[1]))
    return [(h, r, rng.choice(L.pairs[band][(h, r)])) for h, r in pool[: max(0, n)]]


def counterfactual(rng, band, facts, keep=()):
    """Resample tails from the same relation's tail pool (context must beat world knowledge)."""
    L = dl.load()
    out = []
    for i, (h, r, t) in enumerate(facts):
        alt = [x for x in L.tails_by_rel[band][r] if x not in (t, h)]
        out.append((h, r, t if i in keep or not alt else rng.choice(alt)))
    return out


def assign(rng, facts, extra_rels=()):
    words = sorted({w for h, _, t in facts for w in (h, t)})
    rels = sorted({r for _, r, _ in facts} | set(extra_rels))
    return (dict(zip(words, rng.sample(range(24), len(words)))),
            dict(zip(rels, rng.sample(range(56, 64), len(rels)))))


def lexicon(rng, W, held):
    """stable: keep real word ids; episodic: arbitrary Stage-2 style ids (calibration arm)."""
    if dl.settings["lexicon_mode"] == "episodic":
        return dict(zip(W, rng.sample(range(64, 128) if held else range(64), len(W))))
    return W


def teach(rng, stream, W, objs, rels, d, need):
    amb, scenes = max(2, int(d["ambiguity"])), min(max(2, int(d["scenes"])), 8)
    if rng.random() < float(d["ambiguity_prob"]):
        R, mode = _cross(rng, stream, [(objs, 1, amb), (rels, 1, 1)], W, scenes), "ambiguous"
    else:
        R, mode = _teach(rng, stream, "direct", W, 0.0), "direct"
    for k in need:
        if k not in R:
            stream += [EV_KV, e(k), fact_token(W[k]), SEP]
            R[k] = W[k]
    return R, mode


def meta(track, stream, group, qtype, base, held, ood, R, mode, *, cf=False, band="A", nd=0, reads=(), **extra):
    last = len(stream) - 1
    return common_metadata(
        track=track, sequence_length=len(stream), query_type=qtype, relevant_updates=len(R),
        distractor_count=nd, candidate_baseline_pct=base, progression_group=group,
        heldout_words=held, ood=bool(ood), teacher=mode, resolved_count=len(R), stage3=True,
        stage3_words=dl.settings["lexicon_mode"] == "stable", lexicon_mode=dl.settings["lexicon_mode"],
        band=band, counterfactual=cf,
        workspace_query_reads=tuple((last, k, fact_token(w)) for k, w in R.items()) + tuple(reads),
        lexicon=tuple((k, fact_token(w)) for k, w in R.items()), **extra,
    )