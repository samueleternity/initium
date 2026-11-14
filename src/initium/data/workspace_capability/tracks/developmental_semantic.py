"""Stage 3c: relational facts from the band graph (direct, 2-fact composition, distractors, NONE)."""

from __future__ import annotations

from initium.data.workspace_capability.episode import (
    BOS, EOS, EV_FACT, NONE_TOKEN, QUERY, SEP, TaskEpisode, entity_token as e, fact_token,
)
from initium.data.workspace_capability.tracks import _dev_lexicon as dl
from initium.data.workspace_capability.tracks import _dev_semantics as ds

TRACK = "developmental_semantic"


def generate(d, rng, *, ood=False):
    grp = rng.choice(tuple(d["groups"]))
    held = rng.random() < float(d["heldout"])
    band = "B" if held else "A"
    combos = int(d.get("combos", 0))
    cf = rng.random() < float(d["counterfactual"])
    nf = max(2, int(d["facts_a"]))
    nd = 0 if grp == "direct" else max(2 if grp == "distractor" else 0, int(d["distractors"]))
    keep, extra_rel = (), ()
    if grp == "compose2":
        h, r1, m, r2, t = ds.pick_chain(rng, band, combos)
        facts, keep = [(h, r1, m), (m, r2, t)], (0,)
        facts += ds.other_facts(rng, band, min(nf - 2 + nd, 6), {(h, r1), (m, r2)}, (h, r1) if nd else None)
    elif grp == "unsupported":
        facts = ds.other_facts(rng, band, min(nf + nd, 8), set())
        s = rng.choice(facts)[0]
        opts = [x for x in range(7) if x not in {f[1] for f in facts if f[0] == s}]
        r = rng.choice([x for x in opts if x in {f[1] for f in facts}] or opts)
        extra_rel = (r,)
    else:
        h, r, t = ds.pick_fact(rng, band, combos)
        facts = [(h, r, t)] + ds.other_facts(rng, band, min(nf - 1 + nd, 7), {(h, r)},
                                              (h, r) if grp == "distractor" else None)
    if cf:
        facts = ds.counterfactual(rng, band, facts, keep)
    wk, rk = ds.assign(rng, facts, extra_rel)
    if grp == "compose2":
        s, r1, r2, o = facts[0][0], facts[0][1], facts[1][1], facts[1][2]
        qrels = [rk[r1], rk[r2]]
    elif grp == "unsupported":
        o, qrels = None, [rk[r]]
    else:
        s, r, o = facts[0]
        qrels = [rk[r]]
    W = ds.lexicon(rng, {**{k: w for w, k in wk.items()}, **{k: r_ for r_, k in rk.items()}}, held)
    need = [wk[s], *qrels]
    stream = [BOS]
    R, mode = ds.teach(rng, stream, W, sorted(wk.values()), sorted(rk.values()), d, need)
    order = facts[:]
    rng.shuffle(order)
    ew = []
    for i, (h_, r_, t_) in enumerate(order[:8]):
        p = len(stream)
        stream += [EV_FACT, e(wk[h_]), e(rk[r_]), e(wk[t_]), SEP]
        ew += [(p + 1, 24 + i, e(wk[h_])), (p + 2, 48 + i, e(rk[r_])), (p + 3, 32 + i, e(wk[t_]))]
    stream += [QUERY, *(fact_token(W[k]) for k in need), SEP]
    ans = (NONE_TOKEN if grp == "unsupported" else e(wk[o]), EOS)
    extra = {"workspace_event_writes": tuple(ew)}
    if not dl.settings["unkeyed"]:
        n_, ck = len(ew) // 3, list(R)
        keys = ck + [24 + i for i in range(n_)] + [48 + i for i in range(n_)] + [32 + i for i in range(n_)]
        exp = [fact_token(R[k]) for k in ck] + [x[2] for x in ew[0::3]] + [x[2] for x in ew[1::3]] + [x[2] for x in ew[2::3]]
        extra["workspace_output_programs"] = ((0, "factmatch", tuple(keys), tuple(exp)),)
    qtype = f"{grp}_{mode}_{'cf' if cf else 'fact'}_{'heldout' if held else 'train'}"
    return TaskEpisode(TRACK, tuple(stream), ans,
                       ds.meta(TRACK, stream, grp, qtype, 100.0 / (len(wk) + (grp == "unsupported")),
                               held, ood, R, mode, cf=cf, band=band, nd=nd, **extra))