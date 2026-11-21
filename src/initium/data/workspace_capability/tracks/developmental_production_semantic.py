"""Stage 3d: structured production. Stride-3 [W(subj)][W(rel)][W(obj)] ... EOS from the band graph."""

from __future__ import annotations

from initium.data.workspace_capability.episode import (
    BOS, EOS, EV_FACT, EV_SUPERSEDE, QUERY, SEP, TaskEpisode, entity_token as e, fact_token,
)
from initium.data.workspace_capability.tracks import _dev_lexicon as dl
from initium.data.workspace_capability.tracks import _dev_semantics as ds
from initium.data.workspace_capability.tracks.developmental import _pair_ok

TRACK = "developmental_production_semantic"


def generate(d, rng, *, ood=False, keyed=True, track=TRACK):
    grp = rng.choice(tuple(d["groups"]))
    k = {"p_entity": 1, "p_comp": 1, "p_all2": 2}.get(grp) or rng.choice((3, 4))
    held = rng.random() < float(d["heldout"])
    band = "B" if held else "A"
    combos = int(d.get("combos", 0))
    L = dl.load()
    cf = rng.random() < float(d["counterfactual"])
    change = rng.random() < float(d.get("state_change", 0.0))
    nd = rng.randint(0, int(d["distractors"]))
    subs = [h for h, rs in L.heads[band].items() if len(rs) >= k]
    if not subs:
        raise ValueError(f"band {band} has no subject with {k} distinct relations; rebuild the lexicon")
    X = rng.choice(subs)
    rs = list(L.heads[band][X])
    rng.shuffle(rs)
    rs.sort(key=lambda r: not _pair_ok(X, r, combos))
    facts = [(X, r, rng.choice(L.heads[band][X][r])) for r in rs[:k]]
    facts += ds.other_facts(rng, band, nd, {(X, r) for r in L.heads[band][X]})
    if cf:
        facts = ds.counterfactual(rng, band, facts)
    upd = None
    if change:  # state change: re-assert one of X's facts with a new object; describe CURRENT only
        i = rng.randrange(k)
        alt = [x for x in L.tails_by_rel[band][facts[i][1]] if x not in (facts[i][2], X)]
        if alt:
            upd = (i, rng.choice(alt))
    allf = facts + ([(X, facts[upd[0]][1], upd[1])] if upd else [])
    wk, rk = ds.assign(rng, allf)
    cur, order = list(facts[:k]), list(range(k))
    rng.shuffle(order)
    if upd:
        cur[upd[0]] = allf[-1]
        order.remove(upd[0])
        order.append(upd[0])  # temporal order: updated fact is most recent
    keys_out = []
    for i in order:
        h_, r_, o_ = cur[i]
        keys_out += [wk[h_], rk[r_], wk[o_]]
    W = ds.lexicon(rng, {**{kk: w for w, kk in wk.items()}, **{kk: r for r, kk in rk.items()}}, held)
    need = [wk[X]] if grp == "p_comp" else sorted(set(keys_out))
    stream = [BOS]
    R, mode = ds.teach(rng, stream, W, sorted(wk.values()), sorted(rk.values()), d, need)
    emit = facts[:]
    rng.shuffle(emit)
    if upd:
        emit.append(allf[-1])
    for h_, r_, t_ in emit:
        stream += [EV_FACT, e(wk[h_]), e(rk[r_]), e(wk[t_]), SEP]
    extra = {}
    if grp == "p_comp":  # comprehension counterpart: word -> concept (same world/seed as p_entity)
        stream += [QUERY, fact_token(W[wk[X]]), SEP]
        ans, stride, qtype = [e(wk[X])], 0, "comprehend_entity"
        extra["workspace_output_programs"] = ((0, "argmatch", tuple(R), tuple(fact_token(w) for w in R.values())),)
    else:
        stream += [QUERY, *(() if grp == "p_entity" else (EV_SUPERSEDE,)), e(wk[X]), SEP]
        ans, stride = [fact_token(W[kk]) for kk in keys_out], 3
        qtype = "describe_entity" if k == 1 else f"describe_all_k{k}"
        if keyed and not dl.settings["unkeyed"]:
            extra["workspace_output_keys"] = tuple(keys_out) + (-1,)
    qtype = f"{qtype}_{mode}_{'cf' if cf else 'fact'}_{'heldout' if held else 'train'}"
    return TaskEpisode(track, tuple(stream), tuple(ans) + (EOS,),
                       ds.meta(track, stream, grp, qtype, 100.0 / len(R), held, ood, R, mode, cf=cf,
                               band=band, nd=nd, answer_stride=stride, answer_concepts=tuple(keys_out),
                               **extra))


def generate_unkeyed(d, rng, *, ood=False):
    return generate(d, rng, ood=ood, keyed=False, track="developmental_production_semantic_unkeyed")