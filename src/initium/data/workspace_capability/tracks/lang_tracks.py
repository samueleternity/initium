"""Stage 4 generators: lang_template, lang_nlg, lang_story_qa, lang_dialogue, lang_syntax_probes, lang_open."""

from __future__ import annotations

import functools
import json
import random as _random

from initium.data.workspace_capability.episode import (
    BOOL_FALSE,
    BOOL_TRUE,
    FACT_BASE,
    MASK_TOKEN,
    SPK_SYS,
    SPK_USER,
    TASK_ACCEPT,
    TASK_CLOZE,
    TASK_D2T,
    TASK_FIRST,
    TASK_PARA,
    TASK_QA,
    TASK_RETELL,
    TASK_T2D,
    TURN_END,
    location_token,
    relation_token,
)
from initium.data.workspace_capability.tracks import _lang_text as lt
from initium.data.workspace_capability.tracks._lang_grammar import (
    DROP_V,
    MOVE_V,
    NAMES,
    OBJS,
    PICK_V,
    PLACES,
    PUNCT,
    ROWS,
    Ep,
    allowed,
    grammar,
    plural,
    row_ok,
)
from initium.data.workspace_capability.tracks.developmental import _pair_ok


def _toks(ws):
    tk = lt.tokenizer()
    return [t for w in ws for t in tk.word(w)]


def _canon_say(G, f, neg=False):
    return G.say(f, G.canon(f[1], G.pos[f[2]]), neg)


@functools.lru_cache(maxsize=8)
def _rows(name, split):
    path = lt.DATA_DIR / lt.EXTERNAL_FILES[name]
    if not path.exists():
        raise FileNotFoundError(f"{path} missing; build it with tools/build_stage4_corpus.py")
    with open(path, encoding="utf-8") as f:
        return tuple(
            r for r in map(json.loads, (l for l in f if l.strip())) if r.get("split") == split
        )


def _split(cfg):
    return "test" if cfg["heldout"] >= 1.0 else "train"


# ---------------------------------------------------------------- 4a template
def _alt_tail(G, band, r, h, t, rng):
    pool = [x for x in G.tails[band][r] if x not in (t, h)]
    return rng.choice(pool) if pool else t


def _world(cfg, rng, grp):
    G = grammar()
    heldout = rng.random() < cfg["heldout"]
    band = "B" if heldout else "A"
    facts = G.facts[band]
    pool = [f for f in facts if _pair_ok(f[0], f[1], cfg["combos"])] or facts
    if grp == "paraphrase":
        paraphrase_relations = (0, 2, 5, 6)

        def can_paraphrase(fact):
            target_pos = G.pos[fact[2]]
            rows_by_class = {}
            for row in ROWS:
                if row.rel == fact[1] and row_ok(row, target_pos) and allowed(row, cfg):
                    rows_by_class[row.cls] = rows_by_class.get(row.cls, 0) + 1
                    if rows_by_class[row.cls] >= 2:
                        return True
            return False

        preferred_pool = [f for f in pool if f[1] in paraphrase_relations]
        paraphrase_pool = [f for f in preferred_pool if can_paraphrase(f)]
        if not paraphrase_pool and preferred_pool:
            # Keep the generator viable when the preferred relation families
            # have no allowed surface-form pair at this lesson's difficulty.
            paraphrase_pool = [f for f in pool if can_paraphrase(f)]
        if not paraphrase_pool:
            raise OverflowError
        pool = paraphrase_pool
    f0 = rng.choice(pool)
    n = max(1, cfg["facts"]) + (grp == "unsupported")
    used, ctx = {(f0[0], f0[1])}, [f0]
    near = [f for f in facts if (f[0] == f0[0] or f[1] == f0[1]) and (f[0], f[1]) not in used]
    rng.shuffle(near)
    ns = set(near)
    rest = [f for f in facts if f not in ns and (f[0], f[1]) not in used]
    rng.shuffle(rest)
    d = cfg["distractors"]
    for f in near[:d] + rest + near[d:]:
        if len(ctx) >= n:
            break
        if (f[0], f[1]) not in used:
            used.add((f[0], f[1]))
            ctx.append(f)
    cf = rng.random() < cfg["counterfactual"]
    if cf:
        ctx = [(h, r, _alt_tail(G, band, r, h, t, rng)) for h, r, t in ctx]
        f0 = ctx[0]
    w = {
        "grp": grp,
        "band": band,
        "heldout": heldout,
        "cf": cf,
        "f0": f0,
        "yes": rng.random() < 0.5,
        "t2": None,
    }
    if grp == "unsupported":
        ctx = ctx[1:]
        if not ctx:
            raise OverflowError
    if grp == "qa_yn" and not w["yes"]:
        bad = set(G.hr[band].get((f0[0], f0[1]), ())) | {f0[0], f0[2]}
        c = [x for x in G.tails[band][f0[1]] if x not in bad]
        if not c:
            raise OverflowError
        w["t2"] = rng.choice(c)
    rng.shuffle(ctx)
    w["ctx"] = ctx
    return w


def _render(cfg, w, rng):
    G, grp, f0 = grammar(), w["grp"], w["f0"]
    s, r, t = f0
    pt = G.pos[t]
    cx = _world(cfg, rng, "qa_wh")["ctx"] if lt.settings["shuffled_context"] else w["ctx"]
    a = b = None
    if grp == "paraphrase":
        by = {}
        for x in ROWS:
            if x.rel == r and row_ok(x, pt) and allowed(x, cfg):
                by.setdefault(x.cls, []).append(x)
        multi = [v for v in by.values() if len(v) >= 2]
        if not multi:
            raise OverflowError
        rs = rng.choice(multi)
        a = rng.choice(rs)
        b = [x for x in rs if x is not a][0]
    sents = [G.say(f, a if (a is not None and f == f0) else G.row_for(f, cfg, rng)) for f in cx]
    out, i = [], 0
    while i < len(sents):
        if cfg["conj"] and i + 1 < len(sents) and rng.random() < 0.3:
            out.append(sents[i][:-1] + [sents[i][-1].rstrip(".")] + ["and"] + sents[i + 1])
            i += 2
        else:
            out.append(sents[i])
            i += 1
    seq = list(out)
    for _ in range(int(cfg.get("filler", 0))):
        seq.insert(
            rng.randrange(len(seq) + 1),
            (rng.choice(NAMES), rng.choice(MOVE_V), "to", "the", rng.choice(PLACES) + "."),
        )
    ep = Ep(0)
    for ws in seq:
        ep.text(ws, ctx=not isinstance(ws, tuple))
    sub = ""
    if grp == "qa_wh":
        ep.ctl(TASK_QA)
        ep.text(G.ask_wh(s, r), ctx=False)
        ans = _canon_say(G, f0)
    elif grp == "qa_yn":
        ep.ctl(TASK_QA)
        if w["yes"]:
            ep.text(G.ask_yn(s, r, t, pt), ctx=False)
            ans = ["yes,"] + _canon_say(G, f0)
        else:
            t2 = w["t2"]
            ep.text(G.ask_yn(s, r, t2, G.pos[t2]), ctx=False)
            ans = ["no,"] + _canon_say(G, (s, r, t2), True)
        sub = "_yes" if w["yes"] else "_no"
    elif grp == "retell":
        ep.ctl(TASK_RETELL)
        q = ["describe"] + G.np(s, "the")
        q[-1] += "."
        ep.text(q, ctx=False)
        ans = [x for f in w["ctx"] if f[0] == s for x in _canon_say(G, f)]
    elif grp == "paraphrase":
        ep.ctl(TASK_PARA)
        ep.text(G.say(f0, a), ctx=False)
        ans = G.say(f0, b)
    else:  # unsupported
        ep.ctl(TASK_QA)
        rng.choice(
            [
                x
                for x in range(7)
                if all((h, x) != (s, x) or h != s for h, rr, _ in w["ctx"] if rr == x)
            ]
            or [r]
        )
        if rng.random() < 0.5:
            ep.text(G.ask_wh(s, r), ctx=False)
        else:
            t2 = rng.choice(G.tails[w["band"]][r] or [t])
            ep.text(G.ask_yn(s, r, t2, G.pos[t2]), ctx=False)
        ans = ["i", "do", "not", "know."]
    if grp != "unsupported":
        ep.ask(f0)
    return ep.done(
        "lang_template",
        _toks(ans),
        f"{grp}{sub}_{'cf' if w['cf'] else 'fact'}_{'heldout' if w['heldout'] else 'train'}",
        grp,
        grammar_checked=True,
        counterfactual=w["cf"],
        heldout_words=w["heldout"],
    )


def _g_template(cfg, rng):
    grp = lt.sample_group(cfg, rng)
    return _render(cfg, _world(cfg, rng, grp), rng)


def template_pair(cfg, rng):
    """Two episodes over the same world/question with different surface realisations of the context."""
    w = _world(cfg, rng, rng.choice(("qa_wh", "qa_yn")))
    return tuple(_render(cfg, w, _random.Random(rng.getrandbits(32))) for _ in range(2))


# ---------------------------------------------------------------- 4b nlg
def _ext_nlg(cfg, rng):
    rec = rng.choice(_rows("nlg", _split(cfg)))
    ep = Ep(1, annotate=False)
    ep.ctl(TASK_D2T)
    for s, r, o in rec["triples"]:
        ep.text(s.split() + ["|"] + r.split() + ["|"] + o.split() + [";"])
    return ep.done(
        "lang_nlg", _toks(rec["text"].split()), "external_nlg", "external", grammar_checked=False
    )


def _g_nlg(cfg, rng):
    grp = lt.sample_group(cfg, rng)
    if grp == "external":
        return _ext_nlg(cfg, rng)
    G = grammar()
    w = _world(cfg, rng, "qa_wh")
    ctx = w["ctx"]
    ep = Ep(1)
    meta = dict(counterfactual=w["cf"], heldout_words=w["heldout"])
    if grp == "d2t":
        ep.ctl(TASK_D2T)
        for f in ctx:
            ep.typed(f)
        ans = [x for f in ctx for x in _canon_say(G, f)]
        return ep.done("lang_nlg", _toks(ans), "d2t", grp, grammar_checked=True, **meta)
    for f in ctx:
        ep.text(G.say(f, G.row_for(f, w_cfg(cfg), rng)))
    if grp == "t2d":
        ep.ctl(TASK_T2D)
        ans = [x for f in ctx for x in (FACT_BASE + f[0], relation_token(f[1]), FACT_BASE + f[2])]
        return ep.done("lang_nlg", ans, "t2d", grp, kind="typed", **meta)
    ep.ctl(TASK_RETELL)
    ep.text(["tell", "me", "everything."], ctx=False)
    ans = [x for f in sorted(ctx, key=lambda f: (f[0], f[1])) for x in _canon_say(G, f)]
    return ep.done("lang_nlg", _toks(ans), "retell_all", grp, grammar_checked=True, **meta)


def w_cfg(cfg):
    return cfg


# ---------------------------------------------------------------- 4c story QA
def _ext_story(cfg, rng):
    rec = rng.choice(_rows("story", _split(cfg)))
    ep = Ep(2, annotate=False)
    ep.text(rec["context"].split())
    ep.ctl(TASK_QA)
    ep.text(rec["question"].split())
    return ep.done(
        "lang_story_qa",
        _toks(rec["answer"].split()),
        "external_story",
        "external",
        grammar_checked=False,
    )


def _is_in(p, l):
    return [NAMES[p], "is", "in", "the", PLACES[l] + "."]


def _obj_in(o, l):
    return ["the", OBJS[o], "is", "in", "the", PLACES[l] + "."]


def _g_story(cfg, rng):
    grp = lt.sample_group(cfg, rng)
    if grp == "external":
        return _ext_story(cfg, rng)
    ep = Ep(2)
    if grp in ("false_belief", "true_belief", "reality"):
        p1, p2 = rng.sample(range(6), 2)
        o = rng.randrange(6)
        l1, l2, l3 = rng.sample(range(6), 3)
        for s in (_is_in(p1, l1), _is_in(p2, l1), _obj_in(o, l1)):
            ep.story(s)
        mp = [NAMES[p2], rng.choice(MOVE_V), "to", "the", PLACES[l2] + "."]
        mo = [NAMES[p1], "moved", "the", OBJS[o], "to", "the", PLACES[l3] + "."]
        for s in [mp, mo] if grp != "true_belief" else [mo, mp]:
            ep.story(s)
        ep.ctl(TASK_QA)
        if grp == "reality":
            q, a = ["where", "is", "the", OBJS[o] + "?"], _obj_in(o, l3)
        else:
            q = ["where", "will", NAMES[p2], "look", "for", "the", OBJS[o] + "?"]
            a = [
                NAMES[p2],
                "will",
                "look",
                "in",
                "the",
                PLACES[l1 if grp == "false_belief" else l3] + ".",
            ]
        ep.story(q, ctx=False)
        return ep.done("lang_story_qa", _toks(a), grp, grp, grammar_checked=False)
    persons = rng.sample(range(6), rng.randint(2, 3))
    objs = rng.sample(range(6), 2)
    pl, ol, held = {}, {}, {}
    for p in persons:
        pl[p] = rng.randrange(6)
        ep.story(_is_in(p, pl[p]))
    for o in objs:
        ol[o] = rng.randrange(6)
        ep.story(_obj_in(o, ol[o]))
    for _ in range(cfg["events"]):
        kind, p = rng.choice(("move", "move", "pick", "drop")), rng.choice(persons)
        if kind == "pick":
            c = [o for o in objs if o not in held and ol[o] == pl[p]]
            if c and p not in held.values():
                o = rng.choice(c)
                held[o] = p
                ep.story([NAMES[p]] + rng.choice(PICK_V).split() + ["the", OBJS[o] + "."])
                continue
        elif kind == "drop":
            hs = [o for o, hp in held.items() if hp == p]
            if hs:
                o = hs[0]
                del held[o]
                ol[o] = pl[p]
                ep.story([NAMES[p]] + rng.choice(DROP_V).split() + ["the", OBJS[o] + "."])
                continue
        l = rng.choice([x for x in range(6) if x != pl[p]])
        pl[p] = l
        for o, hp in held.items():
            if hp == p:
                ol[o] = l
        ep.story([NAMES[p], rng.choice(MOVE_V), "to", "the", PLACES[l] + "."])
    objloc = lambda o: pl[held[o]] if o in held else ol[o]  # noqa: E731
    p, o = rng.choice(persons), rng.choice(objs)
    if grp == "where_person":
        q, a = ["where", "is", NAMES[p] + "?"], _is_in(p, pl[p])
    elif grp == "where_object":
        q, a = ["where", "is", "the", OBJS[o] + "?"], _obj_in(o, objloc(o))
    elif grp == "who_has":
        q = ["who", "has", "the", OBJS[o] + "?"]
        a = [NAMES[held[o]] if o in held else "nobody", "has", "the", OBJS[o] + "."]
    elif grp == "yes_no":
        yes = rng.random() < 0.5
        l = pl[p] if yes else rng.choice([x for x in range(6) if x != pl[p]])
        q = ["is", NAMES[p], "in", "the", PLACES[l] + "?"]
        a = (
            ["yes,", NAMES[p], "is", "in", "the", PLACES[l] + "."]
            if yes
            else ["no,", NAMES[p], "is", "not", "in", "the", PLACES[l] + "."]
        )
    else:  # carry
        hs = [x for x, hp in held.items() if hp == p]
        q = ["what", "is", NAMES[p], "carrying?"]
        a = (
            [NAMES[p], "is", "carrying", "the", OBJS[hs[0]] + "."]
            if hs
            else [NAMES[p], "is", "carrying", "nothing."]
        )
    ep.ctl(TASK_QA)
    ep.story(q, ctx=False)
    if grp == "yes_no" and ep.ann:
        # The utterance names both a person and a candidate place. Give the
        # workspace an explicit causal read target for the person's current
        # location, so the model can learn the comparison before verbalizing
        # the yes/no plus rationale. This does not put the answer in the input.
        ep.rd.append((len(ep.t) - 1, p, location_token(pl[p])))
    return ep.done(
        "lang_story_qa", _toks(a), f"{grp}_ev{cfg['events']}", grp, grammar_checked=False
    )


# ---------------------------------------------------------------- 4d dialogue
def _turn(G, ctx, kind, prev, rng):
    if kind == "pronoun" and prev:
        rs = [r for (h, r) in ctx if h == prev[0] and r != prev[1]]
        if not rs:
            return None
        s, r = prev[0], rng.choice(rs)
        t = ctx[(s, r)]
        return G.ask_wh(s, r, pron=True), _canon_say(G, (s, r, t)), (s, r), (s, r, t)
    if kind == "ellipsis" and prev:
        ss = [h for (h, r) in ctx if r == prev[1] and h != prev[0]]
        if not ss:
            return None
        s, r = rng.choice(ss), prev[1]
        t = ctx[(s, r)]
        q = (["and"] if rng.random() < 0.5 else ["no,"]) + G.np(s, "the")
        q[-1] += "?" if q[0] == "and" else "."
        return q, _canon_say(G, (s, r, t)), (s, r), (s, r, t)
    if kind == "unsupported":
        s = rng.choice(sorted({h for h, _ in ctx}))
        rs = [r for r in range(7) if (s, r) not in ctx]
        if not rs:
            return None
        r = rng.choice(rs)
        return G.ask_wh(s, r), ["i", "do", "not", "know."], (s, r), None
    if kind == "fresh":
        (s, r), t = rng.choice(sorted(ctx.items()))
        return G.ask_wh(s, r), _canon_say(G, (s, r, t)), (s, r), (s, r, t)
    return None


def _g_dialogue(cfg, rng):
    G = grammar()
    grp = lt.sample_group(cfg, rng)
    band = "B" if rng.random() < cfg["heldout"] else "A"
    hr = G.hr[band]
    heads = {}
    for h, r in hr:
        heads.setdefault(h, []).append(r)
    multi = [h for h, rs in heads.items() if len(rs) >= 2]
    if len(multi) < 2:
        raise RuntimeError("band lacks heads with >=2 relations for dialogue")
    topics = rng.sample(multi, 2)
    ctx = {}
    for h in topics:
        for r in rng.sample(heads[h], min(2 if cfg["turns"] < 2 else 3, len(heads[h]))):
            ctx[(h, r)] = rng.choice(hr[(h, r)])
    others = [k for k in hr if k[0] not in topics]
    rng.shuffle(others)
    for k in others:
        if len(ctx) >= cfg["facts"]:
            break
        ctx[k] = rng.choice(hr[k])
    cf = rng.random() < cfg["counterfactual"]
    if cf:
        ctx = {(h, r): _alt_tail(G, band, r, h, t, rng) for (h, r), t in ctx.items()}
    T = (
        rng.randint(2, max(2, cfg["turns"]))
        if grp in ("pronoun", "ellipsis")
        else rng.randint(1, max(1, cfg["turns"]))
    )
    c2 = {**cfg, "plural": 0}
    facts = [(h, r, t) for (h, r), t in ctx.items()]
    rng.shuffle(facts)
    ep = Ep(3)
    for f in facts:
        ep.text(G.say(f, G.row_for(f, c2, rng)))
    prev, turns = None, []
    for k in range(T):
        last = k == T - 1
        res = _turn(
            G,
            ctx,
            grp if last else rng.choice(("fresh", "fresh", "pronoun", "ellipsis", "unsupported")),
            prev,
            rng,
        )
        if res is None:
            if last:
                raise OverflowError
            res = _turn(G, ctx, "fresh", prev, rng)
        turns.append(res)
        prev = res[2]
    for q, a, _, _ in turns[:-1]:
        ep.ctl(SPK_USER)
        ep.text(q, ctx=False)
        ep.ctl(SPK_SYS)
        ep.text(a, ctx=False)
        ep.ctl(TURN_END)
    q, a, _, fact = turns[-1]
    ep.ctl(SPK_USER)
    ep.text(q, ctx=False)
    ep.ctl(SPK_SYS)
    if fact is not None:
        ep.ask(fact)
    return ep.done(
        "lang_dialogue",
        _toks(a),
        f"{grp}_turn{T}",
        grp,
        grammar_checked=True,
        counterfactual=cf,
        heldout_words=band == "B",
        turn=T,
    )


# ---------------------------------------------------------------- 4e syntax probes
TRAIN_PHEN, HELD_PHEN = ("agree", "detnum", "order"), ("article", "aux")
_AGREE = {
    "is": "are",
    "are": "is",
    "was": "were",
    "were": "was",
    "has": "have",
    "have": "has",
    "causes": "cause",
    "cause": "causes",
}
_AUX = {
    "is": "has",
    "are": "can",
    "was": "has",
    "were": "can",
    "can": "is",
    "could": "is",
    "has": "is",
    "have": "are",
}
_ART = {"a": "an", "an": "a"}


def _swap(ws, mp):
    for i, w in enumerate(ws):
        c = w.strip(PUNCT)
        if c in mp:
            out = list(ws)
            out[i] = w.replace(c, mp[c])
            return out
    return None


def _corrupt(G, ws, kind, rng):
    if kind == "agree":
        return _swap(ws, _AGREE)
    if kind == "aux":
        return _swap(ws, _AUX)
    if kind == "article":
        return _swap(ws, _ART)
    core = [w.strip(PUNCT) for w in ws]
    if kind == "detnum":
        M, _ = G.parse(ws)
        if not M:
            return None
        st, we, idx, _, _ = M[0]
        out = core[:st] + ["a", plural(G.word[idx])] + core[we + 1 :]
    else:  # order
        if len(core) < 3:
            return None
        i = rng.randrange(len(core) - 1)
        out = list(core)
        out[i], out[i + 1] = out[i + 1], out[i]
    out[-1] += ws[-1][len(ws[-1].rstrip(PUNCT)) :]
    return out


def _g_probe(cfg, rng):
    G = grammar()
    grp = lt.sample_group(cfg, rng)
    if grp == "external":
        rec = rng.choice(_rows("accept", _split(cfg)))
        ep = Ep(4, annotate=False)
        ep.ctl(TASK_ACCEPT)
        ep.text(rec["sentence"].split())
        ep.mark(0, BOOL_TRUE if int(rec["label"]) else BOOL_FALSE)
        return ep.done(
            "lang_syntax_probes",
            [BOOL_TRUE if int(rec["label"]) else BOOL_FALSE],
            "external",
            "external",
            kind="bool",
        )
    band = "B" if rng.random() < cfg["heldout"] * 0 else "A"
    for _ in range(20):
        f = rng.choice(G.facts[band])
        ws = G.say(f, G.row_for(f, cfg, rng))
        if grp == "good":
            label, name = True, "good"
            break
        name = rng.choice(HELD_PHEN) if cfg["heldout"] >= 1.0 else grp
        bad = _corrupt(G, ws, name, rng)
        if bad is not None and bad != ws and not G.sentence_ok(bad):
            ws, label = bad, False
            break
    else:
        raise OverflowError
    ep = Ep(4, annotate=False)
    ep.ctl(TASK_ACCEPT)
    ep.text(ws, ctx=False)
    ep.mark(0, BOOL_TRUE if label else BOOL_FALSE)
    return ep.done(
        "lang_syntax_probes", [BOOL_TRUE if label else BOOL_FALSE], name, name, kind="bool"
    )


# ---------------------------------------------------------------- 4f open (external, exploratory)
_STOP = frozenset(
    "that this with from have were been they their there which would about into than then them".split()
)


def _g_open(cfg, rng):
    grp = lt.sample_group(cfg, rng)
    rec = rng.choice(_rows("open", _split(cfg)))
    words = rec["text"].split()[: cfg["words"]]
    ep = Ep(5)
    if grp == "cloze":
        c = [
            i
            for i, w in enumerate(words)
            if w.strip(PUNCT).isalpha() and len(w.strip(PUNCT)) >= 4 and w.strip(PUNCT) not in _STOP
        ]
        if not c:
            raise OverflowError
        i = rng.choice(c)
        ep.open_text(words[:i], _STOP)
        ep.ctl(MASK_TOKEN)
        ep.open_text(words[i + 1 :], _STOP)
        ep.open_reads()
        ep.ctl(TASK_CLOZE)
        return ep.done(
            "lang_open", _toks([words[i].strip(PUNCT)]), "cloze", grp, grammar_checked=False
        )
    end = next((i for i, w in enumerate(words) if w[-1:] in ".?!"), None)
    if end is None or end >= 40 or end + 1 >= len(words):
        raise OverflowError
    ep.open_text(words[end + 1 :], _STOP)
    ep.open_reads()
    ep.ctl(TASK_FIRST)
    return ep.done("lang_open", _toks(words[: end + 1]), "first", grp, grammar_checked=False)


GENERATORS = {
    "lang_template": _g_template,
    "lang_nlg": _g_nlg,
    "lang_story_qa": _g_story,
    "lang_dialogue": _g_dialogue,
    "lang_syntax_probes": _g_probe,
    "lang_open": _g_open,
}
