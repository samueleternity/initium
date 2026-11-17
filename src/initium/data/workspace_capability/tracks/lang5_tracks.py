"""Stage 5 generators: lang_parse/define/induce/logic/discourse/act/nlu_external (ROADMAP stage 5, A2).

Multi-turn episodes: the scored (final) segment is the episode; earlier segments live in metadata["prev"].
All slot labels come from generator identity (or the text-only annotator when identity='annotator').
"""

from __future__ import annotations

from contextlib import contextmanager

from initium.data.workspace_capability.episode import (
    MARK_O, MARK_R, SEP, SPK_SYS, SPK_USER, TASK_QA, TURN_END,
    entity_token, location_token, relation_token,
)
from initium.data.workspace_capability.tracks import _lang_grammar as lg
from initium.data.workspace_capability.tracks import _lang_text as lt
from initium.data.workspace_capability.tracks.lang_tracks import _canon_say, _rows, _split, _toks, _world

TASK_MR, TASK_S2MR, TASK_MR2S, TASK_LOGIC, TASK_NLU, NEG_TOK, PLU_TOK = range(351, 358)
KEYS_D, PSEUDO, REL_ALL, REL_NO = 48, 100_000, 14, 15  # keys 0-47 discourse; 48-79 S; 80-111 O; 112-127 R
REL_SOME, REL_IF, SOME_TOK, IF_TOK = 16, 17, 358, 359  # relation ids 0-15 are full; extras map to spare control tokens


def _rel_tok(v):
    return relation_token(v) if v < 16 else (SOME_TOK, IF_TOK)[v - 16]

FEM = frozenset(("mary", "sandra", "emma"))
_NUM = ("zero", "one", "two", "three", "four", "five")
_ORD = ("svo", "sov", "vso")
ACTS = ("a_inform", "a_ask", "a_confirm", "a_correct", "a_social", "a_clarify", "a_refuse")


class Ep5(lg.Ep):
    """Stage-4 builder with 128-key layout, first-piece slot targets, LRU key recycling, forkable segments."""

    def __init__(self, marker, annotate=True):
        super().__init__(marker, annotate)
        self.nc = self.clock = self.evict = self.evc = 0
        self.binding_aux: list[tuple[int, int]] = []
        self.closed = set()  # entities already answered about and not re-mentioned: recycled first
        self.last = {}
        self.truth = lt.settings["identity"] == "truth"

    def fork(self, marker):
        n = Ep5(marker, self.ann)
        n.ent, n.last, n.clock, n.cl, n.nc, n.evict = self.ent, self.last, self.clock, self.cl, self.nc, self.evict
        n.closed, n.evc = self.closed, self.evc
        return n

    def piece(self, x):
        return self.tk.word(x if isinstance(x, str) else self.G.word[x])[0]

    def _key(self, name, write):
        self.clock += 1
        k = self.ent.get(name)
        if k is None:
            if not write:
                return None
            if len(self.ent) < KEYS_D:
                k = len(self.ent)
            else:  # recycle least-recently-used entity key
                closed = [n for n in self.ent if n in self.closed]
                old = min(closed or self.ent, key=self.last.__getitem__)
                self.closed.discard(old)
                self.evc += bool(closed)
                k = self.ent.pop(old)
                del self.last[old]
                self.evict += 1
            self.ent[name] = k
        self.last[name] = self.clock
        self.closed.discard(name)
        return k

    def _commit(self, s, r, o, neg=False):
        i = self.nc % 16
        self.nc += 1
        p = len(self.t)
        self.t.extend((lg.MARK_S, lg.MARK_R, lg.MARK_O))
        self.addr.extend(((p, 48 + i, True), (p + 1, 112 + i, True), (p + 2, 80 + i, True)))
        self.evw.extend(((p, 48 + i, self.piece(s)), (p + 1, 112 + i, _rel_tok(r + 7 * neg)),
                         (p + 2, 80 + i, self.piece(o))))
        if len(self.cl) < 16:
            self.cl.append((s, r, o))
        else:
            self.cl[i] = (s, r, o)

    def ask(self, fact):
        if not self.ann:
            return
        s, r, o = fact
        for i in range(len(self.cl) - 1, -1, -1):
            if self.cl[i] == (s, r, o):
                st = len(self.t) - 1
                self.rd.extend(((st, 48 + i, self.piece(s)), (st, 112 + i, relation_token(r)), (st, 80 + i, self.piece(o))))
                return

    def text(self, ws, *, ctx=True):
        sp = self._enc(ws)
        if not self.ann:
            return
        M, clauses = self.G.parse(ws)
        for _s, we, idx, _p, _d in M:
            k = self._key(idx, ctx)
            if k is None:
                continue
            pos = self._pos(ws, sp, we)
            self.addr.append((pos, k, ctx))
            if ctx:
                self.evw.append((pos, k, self.piece(idx)))
        if ctx:
            for s, r, o, neg in clauses:
                self._commit(s, r, o, neg)

    def manual(self, ws, surf, *, ctx=True):
        """Generator-labelled mentions: surf maps surface -> (entity name, base surface)."""
        sp = self._enc(ws)
        if not self.ann:
            return
        for i, w in enumerate(ws):
            m = surf.get(w.strip(lg.PUNCT))
            k = self._key(m[0], ctx) if m else None
            if k is None:
                continue
            pos = self._pos(ws, sp, i)
            self.addr.append((pos, k, ctx))
            if ctx:
                self.evw.append((pos, k, self.piece(m[1])))

    def bind_surface(self, ws, bindings, *, write=True):
        """Attach an oracle concept key and typed value to each mapped surface word."""
        sp = self._enc(ws)
        if not self.ann:
            return
        for i, word in enumerate(ws):
            binding = bindings.get(word.strip(lg.PUNCT))
            if binding is None:
                continue
            key, target = binding
            pos = self._pos(ws, sp, i)
            self.addr.append((pos, int(key), bool(write)))
            if write:
                self.evw.append((pos, int(key), int(target)))
            else:
                self.binding_aux.append((pos, int(target)))

    def bind_mr(self, values, bindings, *, write=True):
        """Append typed MR tokens and annotate their reverse lexical bindings."""
        for value in values:
            pos = len(self.t)
            self.t.append(int(value))
            binding = bindings.get(int(value))
            if binding is None or not self.ann:
                continue
            key, target = binding
            self.addr.append((pos, int(key), bool(write)))
            if write:
                self.evw.append((pos, int(key), int(target)))
            else:
                self.binding_aux.append((pos, int(target)))

    def output_keys(self, words, key_by_word):
        """Expand word-level semantic keys to answer-token positions."""
        keys = []
        for word in words:
            pieces = self.tk.word(word)
            key = int(key_by_word.get(word.strip(lg.PUNCT), -1))
            keys.extend((key,) + (-1,) * (len(pieces) - 1))
        return tuple(keys) + (-1,)

    def story_ws(self, ws, ments):
        """ments: (word_idx, name, loc|None, is_pronoun); loc -> slot write of location_token at that word."""
        sp = self._enc(ws)
        if not self.ann:
            return
        for i, name, loc, pron in ments:
            if pron and not self.truth:
                continue
            k = self._key(name, True)
            if k is None:
                continue
            pos = self._pos(ws, sp, i)
            self.addr.append((pos, k, loc is not None))
            if loc is not None:
                self.evw.append((pos, k, location_token(loc)))

    def upd(self, name, loc):
        """State-update placeholder token (input-only) with a slot write."""
        p = len(self.t)
        self.t.append(lg.MARK_O)
        k = self._key(name, True) if self.ann else None
        if k is not None:
            self.addr.append((p, k, True))
            self.evw.append((p, k, location_token(loc)))

    def read_slot(self, name, tgt):
        k = self.ent.get(name)
        if self.ann and k is not None:
            self.rd.append((len(self.t) - 1, k, tgt))
            self.closed.add(name)

    def done(self, track, answer, qtype, group, kind="text", **meta):
        if self.binding_aux:
            meta.setdefault("workspace_binding_aux", tuple(self.binding_aux))
        return super().done(track, answer, qtype, group, kind, evictions=self.evict, closed_evictions=self.evc, **meta)


def pseudo(rng, n):
    G, out = lg.grammar(), []
    while len(out) < n:
        w = "".join(rng.choice("bdfgklmnprstvz") + rng.choice("aeiou") for _ in range(rng.randint(2, 3)))
        w += rng.choice(("", "n", "r", "l", "m"))
        if w not in out and w not in G.surf and w not in lg.FUNCTION:
            out.append(w)
    return out


@contextmanager
def _pseudo_nouns(G, words):
    ids = []
    try:
        for i, w in enumerate(words):
            j = PSEUDO + i
            G.word[j], G.pos[j] = w, "n"
            G.surf[w], G.surf[lg.plural(w)] = (j, False), (j, True)
            ids.append(j)
        yield ids
    finally:
        for j, w in zip(ids, words):
            G.word.pop(j, None), G.pos.pop(j, None), G.surf.pop(w, None), G.surf.pop(lg.plural(w), None)


# ---------------------------------------------------------------- 5a parse
def _g_parse(cfg, rng):
    G, grp = lg.grammar(), rng.choice(cfg["groups"])
    w = _world(cfg, rng, "qa_wh")
    items = []
    for f in w["ctx"][: max(1, cfg["facts"])]:
        neg = grp in ("parse_neg", "parse_ctx") and rng.random() < 0.4
        if neg:
            bad = [x for x in G.tails[w["band"]][f[1]] if x not in (f[0], f[2])]
            if bad:
                f = (f[0], f[1], rng.choice(bad))
            else:
                neg = False
        items.append((f, neg))
    sents = [G.say(f, G.row_for(f, cfg, rng), neg) for f, neg in items]
    out, i = [], 0
    while i < len(sents):
        if grp in ("parse_conj", "parse_ctx") and cfg["conj"] and i + 1 < len(sents) and rng.random() < 0.5:
            out.append(sents[i][:-1] + [sents[i][-1].rstrip(".")] + ["and"] + sents[i + 1])
            i += 2
        else:
            out.append(sents[i])
            i += 1
    ep = Ep5(6)
    for ws in out:
        ep.text(ws)
    ep.ctl(TASK_MR)
    ans = [t for (h, r, o), neg in items for t in (ep.piece(h), relation_token(r + 7 * neg), ep.piece(o))]
    return ep.done("lang_parse", ans, grp, grp, kind="mr", grammar_checked=True,
                   counterfactual=w["cf"], heldout_words=w["heldout"])


# ---------------------------------------------------------------- 5b define
def _g_define(cfg, rng):
    G, grp = lg.grammar(), rng.choice(cfg["groups"])
    band = "B" if rng.random() < cfg["heldout"] else "A"
    pool = G.tails[band][0]
    props = G.tails[band][1]
    n = max(1, cfg["pseudo"])

    if len(pool) < n + 2 or (grp == "def_prop" and not props):
        raise OverflowError
    cats, W = rng.sample(pool, n + 1), pseudo(rng, n + 1)
    with _pseudo_nouns(G, W) as ids:
        facts = [(ids[i], 0, cats[i]) for i in range(n)]
        if grp == "def_prop":
            facts.append((ids[0], 1, rng.choice(props)))
        sents = [G.say(f, G.canon(f[1], G.pos[f[2]])) for f in facts]
        for f in rng.sample(G.facts[band], min(len(G.facts[band]), cfg["distractors"])):
            sents.append(G.say(f, G.row_for(f, cfg, rng)))
        rng.shuffle(sents)
        target = facts[-1] if grp == "def_prop" else facts[0]
        if grp == "def_redef":
            target = (ids[0], 0, rng.choice([c for c in pool if c != cats[0]]))
            sents.append(["no,"] + G.say(target, G.canon(0, "n")))
        ep = Ep5(7)
        for ws in sents:
            ep.text(ws)
        ep.ctl(TASK_QA)
        if grp == "def_unknown":
            ep.text(G.ask_wh(ids[n], 0), ctx=False)
            ans = ["i", "do", "not", "know."]
        else:
            ep.text(G.ask_wh(target[0], target[1]), ctx=False)
            ans = _canon_say(G, target)
            ep.ask(target)
        key_by_word = {}
        for word in ans:
            parsed = G.surf.get(word.strip(lg.PUNCT))
            if parsed is not None:
                key_by_word[word.strip(lg.PUNCT)] = ep.ent.get(parsed[0], -1)
        return ep.done(
            "lang_define",
            _toks(ans),
            grp,
            grp,
            grammar_checked=True,
            heldout_words=band == "B",
            pseudo=True,
            workspace_output_keys=ep.output_keys(ans, key_by_word),
            workspace_output_kv_targets=tuple(_toks(ans)),
        )


# ---------------------------------------------------------------- 5c induce
def _g_induce(cfg, rng):
    grp, held = rng.choice(cfg["groups"]), bool(cfg["combos"])
    na, pa = cfg["rules"] >= 2 or held, cfg["rules"] >= 3 or held
    c = rng.choice([c for c in range(12) if (c % 4 == 3) == held])  # held-out: neg-particle post AND plural post
    order, rem = divmod(c, 4)
    negpos, plupos = divmod(rem, 2)
    n = cfg["nouns"]
    W = pseudo(rng, n + 5)
    N, V, NP, PP = W[:n], W[n : n + 3], W[n + 3], W[n + 4]
    surface_to_semantic = {
        **{word: (48 + i, entity_token(i)) for i, word in enumerate(N)},
        **{word: (112 + i, relation_token(i)) for i, word in enumerate(V)},
        NP: (115, NEG_TOK),
        PP: (116, PLU_TOK),
    }
    mr_to_surface = {
        **{entity_token(i): (80 + i, lt.tokenizer().word(word)[0]) for i, word in enumerate(N)},
        **{relation_token(i): (120 + i, lt.tokenizer().word(word)[0]) for i, word in enumerate(V)},
        NEG_TOK: (123, lt.tokenizer().word(NP)[0]),
        PLU_TOK: (124, lt.tokenizer().word(PP)[0]),
    }
    semantic_output_key = {
        **{entity_token(i): 48 + i for i in range(n)},
        **{relation_token(i): 112 + i for i in range(len(V))},
        NEG_TOK: 115,
        PLU_TOK: 116,
    }
    surface_output_key = {
        **{word: 80 + i for i, word in enumerate(N)},
        **{word: 120 + i for i, word in enumerate(V)},
        NP: 123,
        PP: 124,
    }

    def build(s, v, o, neg, plu):
        S = ([PP, N[s]] if plupos == 0 else [N[s], PP]) if plu else [N[s]]
        Vb = ([NP, V[v]] if negpos == 0 else [V[v], NP]) if neg else [V[v]]
        parts = {"s": S, "v": Vb, "o": [N[o]]}
        ws = [x for ch in _ORD[order] for x in parts[ch]]
        ws[-1] += "."
        mr = [entity_token(s), relation_token(v), entity_token(o)] + ([NEG_TOK] if neg else []) + ([PLU_TOK] if plu else [])
        return ws, mr

    perm, ex = rng.sample(range(n), n), []
    for i in range(cfg["examples"]):
        ex.append((perm[i % n], i % 3, rng.randrange(n), na and (i == 0 or rng.random() < 0.35),
                   pa and (i == 1 or rng.random() < 0.35)))
    seen_s, seen_v, seen_o = (sorted({e[j] for e in ex}) for j in (0, 1, 2))
    for _ in range(16):
        q = (rng.choice(seen_s), rng.choice(seen_v), rng.choice(seen_o), na and rng.random() < 0.5, pa and rng.random() < 0.5)
        if q not in ex:
            break
    ep = Ep5(8)
    for e in ex:
        ws, mr = build(*e)
        ep.bind_surface(ws, surface_to_semantic, write=True)
        ep.ctl(MARK_R)
        ep.bind_mr(mr, mr_to_surface, write=True)
        ep.ctl(TURN_END)
    ws, mr = build(*q)
    if grp == "ind_s2mr":
        ep.ctl(TASK_S2MR)
        ep.bind_surface(ws, surface_to_semantic, write=False)
        ans, kind = mr, "mr"
        output_keys = tuple(semantic_output_key.get(int(t), -1) for t in mr) + (-1,)
    else:
        ep.ctl(TASK_MR2S)
        ep.bind_mr(mr, mr_to_surface, write=False)
        ans, kind = _toks(ws), "text"
        output_keys = ep.output_keys(ws, surface_output_key)
    return ep.done(
        "lang_induce", ans, f"{grp}_c{c}", grp, kind=kind, heldout_combo=held,
        grammar_checked=False, workspace_output_keys=output_keys,
        workspace_output_kv_targets=tuple(ans),
    )


# ---------------------------------------------------------------- 5d logic
def _join(parts):
    out = [w for i, p in enumerate(parts) for w in (p + (["and"] if i < len(parts) - 1 else []))]
    out[-1] += "."
    return out
def _g_logic(cfg, rng):
    grp, d = rng.choice(cfg["groups"]), max(1, int(cfg["depth"]))
    C, pl, art = pseudo(rng, d + 3), lg.plural, lg.art
    x = rng.choice(lg.NAMES)
    surf = {n: (n, n) for n in lg.NAMES}
    for c in C:
        surf[c] = surf[pl(c)] = (c, c)

    def inst(n, c, neg=False):
        return [n, "is", *(["not"] if neg else []), art(c), c + "."], (n, 7 if neg else 0, c)

    def uni(a, b):
        return ["all", pl(a), "are", pl(b) + "."], (a, REL_ALL, b)

    def non(a, b):
        return ["no", pl(a), "are", pl(b) + "."], (a, REL_NO, b)

    prem, derived, cite = [inst(x, C[0])], [], [[x, "is", art(C[0]), C[0]]]
    if grp == "l_contra":
        prem.append(inst(x, C[0], True))
        q = ["is", "everything", "consistent?"]
        ans = ["no,", x, "is", art(C[0]), C[0], "and", x, "is", "not", art(C[0]), C[0] + "."]
    elif grp == "l_count":
        ns = rng.sample(lg.NAMES, rng.randint(2, 3))
        prem = [inst(n, C[0]) for n in ns]
        q, ans = ["how", "many", pl(C[0]), "are", "there?"], [_NUM[len(ns)] + "."]
    elif grp == "l_some":
        if rng.random() < 0.5:  # existential chain: some A are B, all B are C ... -> some A are tail
            prem = [(["some", pl(C[0]), "are", pl(C[1]) + "."], (C[0], REL_SOME, C[1]))]
            cite = [["some", pl(C[0]), "are", pl(C[1])]]
            for i in range(1, d):
                prem.append(uni(C[i], C[i + 1]))
                cite.append(["all", pl(C[i]), "are", pl(C[i + 1])])
                derived.append((C[0], REL_SOME, C[i + 1]))
            q, ans = ["are", "some", pl(C[0]), pl(C[d]) + "?"], ["yes,"] + _join(cite)
        else:  # 'some' licenses nothing about an individual
            prem = [inst(x, C[0]), (["some", pl(C[0]), "are", pl(C[1]) + "."], (C[0], REL_SOME, C[1]))]
            q, ans = ["is", x, art(C[1]), C[1] + "?"], ["i", "do", "not", "know."]
    elif grp == "l_cond":
        cite = [["if", x, "is", art(C[i]), C[i], "then", x, "is", art(C[i + 1]), C[i + 1]] for i in range(d)]
        prem = [(c_[:-1] + [c_[-1] + "."], (C[i], REL_IF, C[i + 1])) for i, c_ in enumerate(cite)]
        if rng.random() < 0.7:  # modus ponens chain
            prem.append(inst(x, C[0]))
            derived = [(x, 0, C[i + 1]) for i in range(d)]
            q = ["is", x, art(C[d]), C[d] + "?"]
            ans = ["yes,"] + _join([[x, "is", art(C[0]), C[0]]] + cite)
        else:  # affirming the consequent proves nothing
            prem.append(inst(x, C[d]))
            q, ans = ["is", x, art(C[0]), C[0] + "?"], ["i", "do", "not", "know."]
    else:
        up = d - 1 if grp == "l_neg" else d
        for i in range(up):
            prem.append(uni(C[i], C[i + 1]))
            cite.append(["all", pl(C[i]), "are", pl(C[i + 1])])
            derived.append((x, 0, C[i + 1]))
        if grp == "l_neg":
            prem.append(non(C[up], C[d + 1]))
            cite.append(["no", pl(C[up]), "are", pl(C[d + 1])])
            derived.append((x, 7, C[d + 1]))
        tgt = C[d] if grp == "l_chain" else C[d + 1]
        q = ["is", x, art(tgt), tgt + "?"]
        if grp == "l_unknown":
            ans, derived = ["i", "do", "not", "know."], []
        else:
            ans = ["yes," if grp == "l_chain" else "no,"] + [
                w for i, c_ in enumerate(cite) for w in (c_ + (["and"] if i < len(cite) - 1 else []))
            ]
            ans[-1] += "."
    others = [n for n in lg.NAMES if n != x]
    for i in range(min(cfg["distractors"], 5, max(0, 15 - 2 * d))):
        prem.append(inst(others[i], C[d + 2]))
    rng.shuffle(prem)
    ep = Ep5(9)
    for ws, f in prem:
        ep.manual(ws, surf)
        ep._commit(*f)
    ep.ctl(TASK_LOGIC)
    ep.manual(q, surf, ctx=False)
    for f in derived:  # derivation scaffold: placeholder tokens whose slots must hold the derived facts
        ep._commit(*f)
    key_by_word = {
        word: ep.ent.get(binding[0], -1)
        for word, binding in surf.items()
    }
    return ep.done(
        "lang_logic",
        _toks(ans),
        f"{grp}_d{d}",
        grp,
        grammar_checked=False,
        depth=d,
        workspace_output_keys=ep.output_keys(ans, key_by_word),
        workspace_output_kv_targets=tuple(_toks(ans)),
    )


# ---------------------------------------------------------------- 5e/5f shared world
class _W:
    def __init__(s, rng, ep, fem2=False):
        s.r, s.ep = rng, ep
        if fem2:
            f = [i for i, n in enumerate(lg.NAMES) if n in FEM]
            m = [i for i, n in enumerate(lg.NAMES) if n not in FEM]
            s.P = rng.sample(f, 2) + rng.sample(m, 1)
        else:
            s.P = rng.sample(range(6), 3)
        s.O = rng.sample(range(6), 2)
        s.pl, s.ol, s.held, s.rec = {}, {}, {}, []

    def ref(s, p, force=False):
        fem = lg.NAMES[p] in FEM
        last = next((q for q in reversed(s.rec) if (lg.NAMES[q] in FEM) == fem), None)
        if last == p and (force or s.r.random() < 0.7):
            return ("she" if fem else "he"), True
        return lg.NAMES[p], False

    def put_p(s, p):
        l = s.r.randrange(6)
        s.pl[p] = l
        s.ep.story_ws([lg.NAMES[p], "is", "in", "the", lg.PLACES[l] + "."], [(0, ("p", p), None, False), (4, ("p", p), l, False)])
        s.rec.append(p)

    def put_o(s, o):
        l = s.r.randrange(6)
        s.ol[o] = l
        s.ep.story_ws(["the", lg.OBJS[o], "is", "in", "the", lg.PLACES[l] + "."], [(1, ("o", o), None, False), (5, ("o", o), l, False)])

    def relocate(s, p, l):
        s.pl[p] = l
        for o, hp in list(s.held.items()):
            if hp == p:
                s.ol[o] = l
                s.ep.upd(("o", o), l)

    def move(s, p):
        l = s.r.choice([x for x in range(6) if x != s.pl[p]])
        w, pr = s.ref(p)
        ws = [w, s.r.choice(lg.MOVE_V), "to", "the", lg.PLACES[l] + "."]
        s.ep.story_ws(ws, [(0, ("p", p), None, pr), (4, ("p", p), l, False)])
        s.relocate(p, l)
        s.rec.append(p)
        return ws

    def pick(s, p):
        c = [o for o in s.O if o not in s.held and s.ol[o] == s.pl[p]]
        if not c or p in s.held.values():
            return None
        o, (w, pr) = s.r.choice(c), s.ref(p)
        ws = [w] + s.r.choice(lg.PICK_V).split() + ["the", lg.OBJS[o] + "."]
        s.ep.story_ws(ws, [(0, ("p", p), None, pr), (len(ws) - 1, ("o", o), s.pl[p], False)])
        s.held[o] = p
        s.rec.append(p)
        return ws

    def drop(s, p):
        hs = [o for o, hp in s.held.items() if hp == p]
        if not hs:
            return None
        o, (w, pr) = hs[0], s.ref(p)
        ws = [w] + s.r.choice(lg.DROP_V).split() + ["the", lg.OBJS[o] + "."]
        del s.held[o]
        s.ol[o] = s.pl[p]
        s.ep.story_ws(ws, [(0, ("p", p), None, pr), (len(ws) - 1, ("o", o), s.pl[p], False)])
        s.rec.append(p)
        return ws

    def event(s):
        p = s.r.choice(s.P)
        for k in s.r.sample(("move", "move", "pick", "drop"), 4):
            ws = getattr(s, k)(p)
            if ws:
                return ws

    def objloc(s, o):
        return s.pl[s.held[o]] if o in s.held else s.ol[o]

    def where_p(s, p):
        return [lg.NAMES[p], "is", "in", "the", lg.PLACES[s.pl[p]] + "."]

    def where_o(s, o):
        return ["the", lg.OBJS[o], "is", "in", "the", lg.PLACES[s.objloc(o)] + "."]

    def carry(s, p):
        hs = [o for o, hp in s.held.items() if hp == p]
        return [lg.NAMES[p], "is", "carrying", "the", lg.OBJS[hs[0]] + "."] if hs else [lg.NAMES[p], "is", "carrying", "nothing."]


def _g_discourse(cfg, rng):
    grp, T = rng.choice(cfg["groups"]), rng.randint(1, max(1, cfg["turns"]))
    ep = Ep5(10)
    w = _W(rng, ep)
    for p in w.P:
        w.put_p(p)
    for o in w.O:
        w.put_o(o)
    for _ in range(cfg["events"]):
        w.event()
    segs = []
    for k in range(T):
        kind = grp if k == T - 1 else rng.choice(("d_where", "d_where", "d_pron", "d_update"))
        if kind == "d_dist":
            for _ in range(cfg["distract"]):
                w.event()
        elif kind == "d_update":
            w.event()
            w.event()
        ep.ctl(SPK_USER)
        p = rng.choice(w.P)
        rd = [(("p", p), location_token(w.pl[p]))]
        cd = -1
        if kind == "d_retell":
            q, ments, ans = ["tell", "me", "about", lg.NAMES[p] + "."], [(3, ("p", p), None, False)], w.where_p(p) + w.carry(p)
        elif kind in ("d_pron", "d_dist"):
            word, pr = w.ref(p, True)
            cd = next((i for i, q_ in enumerate(reversed(w.rec)) if q_ == p), len(w.rec)) if pr else -1
            q, ments, ans = ["where", "is", word + "?"], [(2, ("p", p), None, pr)], w.where_p(p)
        elif rng.random() < 0.35:
            o = rng.choice(w.O)
            q, ments, ans = ["where", "is", "the", lg.OBJS[o] + "?"], [(3, ("o", o), None, False)], w.where_o(o)
            rd = [(("o", o), location_token(w.objloc(o)))]
        else:
            q, ments, ans = ["where", "is", lg.NAMES[p] + "?"], [(2, ("p", p), None, False)], w.where_p(p)
        ep.story_ws(q, ments)
        ep.ctl(SPK_SYS)
        for name, tgt in rd:
            ep.read_slot(name, tgt)
        key_by_word = {}
        # These entity slots contain location values, not the entity names.
        # Route only the location word through memory; keep names on the
        # backbone path so a slot cannot be trained to emit two different
        # lexical values from the same payload.
        location_key = ep.ent.get(rd[0][0], -1) if rd else -1
        for place in lg.PLACES:
            if location_key >= 0:
                key_by_word[place] = location_key
        segs.append(
            ep.done(
                "lang_discourse",
                _toks(ans),
                f"{kind}_turn{k + 1}",
                kind,
                grammar_checked=False,
                coref_dist=cd,
                workspace_output_keys=ep.output_keys(ans, key_by_word),
                workspace_output_kv_targets=tuple(_toks(ans)),
            )
        )

        if k < T - 1:
            ep = w.ep = ep.fork(10)
    segs[-1].metadata["prev"] = tuple(segs[:-1])
    return segs[-1]


def _act(w, act):
    r, ep = w.r, w.ep
    p = r.choice(w.P)
    if act == "a_inform":
        w.event()
        return ["ok."], [], "ack"
    if act == "a_ask":
        ep.story_ws(["where", "is", lg.NAMES[p] + "?"], [(2, ("p", p), None, False)])
        return w.where_p(p), [(("p", p), location_token(w.pl[p]))], "answer"
    if act == "a_confirm":
        yes = r.random() < 0.5
        l = w.pl[p] if yes else r.choice([x for x in range(6) if x != w.pl[p]])
        ep.story_ws(["is", lg.NAMES[p], "in", "the", lg.PLACES[l] + "?"], [(1, ("p", p), None, False)])
        ans = (["yes,"] + w.where_p(p)) if yes else ["no,", lg.NAMES[p], "is", "not", "in", "the", lg.PLACES[l] + "."]
        return ans, [(("p", p), location_token(w.pl[p]))], "confirm" if yes else "deny"
    if act == "a_correct":
        l = r.choice([x for x in range(6) if x != w.pl[p]])
        ep.story_ws(["no,", lg.NAMES[p], "is", "in", "the", lg.PLACES[l] + "."], [(1, ("p", p), None, False), (5, ("p", p), l, False)])
        w.relocate(p, l)
        w.rec.append(p)
        return ["ok,"] + w.where_p(p), [(("p", p), location_token(l))], "ack_correct"
    if act == "a_social":
        u, a = r.choice((("hello.", "hello."), ("thanks.", "you are welcome."), ("goodbye.", "goodbye.")))
        ep.story_ws([u], [])
        return a.split(), [], "social"
    if act == "a_clarify":  # two feminine persons were both just mentioned -> "she" is ambiguous
        for q in w.P[:2]:
            w.move(q)
        ep.story_ws(["where", "is", "she?"], [])
        return ["who", "do", "you", "mean?"], [], "clarify"
    if r.random() < 0.5:  # a_refuse
        ep.story_ws(["open", "the", "door."], [])
        return ["i", "cannot", "do", "that."], [], "refuse"
    x = r.choice([i for i in range(6) if i not in w.O])
    ep.story_ws(["where", "is", "the", lg.OBJS[x] + "?"], [])
    return ["i", "do", "not", "know."], [], "unknown"


def _g_act(cfg, rng):
    grp, T = rng.choice(cfg["groups"]), rng.randint(2, max(2, cfg["turns"]))
    ep = Ep5(11)
    w = _W(rng, ep, True)
    for p in w.P:
        w.put_p(p)
    for o in w.O:
        w.put_o(o)
    segs = []
    for k in range(T):
        act = grp if k == T - 1 else rng.choice(ACTS)
        ep.ctl(SPK_USER)
        ans, rd, resp = _act(w, act)
        ep.ctl(SPK_SYS)
        for name, tgt in rd:
            ep.read_slot(name, tgt)
        key_by_word = {}
        if rd:
            entity = rd[0][0]
            entity_key = ep.ent.get(entity, -1)
            for place in lg.PLACES:
                if entity_key >= 0:
                    key_by_word[place] = entity_key
        segs.append(
            ep.done(
                "lang_act",
                _toks(ans),
                f"{act}_turn{k + 1}",
                act,
                grammar_checked=False,
                response_act=resp,
                workspace_output_keys=ep.output_keys(ans, key_by_word),
                workspace_output_kv_targets=tuple(_toks(ans)),
            )
        )
        if k < T - 1:
            ep = w.ep = ep.fork(11)
    segs[-1].metadata["prev"] = tuple(segs[:-1])
    return segs[-1]


# ---------------------------------------------------------------- 5g external NLU (input text only; targets typed/labels/spans)
# Normalised rows (to be emitted by the corpus builder): nlu_triples {split,text,triples:[[s,r,o]]},
# nli {split,premise,hypothesis,label}, qa {split,context,question,answer}.
def _g_nlu(cfg, rng):
    grp = rng.choice(cfg["groups"])
    rec = rng.choice(_rows({"triples": "nlu_triples", "nli": "nli", "qa": "qa"}[grp], _split(cfg)))
    ep = Ep5(12, annotate=False)
    ep.ctl(TASK_NLU)
    if grp == "triples":
        ep.text(rec["text"].split())
        ans = [x for s, r, o in rec["triples"] for x in (*s.split(), "|", *r.split(), "|", *o.split(), ";")]
    elif grp == "nli":
        ep.text(rec["premise"].split())
        ep.ctl(SEP)
        ep.text(rec["hypothesis"].split())
        ans = [rec["label"]]
    else:
        ep.text(rec["context"].split())
        ep.ctl(TURN_END)
        ep.text(rec["question"].split())
        ans = rec["answer"].split()
    return ep.done("lang_nlu_external", _toks(ans), grp, grp, grammar_checked=False)


GENERATORS = {
    "lang_parse": _g_parse, "lang_define": _g_define, "lang_induce": _g_induce, "lang_logic": _g_logic,
    "lang_discourse": _g_discourse, "lang_act": _g_act, "lang_nlu_external": _g_nlu,
}
