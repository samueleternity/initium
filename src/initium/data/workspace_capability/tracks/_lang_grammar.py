"""Stage 4 controlled grammar: templates, verbaliser, text-only parser (annotator) and episode builder."""

from __future__ import annotations

import functools
import re
from typing import NamedTuple

from initium.data.workspace_capability.episode import (
    BOS,
    EOS,
    FACT_BASE,
    LANG_TRACK_BASE,
    MARK_O,
    MARK_R,
    MARK_S,
    SEP,
    TaskEpisode,
    location_token,
    relation_token,
)
from initium.data.workspace_capability.tracks import _dev_lexicon as dl
from initium.data.workspace_capability.tracks import _lang_text as lt
from initium.data.workspace_capability.tracks._shared import common_metadata

MAX_T, MAX_ANS = 512, 96
PUNCT = ".,?!;:"
DET = ("the", "a", "an")
FUNCTION = frozenset(
    "the a an is are was were in can could be found used for to has have had by and not cannot does do did "
    "what where who it they this that yes no i know cause causes caused like describe tell me everything "
    "nobody nothing".split()
)
NAMES = ("mary", "john", "sandra", "daniel", "emma", "peter")
OBJS = ("apple", "ball", "book", "key", "cup", "hat")
PLACES = ("hallway", "kitchen", "garden", "office", "bedroom", "bathroom")
MOVE_V = ("moved", "went", "travelled", "journeyed")
PICK_V = ("picked up", "took", "grabbed")
DROP_V = ("dropped", "put down", "discarded")
_SPLIT = re.compile(r"(?<=[.?!])\s+")


def plural(w):
    if w.endswith(("s", "x", "z", "ch", "sh")):
        return w + "es"
    if len(w) > 1 and w.endswith("y") and w[-2] not in "aeiou":
        return w[:-1] + "ies"
    return w + "s"


def art(w):
    return "an" if w[0] in "aeiou" else "a"


def negate(b):
    h = b[0]
    if h in ("is", "are", "was", "were", "could"):
        return (h, "not") + b[1:]
    if h == "can":
        return ("cannot",) + b[1:]
    return {
        "has": ("does", "not", "have"),
        "have": ("do", "not", "have"),
        "causes": ("does", "not", "cause"),
        "cause": ("do", "not", "cause"),
    }[h]


class Row(NamedTuple):
    rel: int
    smode: str
    between: tuple
    omode: str
    passive: bool
    tense: str
    num: str
    flag: str
    alt: bool
    held: bool = False

    @property
    def cls(self):
        return (self.rel, self.tense, self.num, self.flag)


def _r(rel, s, b, o, *, p=False, t="p", n="s", f="", alt=False, h=False):
    return Row(rel, s, tuple(b.split()), o, p, t, n, f, alt, h)


ROWS = (
    _r(0, "the", "is", "a"),
    _r(0, "a", "is", "a", alt=True),
    _r(0, "pl", "are", "pl", n="p"),
    _r(0, "the", "was", "a", t="t"),
    _r(0, "pl", "were", "pl", t="t", n="p"),
    _r(1, "the", "is", "bare"),
    _r(1, "pl", "are", "bare", n="p"),
    _r(1, "the", "was", "bare", t="t"),
    _r(2, "the", "is in", "the"),
    _r(2, "the", "can be found in", "the", alt=True, h=True),
    _r(2, "pl", "are in", "the", n="p"),
    _r(2, "the", "was in", "the", t="t"),
    _r(3, "the", "can", "bare"),
    _r(3, "pl", "can", "bare", n="p"),
    _r(3, "the", "could", "bare", t="t"),
    _r(4, "the", "is used for", "the"),
    _r(4, "pl", "are used for", "the", n="p"),
    _r(4, "the", "is used to", "bare", f="v"),
    _r(5, "the", "has", "a"),
    _r(5, "the", "has", "the", alt=True, h=True),
    _r(5, "pl", "have", "pl", n="p"),
    _r(6, "the", "causes", "the"),
    _r(6, "the", "is caused by", "the", p=True, alt=True),
    _r(6, "pl", "cause", "the", n="p"),
    _r(6, "the", "can cause", "the", f="m", alt=True, h=True),
)
_TP = {0: "n", 1: "a", 2: "n", 3: "v", 4: "nv", 5: "n", 6: "n"}
_KIND = {"a": "a", "the": "the", "pl": "pl", "bare": None}
TAB, NTAB = {}, {}
for _row in ROWS:
    _k = "*" if _row.passive else _KIND[_row.omode]
    TAB[(_row.between, _k)] = _row
    NTAB[(negate(_row.between), _k)] = _row


def row_ok(row, pt):
    if row.rel == 4:
        return (pt == "v") == (row.flag == "v")
    return pt is not None and pt in _TP[row.rel]


def allowed(row, cfg):
    return (
        (not row.held or cfg.get("templates", 0))
        and (not row.alt or cfg["variety"] >= 2)
        and (row.num == "s" or cfg["plural"])
        and (row.tense == "p" or cfg["past"])
        and (not row.passive or cfg["passive"])
    )


class Grammar:
    def __init__(self):
        L = dl.load()
        self.L, self.word, self.pos = L, dict(L.idx_to_word), dict(L.pos)
        rel, surf, bad = set(L.rel_words), {}, set()
        for i, w in self.word.items():
            if i in rel or w in FUNCTION:
                bad.add(i)
                continue
            for s, p in ((w, False), (plural(w), True)):
                o = surf.get(s)
                if o is not None and o[0] != i:
                    bad.update((i, o[0]))
                else:
                    surf[s] = (i, p)
        self.ok = frozenset(i for i in self.word if i not in bad)
        self.surf = {s: v for s, v in surf.items() if v[0] in self.ok}
        self.facts, self.hr, self.tails = {}, {}, {}
        for b in "AB":
            hr, fl = {}, []
            for (h, r), ts in L.pairs[b].items():
                if h not in self.ok or self.pos.get(h) != "n":
                    continue
                good = [t for t in ts if t in self.ok and self._tp(r, t)]
                if good:
                    hr[(h, r)] = good
                    fl.extend((h, r, t) for t in good)
            if not fl:
                raise RuntimeError(
                    f"band {b} has no verbalisable facts; rebuild the Stage 3 lexicon"
                )
            self.facts[b], self.hr[b] = sorted(fl), hr
            self.tails[b] = {
                r: [t for t in L.tails_by_rel[b].get(r, ()) if t in self.ok and self._tp(r, t)]
                for r in range(7)
            }

    def _tp(self, r, t):
        p = self.pos.get(t)
        return p is not None and p in _TP[r]

    # ---- verbalisation -------------------------------------------------
    def np(self, idx, mode):
        w = self.word[idx]
        if mode == "the":
            return ["the", w]
        if mode == "a":
            return [art(w), w]
        if mode == "pl":
            return [plural(w)]
        return [w]

    def say(self, f, row, neg=False):
        s, r, o = f[:3]
        b = negate(row.between) if neg else row.between
        S, O = self.np(s, row.smode), self.np(o, row.omode)
        ws = (O + list(b) + S) if row.passive else (S + list(b) + O)
        ws[-1] += "."
        return ws

    def canon(self, r, pt):
        for row in ROWS:
            if (
                row.rel == r
                and row.tense == "p"
                and row.num == "s"
                and not row.alt
                and row.flag != "m"
                and row_ok(row, pt)
            ):
                return row
        raise KeyError((r, pt))

    def row_for(self, f, cfg, rng):
        pt = self.pos[f[2]]
        c = [x for x in ROWS if x.rel == f[1] and row_ok(x, pt) and allowed(x, cfg)]
        c = [x for x in c if x.held] or c if cfg.get("templates", 0) else c
        return rng.choice(c)

    def ask_wh(self, s, r, pron=False):
        S = ["it"] if pron else self.np(s, "the")
        q = {
            0: ["what", "is"] + S,
            1: ["what", "is"] + S + ["like"],
            2: ["where", "is"] + S,
            3: ["what", "can"] + S + ["do"],
            4: ["what", "is"] + S + ["used", "for"],
            5: ["what", "does"] + S + ["have"],
            6: ["what", "does"] + S + ["cause"],
        }[r]
        q[-1] += "?"
        return q

    def ask_yn(self, s, r, o, pt):
        S, w = self.np(s, "the"), self.word[o]
        q = {
            0: ["is"] + S + self.np(o, "a"),
            1: ["is"] + S + [w],
            2: ["is"] + S + ["in"] + self.np(o, "the"),
            3: ["can"] + S + [w],
            4: ["is"]
            + S
            + (["used", "to", w] if pt == "v" else ["used", "for"] + self.np(o, "the")),
            5: ["does"] + S + ["have"] + self.np(o, "a"),
            6: ["does"] + S + ["cause"] + self.np(o, "the"),
        }[r]
        q[-1] += "?"
        return q

    # ---- text-only parser (annotator) ---------------------------------------
    def parse(self, ws):
        """Returns (mentions, clauses). mention=(start_with_det, word_idx, lemma_idx, plural, det);
        clause=(subj, rel, obj, negated). Function of the words only."""
        toks = [w.strip(PUNCT) for w in ws]
        M = []
        for i, w in enumerate(toks):
            m = self.surf.get(w)
            if m is None:
                continue
            d = toks[i - 1] if i and toks[i - 1] in DET else None
            M.append((i - (d is not None), i, m[0], m[1], "a" if d == "an" else d))
        clauses = []
        cuts = [-1] + [i for i, w in enumerate(toks) if w == "and"] + [len(toks)]
        for a, b in zip(cuts, cuts[1:]):
            ms = [m for m in M if a < m[0] and m[1] < b]
            if len(ms) < 2:
                continue
            m1, m2 = ms[0], ms[-1]
            btw = tuple(toks[m1[1] + 1 : m2[0]])
            kind = m2[4] or ("pl" if m2[3] else None)
            for tab, neg in ((TAB, False), (NTAB, True)):
                row = tab.get((btw, kind)) or tab.get((btw, "*"))
                if row is not None:
                    s, o = (m2[2], m1[2]) if row.passive else (m1[2], m2[2])
                    clauses.append((s, row.rel, o, neg))
                    break
        return M, clauses

    def parse_text(self, text):
        out = set()
        for s in _SPLIT.split(text.strip()):
            if s:
                out.update(self.parse(s.split())[1])
        return frozenset(out)

    def sentence_ok(self, ws):
        ws = list(ws)
        if ws and ws[0] in ("yes,", "no,"):
            ws = ws[1:]
        if [w.strip(PUNCT) for w in ws] == ["i", "do", "not", "know"]:
            return True
        _, cl = self.parse(ws)
        if len(cl) != 1:
            return False
        s, r, o, neg = cl[0]
        pt = self.pos[o]
        return any(ws == self.say((s, r, o), row, neg) for row in ROWS if row_ok(row, pt))

    def text_ok(self, text):
        ss = [s for s in _SPLIT.split(text.strip()) if s]
        return bool(ss) and all(self.sentence_ok(s.split()) for s in ss)


@functools.lru_cache(maxsize=1)
def grammar() -> Grammar:
    return Grammar()


class Ep:
    """Episode builder. Annotation (read keys, write masks, fact-slot markers, slot-supervision labels) is a
    deterministic function of the emitted text only; the generator never passes facts to it."""

    def __init__(self, marker, annotate=True):
        self.tk, self.G = lt.tokenizer(), grammar()
        self.t = [BOS, LANG_TRACK_BASE + marker, SEP]
        self.addr, self.evw, self.rd = [], [], []
        self.ent, self.cl, self.pl = {}, [], {}
        self.ann = bool(annotate and lt.settings["annotator"])
        self.ol, self.held = {}, {}

    def ctl(self, *ids):
        self.t.extend(ids)

    def _w(self, pos, key, loc):
        if pos < len(self.t):
            self.addr.append((pos, key, True))
            self.evw.append((pos, key, location_token(loc)))

    def mark(self, key, target):
        """Write the current sentence-end state to a slot; target is used by the aux loss only."""
        if lt.settings["annotator"]:
            p = len(self.t) - 1
            self.addr.append((p, key, True))
            self.evw.append((p, key, target))

    def _enc(self, ws):
        sp = []
        for w in ws:
            ids = self.tk.word(w)
            sp.append((len(self.t), len(ids)))
            self.t.extend(ids)
        return sp

    def _pos(self, ws, sp, i):
        st, n = sp[i]
        core = self.tk.word(ws[i].strip(PUNCT))
        k = len(core)
        return st + (k if 0 < k <= n and tuple(self.t[st : st + k]) == core else n) - 1

    def _key(self, idx, write):
        k = self.ent.get(idx)
        if k is None and write and len(self.ent) < 24:
            k = self.ent[idx] = len(self.ent)
        return k

    def _commit(self, s, r, o):
        if len(self.cl) >= 8:
            return
        i, p = len(self.cl), len(self.t)
        self.t.extend((MARK_S, MARK_R, MARK_O))
        self.addr.extend(((p, 24 + i, True), (p + 1, 48 + i, True), (p + 2, 32 + i, True)))
        self.evw.extend(
            (
                (p, 24 + i, FACT_BASE + s),
                (p + 1, 48 + i, relation_token(r)),
                (p + 2, 32 + i, FACT_BASE + o),
            )
        )
        self.cl.append((s, r, o))

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
                self.evw.append((pos, k, FACT_BASE + idx))
        if ctx:
            for s, r, o, neg in clauses:
                if not neg:
                    self._commit(s, r, o)

    def typed(self, f):
        s, r, o = f
        p = len(self.t)
        self.t.extend((FACT_BASE + s, relation_token(r), FACT_BASE + o))
        if self.ann:
            for pos, idx in ((p, s), (p + 2, o)):
                k = self._key(idx, True)
                if k is not None:
                    self.addr.append((pos, k, True))
                    self.evw.append((pos, k, FACT_BASE + idx))
            self._commit(s, r, o)

    def story(self, ws, *, ctx=True):
        sp = self._enc(ws)
        if not self.ann:
            return
        toks = [w.strip(PUNCT) for w in ws]
        p = next((NAMES.index(w) for w in toks if w in NAMES), None)
        o = next((OBJS.index(w) for w in toks if w in OBJS), None)
        li = next((PLACES.index(w) for w in reversed(toks) if w in PLACES), None)
        for i, w in enumerate(toks):
            if w in NAMES:
                self.addr.append((self._pos(ws, sp, i), NAMES.index(w), False))
            elif w in OBJS:
                self.addr.append((self._pos(ws, sp, i), 8 + OBJS.index(w), False))
        if not ctx:
            return
        if li is not None and (o is not None or p is not None):
            pos = self._pos(ws, sp, max(i for i, w in enumerate(toks) if w in PLACES))
            key = 8 + o if o is not None else p
            self.addr.append((pos, key, True))
            self.evw.append((pos, key, location_token(li)))
            if o is not None:
                old = self.ol.get(o)
                self.ol[o] = li
                if old is not None and old != li:
                    self._w(pos + 1, 16 + o, old)  # previous-location slot (belief source)
            else:
                self.pl[p] = li
                h = next((x for x, hp in self.held.items() if hp == p), None)
                if h is not None:
                    self.ol[h] = li
                    self._w(pos + 1, 8 + h, li)  # carried object follows its carrier
        elif o is not None and p is not None and p in self.pl:
            pos = self._pos(ws, sp, next(i for i, w in enumerate(toks) if w in OBJS))
            self.addr.append((pos, 8 + o, True))
            self.evw.append((pos, 8 + o, location_token(self.pl[p])))
            if any(w in toks for w in ("dropped", "down", "discarded")):
                self.held.pop(o, None)
            else:
                self.held[o] = p
            self.ol[o] = self.pl[p]

    def ask(self, fact):
        """Slot-readout supervision at the last input step for the context clause that supports `fact`."""
        if not self.ann:
            return
        s, r, o = fact
        for i, c in enumerate(self.cl):
            if c == (s, r, o):
                st = len(self.t) - 1
                self.rd.extend(
                    (
                        (st, 24 + i, FACT_BASE + s),
                        (st, 48 + i, relation_token(r)),
                        (st, 32 + i, FACT_BASE + o),
                    )
                )
                return

    def open_text(self, ws, stop):
        """Open-vocab annotator: each distinct content word (max 24) -> key, written at its last piece; target = its first piece token."""
        sp = self._enc(ws)
        if not self.ann:
            return
        oe, ot = self.__dict__.setdefault("oe", {}), self.__dict__.setdefault("ot", {})
        for i, w in enumerate(ws):
            c = w.strip(PUNCT)
            if len(c) < 4 or not c.isalpha() or c in stop or c in FUNCTION:
                continue
            k = oe.get(c)
            if k is None:
                if len(oe) >= 24:
                    continue
                k = oe[c] = len(oe)
            pos, tgt = self._pos(ws, sp, i), self.t[sp[i][0]]
            ot[k] = tgt
            self.addr.append((pos, k, True))
            self.evw.append((pos, k, tgt))

    def open_reads(self):
        """Retention supervision: at the last input step every written slot must decode its word."""
        if self.ann:
            st = len(self.t) - 1
            self.rd.extend((st, k, t) for k, t in self.__dict__.get("ot", {}).items())

    def done(self, track, answer, qtype, group, kind="text", **meta):
        ans = tuple(answer) + (EOS,)
        if len(ans) > MAX_ANS or len(self.t) + len(ans) > MAX_T:
            raise OverflowError
        d = {}
        for pos, k, w in self.addr:
            d[pos] = (k, w)
        return TaskEpisode(
            track,
            tuple(self.t),
            ans,
            common_metadata(
                track=track,
                sequence_length=len(self.t),
                query_type=qtype,
                relevant_updates=len(self.cl),
                candidate_baseline_pct=1.0,
                progression_group=group,
                answer_kind=kind,
                answer_stride=1,
                lang_addr=tuple((p, k, w) for p, (k, w) in d.items()),
                workspace_event_writes=tuple(self.evw),
                workspace_query_reads=tuple(self.rd),
                **meta,
            ),
        )
