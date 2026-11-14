"""Stage 3 data loader + runtime settings (stdlib only: imported by curriculum.py)."""

from __future__ import annotations

import functools
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

STAGE3_TRACKS = (
    "developmental_lexical",
    "developmental_semantic",
    "developmental_production_semantic",
    "developmental_production_semantic_unkeyed",
)
DATA_DIR = Path(__file__).resolve().parent.parent / "stage3"
settings = {"lexicon_mode": "stable", "family": "words", "unkeyed": False}


def configure(lexicon_mode=None, family=None, unkeyed=None):
    if lexicon_mode is not None:
        if lexicon_mode not in ("stable", "episodic"):
            raise ValueError("lexicon_mode must be 'stable' or 'episodic'")
        settings["lexicon_mode"] = lexicon_mode
    if family is not None:
        if family not in ("words", "senses", "all"):
            raise ValueError("lexical family must be words, senses or all")
        settings["family"] = family
    if unkeyed is not None:
        settings["unkeyed"] = bool(unkeyed)


def _path(name):
    path = DATA_DIR / name
    if not path.exists():
        raise FileNotFoundError(f"{path} is missing; run tools/build_stage3_lexicon.py first")
    return path


def manifest_sha256():
    return hashlib.sha256(_path("manifest.json").read_bytes()).hexdigest()


@functools.lru_cache(maxsize=1)
def load():
    entries = json.loads(_path("lexicon.json").read_text(encoding="utf-8"))
    graph = json.loads(_path("context_graph.json").read_text(encoding="utf-8"))
    senses = json.loads(_path("senses.json").read_text(encoding="utf-8"))
    rel_words = sorted(e["idx"] for e in entries if e["band"] == "rel")
    if rel_words != list(range(len(graph["relations"]))):
        raise ValueError("relation words must occupy word indices 0..6 in relation order")
    L = SimpleNamespace(
        idx_to_word={e["idx"]: e["word"] for e in entries},
        pos={e["idx"]: e["pos"] for e in entries},
        rel_words=rel_words,
        bands={b: [e["idx"] for e in entries if e["band"] == b] for b in "AB"},
        senses={int(k): [(s["category"], s["lexname"]) for s in v] for k, v in senses.items() if v},
        pairs={}, heads={}, tails_by_rel={}, chains={},
    )
    for band in "AB":
        pairs, heads, tails = defaultdict(list), defaultdict(dict), defaultdict(set)
        for h, r, t in graph[band]:
            pairs[(h, r)].append(t)
            heads[h].setdefault(r, []).append(t)
            tails[r].add(t)
        L.pairs[band], L.heads[band] = dict(pairs), dict(heads)
        L.tails_by_rel[band] = {r: sorted(v) for r, v in tails.items()}
        L.chains[band] = [
            (h, r1, m, r2, t)
            for (h, r1), ms in L.pairs[band].items() for m in ms
            for r2, ts in L.heads[band].get(m, {}).items() for t in ts if t not in (h, m)
        ]
    return L


def word_name(idx):
    return load().idx_to_word.get(int(idx), f"w{idx}")


def stable_lexicon(rng, keys, held):
    """Drop-in for developmental._lexicon: concept key -> stable real-word index (band A, or B if held)."""
    L = load()
    pool = L.bands["B" if held else "A"]
    rel = [k for k in keys if 56 <= k < 64]
    prop = [k for k in keys if 32 <= k < 48]
    obj = [k for k in keys if k not in rel and k not in prop]
    W = dict(zip(rel, rng.sample(L.rel_words, len(rel))))
    taken = set()

    def take(pref, n):
        out = rng.sample([w for w in pref if w not in taken], min(n, len([w for w in pref if w not in taken])))
        if len(out) < n:
            out += rng.sample([w for w in pool if w not in taken and w not in out], n - len(out))
        taken.update(out)
        return out

    W.update(zip(prop, take([w for w in pool if L.pos[w] == "a"], len(prop))))
    W.update(zip(obj, take([w for w in pool if L.pos[w] == "n"], len(obj))))
    spare = take(pool, 1)[0]
    return {k: W[k] for k in keys}, spare