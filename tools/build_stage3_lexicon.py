#!/usr/bin/env python
"""Offline Stage 3 lexicon builder. Run once, commit outputs. Schemas/licences UNVERIFIED:
use --inspect, then --map <kind>.<field>=<column>. ConceptNet is share-alike; WordNet permissive."""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import random
import re
import sys
from pathlib import Path

RELS = ("IsA", "HasProperty", "AtLocation", "CapableOf", "UsedFor", "HasA", "Causes")
NORM = {r.lower(): i for i, r in enumerate(RELS)}
ALIASES = {
    "vocab": {"word": ("lemma", "word", "term", "english", "en"), "pos": ("word_type", "pos", "tag", "type"),
              "freq": ("grade_level", "rank", "freq", "frequency", "count")},
    "wordnet": {"word": ("word", "lemma", "name"), "pos": ("pos",),
                "lexname": ("lexname", "lexicographer_file", "lexfile", "category", "domain"),
                "hypernyms": ("hypernyms", "hypernym")},
    "conceptnet": {"head": ("head", "subject", "arg1", "source", "start", "h"),
                   "rel": ("relation", "rel", "relation_type", "predicate", "r"),
                   "tail": ("tail", "object", "arg2", "target", "end", "t"),
                   "pairs": ("positives", "pairs")},
}
OUT = Path("src/initium/data/workspace_capability/stage3")


def load_rows(src):
    p = Path(src)
    if p.exists():
        if p.suffix == ".parquet":
            import pandas as pd
            return pd.read_parquet(p).to_dict("records")
        if p.suffix == ".csv":
            import csv
            return list(csv.DictReader(p.open(encoding="utf-8")))
        text = p.read_text(encoding="utf-8")
        try:
            data = json.loads(text)
            return data if isinstance(data, list) else data.get("data", list(data.values()))
        except json.JSONDecodeError:
            return [json.loads(line) for line in text.splitlines() if line.strip()]
    from datasets import load_dataset
    if src == "relbert/conceptnet":
        from huggingface_hub import hf_hub_download

        rows = []
        for filename in (
            "dataset/train.jsonl",
            "dataset/valid.jsonl",
            "dataset/test.jsonl",
        ):
            path = hf_hub_download(
                repo_id=src,
                filename=filename,
                repo_type="dataset",
            )
            with open(path, encoding="utf-8") as f:
                rows.extend(
                    json.loads(line)
                    for line in f
                    if line.strip()
                )
        return rows
    if src.count("/") == 1 and not src.endswith(".parquet") and src == "cstr/grundwortschatz-voc-en":
        return [r for s in load_dataset(src, data_files="words.parquet").values() for r in s]
    return [r for split in load_dataset(src).values() for r in split]


def col(row, kind, name, over):
    key = f"{kind}.{name}"
    for k in ((over[key],) if key in over else ALIASES[kind][name]):
        if k in row and row[k] is not None:
            return row[k]
    return None


def lemma(x):
    x = str(x).strip().lower()
    if x.startswith("/c/") and x.count("/") >= 3:
        x = x.split("/")[3]
    x = x.split("/")[0]
    return x if re.fullmatch(r"[a-z]{3,12}", x) else None


def norm_pos(p):
    p = str(p or "").lower()
    m = {"noun": "n", "verb": "v", "adj": "a", "adjective": "a", "s": "a", "adv": "r", "adverb": "r"}
    return m.get(p, p[:1] if p[:1] and p[:1] in "nvar" else None)


def conceptnet_triples(rows, over):
    out = set()
    for row in rows:
        rel, pairs = col(row, "conceptnet", "rel", over), col(row, "conceptnet", "pairs", over)
        if rel is None:
            continue
        if pairs is not None:
            cand = [(p[0], rel, p[1]) for p in pairs]
        else:
            h, t = col(row, "conceptnet", "head", over), col(row, "conceptnet", "tail", over)
            cand = [(h, rel, t)] if h is not None and t is not None else []
        for h, r, t in cand:
            r = NORM.get(re.sub(r"[^a-z]", "", str(r).lower().replace("/r/", "")))
            h, t = lemma(h), lemma(t)
            if r is not None and h and t and h != t:
                out.add((h, r, t))
    return out


def clean(triples):
    s = {(h, r, t) for h, r, t in triples if (t, r, h) not in triples}  # symmetric
    by_pair = collections.defaultdict(set)
    for h, r, t in s:
        by_pair[(h, t)].add(r)
    return {(h, r, t) for h, r, t in s if len(by_pair[(h, t)]) == 1}  # contradictory


def wordnet_getter(src, over):
    if src == "nltk":
        from nltk.corpus import wordnet as wn
        return lambda w: [{"lexname": s.lexname(), "pos": s.pos(),
                           "hypernyms": [l.name() for h in s.hypernyms() for l in h.lemmas()]}
                          for s in wn.synsets(w)]
    out = collections.defaultdict(list)  # assumes rows are senses, ordered by sense rank within a word
    for r in load_rows(src):
        w = lemma(col(r, "wordnet", "word", over) or "")
        if not w:
            continue
        hyp = col(r, "wordnet", "hypernyms", over)
        if isinstance(hyp, str):
            hyp = [x for x in re.split(r"[,;|]\s*", hyp) if x]
        out[w].append({"lexname": str(col(r, "wordnet", "lexname", over)),
                       "pos": norm_pos(col(r, "wordnet", "pos", over)),
                       "hypernyms": [lemma(str(x).split(".")[0]) for x in (hyp or [])]})
    return out.get


def l1(a, b):
    na, nb = sum(a.values()) or 1, sum(b.values()) or 1
    return sum(abs(a[k] / na - b[k] / nb) for k in set(a) | set(b))


def band_stats(cl, triples, pos_of, lex_of):
    t = [x for x in triples if x[0] in cl and x[2] in cl]
    deg, out, heads = collections.Counter(), collections.defaultdict(list), collections.defaultdict(set)
    for h, r, tt in t:
        deg[h] += 1
        deg[tt] += 1
        out[h].append((r, tt))
        heads[h].add(r)
    return {"triples": t, "min_degree": min((deg[w] for w in cl), default=0),
            "relations": collections.Counter(r for _, r, _ in t),
            "chains": sum(1 for h, _, m in t for _r, tt in out[m] if tt not in (h, m)),
            "multi_rel_heads": sum(len(v) >= 3 for v in heads.values()),
            "pos": collections.Counter(pos_of(w) for w in cl),
            "lexname": collections.Counter(lex_of(w) for w in cl)}


def valid(st, a):
    return (st["min_degree"] >= 2 and st["chains"] >= a.min_chains and st["multi_rel_heads"] >= a.min_multi_heads
            and all(st["relations"][i] >= (2 if RELS[i] == "Causes" else 3) for i in range(len(RELS))))


def select(triples, a, rng, pos_of, lex_of):
    adj, deg = collections.defaultdict(set), collections.Counter()
    for h, _, t in triples:
        adj[h].add(t)
        adj[t].add(h)
        deg[h] += 1
        deg[t] += 1
    nodes = sorted(n for n in adj if deg[n] >= 3)

    def grow(seed, banned):
        cl = {seed}
        while len(cl) < a.band_size:
            score = collections.Counter(m for n in cl for m in adj[n] if m not in cl and m not in banned)
            if not score:
                return None
            top = max(score.values())
            cl.add(rng.choice(sorted(m for m, s in score.items() if s == top)))
        return cl

    best = None
    for _ in range(a.tries):
        ca = grow(rng.choice(nodes), set())
        sa = band_stats(ca, triples, pos_of, lex_of) if ca else None
        if not sa or not valid(sa, a):
            continue
        free = [n for n in nodes if n not in ca and not (adj[n] & ca)]
        cb = grow(rng.choice(free), ca) if free else None
        sb = band_stats(cb, triples, pos_of, lex_of) if cb else None
        if not sb or not valid(sb, a):
            continue
        score = sum(l1(sa[k], sb[k]) for k in ("pos", "lexname", "relations"))
        if best is None or score < best[0]:
            best = (score, ca, cb, sa, sb)
    if best is None:
        sys.exit("no valid disjoint band pair found: raise --tries, lower --band-size/--min-chains/--min-multi-heads")
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vocab", default="cstr/grundwortschatz-voc-en", help="HF id/path, or 'none'")
    ap.add_argument("--wordnet", default="kth8/english-words-wordnet", help="HF id/path, or 'nltk'")
    ap.add_argument("--conceptnet", default="relbert/conceptnet")
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--map", action="append", default=[], help="e.g. conceptnet.head=arg1")
    ap.add_argument("--inspect", action="store_true")
    ap.add_argument("--band-size", type=int, default=57)  # 7 relation words + 57 per band <= 64
    ap.add_argument("--tries", type=int, default=400)
    ap.add_argument("--min-chains", type=int, default=5)
    ap.add_argument("--min-multi-heads", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    over = dict(m.split("=", 1) for m in a.map)
    if a.inspect:
        for name, src in (("vocab", a.vocab), ("wordnet", a.wordnet), ("conceptnet", a.conceptnet)):
            if src not in ("none", "nltk"):
                rows = load_rows(src)
                print(f"== {name} ({src}): {len(rows)} rows; first rows:")
                for r in rows[:2]:
                    print("  ", {k: str(v)[:80] for k, v in r.items()})
        return
    rng = random.Random(a.seed)
    vocab = {}
    if a.vocab != "none":
        for r in load_rows(a.vocab):
            w = lemma(col(r, "vocab", "word", over) or "")
            if w:
                vocab[w] = (norm_pos(col(r, "vocab", "pos", over)), col(r, "vocab", "freq", over))
    raw = conceptnet_triples(load_rows(a.conceptnet), over)
    n_raw = len(raw)
    triples = clean(raw)
    if vocab:
        triples = {x for x in triples if x[0] in vocab and x[2] in vocab}
    wn = wordnet_getter(a.wordnet, over)
    role = {}
    for h, r, t in triples:
        role.setdefault(h, "n")
        role.setdefault(t, {1: "a", 3: "v"}.get(r, "n"))

    def pos_of(w):
        s = wn(w) or []
        return (vocab.get(w, (None,))[0] or (norm_pos(s[0]["pos"]) if s else None) or role.get(w, "n"))

    def lex_of(w):
        s = wn(w) or []
        return s[0]["lexname"] if s else "unk"

    _, ca, cb, sa, sb = select(sorted(triples), a, rng, pos_of, lex_of)
    bands = {"A": sorted(ca), "B": sorted(cb)}
    word_idx = {r: i for i, r in enumerate(RELS)}
    for w, i in zip(bands["A"], range(7, 7 + len(bands["A"]))):
        word_idx[w] = i
    for w, i in zip(bands["B"], range(64, 64 + len(bands["B"]))):
        word_idx[w] = i
    band_of = {w: b for b in "AB" for w in bands[b]}
    # senses: <=2 per word with distinct lexnames; category = lexname -> COUNT index (top 32)
    freq = collections.Counter()
    for w in band_of:
        seen = []
        for s in wn(w) or []:
            if s["lexname"] not in seen:
                seen.append(s["lexname"])
        freq.update(seen[:2])
    cat_map = {ln: i for i, (ln, _) in enumerate(sorted(freq.items(), key=lambda kv: (-kv[1], kv[0]))[:32])}
    senses, poly = {}, 0
    for w in band_of:
        out, seen = [], set()
        for s in wn(w) or []:
            if s["lexname"] in cat_map and s["lexname"] not in seen:
                seen.add(s["lexname"])
                hyp = next((word_idx[x] for x in s["hypernyms"]
                            if x in word_idx and band_of.get(x) == band_of[w]), None)
                out.append({"lexname": s["lexname"], "category": cat_map[s["lexname"]], "hypernym": hyp})
            if len(out) == 2:
                break
        if out:
            senses[str(word_idx[w])] = out
        poly += len(out) == 2
    if poly == 0:
        print("WARNING: no polysemous words with two mapped lexnames; 3b s_disambig will fail")
    entries = [{"idx": i, "word": r, "band": "rel", "pos": "r", "kind": "relation"} for r, i in
               ((r, word_idx[r]) for r in RELS)]
    entries += [{"idx": word_idx[w], "word": w, "band": band_of[w], "pos": pos_of(w), "kind": "content"}
                for w in sorted(band_of, key=word_idx.get)]
    graph = {"relations": list(RELS)}
    for b, st in (("A", sa), ("B", sb)):
        graph[b] = [[word_idx[h], r, word_idx[t]] for h, r, t in sorted(st["triples"])]
    out_dir = Path(a.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    files = {"lexicon.json": entries, "senses.json": senses, "context_graph.json": graph,
             "word_to_idx.json": word_idx,
             "idx_to_word.json": {str(i): w for w, i in word_idx.items()}}
    shas = {}
    for name, obj in files.items():
        data = json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()
        (out_dir / name).write_bytes(data)
        shas[name] = hashlib.sha256(data).hexdigest()
    stats = {b: {"words": len(bands[b]), "triples": len(st["triples"]), "min_degree": st["min_degree"],
                 "chains": st["chains"], "multi_rel_heads": st["multi_rel_heads"],
                 "relations": {RELS[i]: st["relations"][i] for i in range(len(RELS))},
                 "pos": dict(st["pos"]), "lexname_top": dict(st["lexname"].most_common(8))}
             for b, st in (("A", sa), ("B", sb))}
    manifest = {"schema": 1, "seed": a.seed, "sources": {"vocab": a.vocab, "wordnet": a.wordnet,
                "conceptnet": a.conceptnet}, "filters": {"relations": list(RELS), "single_word_lemmas": True,
                "band_size": a.band_size, "min_chains": a.min_chains, "min_multi_heads": a.min_multi_heads,
                "raw_triples": n_raw, "clean_triples": len(triples)},
                "lexname_to_category": cat_map, "polysemous_words": poly, "sha256": shas, "stats": stats,
                "licences_note": "ConceptNet share-alike; WordNet permissive; verify before release"}
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=1, sort_keys=True), encoding="utf-8")
    print(json.dumps(stats, indent=1))
    print(f"wrote {out_dir} | bands A/B = {len(bands['A'])}/{len(bands['B'])} | polysemous={poly}")


if __name__ == "__main__":
    main()