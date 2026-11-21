#!/usr/bin/env python
"""Offline Stage 4 builder: external corpora -> normalised jsonl, SentencePiece tokenizer, manifest.

Run once and commit the outputs. Dataset schemas/licences are UNVERIFIED: use --inspect, then
--map <kind>.<field>=<column>. Verify every source licence (share-alike terms) before release.
Needs: sentencepiece (+ datasets / pandas for HF / parquet sources).

Tokenizer contract (checked here, relied on by _lang_text / _lang_grammar.Ep):
  * words are encoded independently and equal sentence-level encoding (split_by_whitespace);
  * punctuation never merges with letters (training text has punctuation space-separated and the
    punctuation characters are required single-char pieces), so pieces("garden") is a prefix of
    pieces("garden.").
"""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import itertools
import json
import random
import re
import sys
import tempfile
import unicodedata
from pathlib import Path

sys.path.insert(0, "src")
from initium.data.workspace_capability.episode import TEXT_BASE  # noqa: E402
from initium.data.workspace_capability.tracks import _dev_lexicon  # noqa: E402
from initium.data.workspace_capability.tracks import _lang_grammar as lg  # noqa: E402
from initium.data.workspace_capability.tracks import _lang_text as lt  # noqa: E402

PUNCT = ".,?!;:|"
_PUNCT_RE = re.compile("([" + re.escape(PUNCT) + "])")
KINDS = ("nlg", "story", "accept", "open", "nlu_triples", "nli", "qa")
TOKEN_BUDGET = {
    "story": 480,
    "nlg": 480,
    "accept": 100,
    "open": 460,
    "nlu_triples": 400,
    "nli": 300,
    "qa": 400,
}
ALIASES = {
    "nlg": {
        "triples": ("triples", "modified_triple_sets", "triple_set", "mtriple_set", "input"),
        "text": ("text", "lex", "target", "references"),
    },
    "story": {
        "context": ("context", "story", "passage", "premise"),
        "question": ("question", "query"),
        "answer": ("answer", "target"),
    },
    "accept": {
        "sentence": ("sentence", "text"),
        "label": ("label", "acceptable"),
        "group": ("phenomenon", "UID", "linguistics_term", "field"),
    },
    "open": {"text": ("text", "story", "content")},
    "nli": {
        "premise": ("premise", "sentence1"),
        "hypothesis": ("hypothesis", "sentence2"),
        "label": ("label",),
    },
    "qa": {
        "context": ("context", "story", "passage"),
        "question": ("question",),
        "answer": ("answer", "answers", "target"),
    },
}


def norm(s) -> str:
    s = unicodedata.normalize("NFKC", str(s)).lower().replace("_", " ")
    return " ".join("".join(c if c.isprintable() else " " for c in s).split())


def spaced(line: str) -> str:
    return " ".join(_PUNCT_RE.sub(r" \1 ", line).split())


def split_of(key: str) -> str:
    return "test" if hashlib.sha1(key.encode("utf-8")).digest()[0] % 10 == 0 else "train"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# ---------------------------------------------------------------- loading / adapters
def load_rows(src: str, limit: int):
    p = Path(src)
    if p.exists():
        if p.suffix == ".parquet":
            import pandas as pd

            it = iter(pd.read_parquet(p).to_dict("records"))
        elif p.suffix == ".csv":
            it = csv.DictReader(p.open(encoding="utf-8", newline=""))
        elif p.suffix == ".jsonl":
            it = (json.loads(line) for line in p.open(encoding="utf-8") if line.strip())
        else:
            data = json.loads(p.read_text(encoding="utf-8"))
            it = iter(data if isinstance(data, list) else data.get("data", list(data.values())))
    else:
        from datasets import load_dataset

        name = None
        if ":" in src:
            src, name = src.split(":", 1)
        it = itertools.chain.from_iterable(load_dataset(src, name, streaming=True).values())
    return itertools.islice(it, limit)


def col(row, kind, name, over):
    key = f"{kind}.{name}"
    for k in (over[key],) if key in over else ALIASES[kind][name]:
        if row.get(k) is not None:
            return row[k]
    return None


def triples_of(x):
    """Extract triples from native structures and WebNLG's stringified-list format."""
    out = []

    def walk(v):
        if isinstance(v, str):
            # GEM/WebNLG commonly stores input as a repr-like list of triple strings.
            # Parse the list first so brackets and wrapper quotes do not leak into fields.
            candidate = v.strip()
            if candidate.startswith(("[", "(", "{")):
                try:
                    parsed = ast.literal_eval(candidate)
                except (ValueError, SyntaxError):
                    parsed = None
                if parsed is not None and parsed != v:
                    walk(parsed)
                    return
            if v.count("|") == 2:
                parts = [p.strip().strip("\"'") for p in v.split("|")]
                if all(parts):
                    out.append(tuple(parts))
        elif hasattr(v, "tolist") and callable(v.tolist):
            walk(v.tolist())
        elif isinstance(v, dict):
            for w in v.values():
                walk(w)
        elif isinstance(v, list | tuple):
            if len(v) == 3 and all(isinstance(w, str) for w in v):
                out.append(tuple(p.strip().strip("\"'") for p in v))
            else:
                for w in v:
                    walk(w)

    walk(x)
    return out


def nlg_records(row, over):
    triples, texts = triples_of(col(row, "nlg", "triples", over)), col(row, "nlg", "text", over)
    if isinstance(texts, dict):
        texts = texts.get("text", ())
    texts = [texts] if isinstance(texts, str) else list(texts or ())
    if triples:
        for t in texts[:3]:
            yield {"triples": [[norm(x) for x in tr] for tr in triples], "text": norm(t)}


def story_records(row, over):
    s = row.get("story")
    if isinstance(s, dict) and "type" in s:  # bAbI layout: interleaved context / question rows
        ctx = []
        for kind, text, answer in zip(s["type"], s["text"], s["answer"]):
            if kind == 0:
                ctx.append(text)
            elif answer:
                yield {
                    "context": norm(" ".join(ctx)),
                    "question": norm(text),
                    "answer": norm(answer),
                }
        return
    c, q, a = (col(row, "story", k, over) for k in ("context", "question", "answer"))
    if isinstance(c, str) and isinstance(q, str) and isinstance(a, str) and a:
        yield {"context": norm(c), "question": norm(q), "answer": norm(a)}


def accept_records(row, over):
    group = norm(col(row, "accept", "group", over) or "")
    good, bad = row.get("sentence_good"), row.get("sentence_bad")  # BLiMP minimal pairs
    if good and bad:
        yield {"sentence": norm(good), "label": 1, "_key": group or norm(good)}
        yield {"sentence": norm(bad), "label": 0, "_key": group or norm(bad)}
        return
    s, y = col(row, "accept", "sentence", over), col(row, "accept", "label", over)
    if s is not None and y is not None:
        yield {"sentence": norm(s), "label": int(y), "_key": norm(s)}


def open_records(row, over):
    t = col(row, "open", "text", over)
    if isinstance(t, str):
        words = norm(t).split()[:120]
        if len(words) >= 20:
            yield {"text": " ".join(words)}


_NLI = {0: "entailment", 1: "neutral", 2: "contradiction"}


def nlu_triples_records(
    row, over
):  # same schema as nlg; --map nlu_triples.<field>=<col> is honoured
    return nlg_records(row, {k.replace("nlu_triples.", "nlg.", 1): v for k, v in over.items()})


def nli_records(row, over):
    p, h, y = (col(row, "nli", k, over) for k in ("premise", "hypothesis", "label"))
    if isinstance(y, str) and y.lstrip("-").isdigit():
        y = int(y)
    y = _NLI.get(y) if isinstance(y, int) else (y if y in _NLI.values() else None)
    if isinstance(p, str) and isinstance(h, str) and y:
        yield {"premise": norm(p), "hypothesis": norm(h), "label": y}


def _first(a):
    if isinstance(a, dict):
        a = a.get("text") or a.get("input_text")
    if hasattr(a, "tolist"):
        a = a.tolist()
    return (a[0] if a else None) if isinstance(a, list | tuple) else a


def qa_records(row, over):
    c, q = (col(row, "qa", k, over) for k in ("context", "question"))
    a = _first(col(row, "qa", "answer", over))
    if all(isinstance(x, str) and x for x in (c, q, a)):
        yield {"context": norm(c), "question": norm(q), "answer": norm(a)}


ADAPTERS = {
    "nlg": nlg_records,
    "story": story_records,
    "accept": accept_records,
    "open": open_records,
    "nlu_triples": nlu_triples_records,
    "nli": nli_records,
    "qa": qa_records,
}
SPLIT_KEY = {
    "nlg": lambda r: r["text"],
    "story": lambda r: r["context"] + r["question"],
    "accept": lambda r: r.pop("_key"),
    "open": lambda r: r["text"],
    "nlu_triples": lambda r: r["text"],
    "nli": lambda r: r["premise"] + r["hypothesis"],
    "qa": lambda r: r["context"] + r["question"],
}


def build_external(kind, src, over, limit):
    out, seen = [], set()
    for row in load_rows(src, limit):
        for rec in ADAPTERS[kind](row, over):
            rec["split"] = split_of(SPLIT_KEY[kind](rec))
            sig = json.dumps(rec, sort_keys=True)
            if sig not in seen:
                seen.add(sig)
                out.append(rec)
    if not out:
        sys.exit(
            f"{kind}: no usable records from {src!r}; run --inspect and pass --map {kind}.<field>=<column>"
        )
    return out


def record_lines(kind, r):
    if kind in ("nlg", "nlu_triples"):
        return [" ".join(f"{s} | {rel} | {o} ;" for s, rel, o in r["triples"]), r["text"]]
    if kind == "story":
        return [r["context"], r["question"], r["answer"]]
    if kind == "nli":
        return [r["premise"], r["hypothesis"], r["label"]]
    if kind == "qa":
        return [r["context"], r["question"], r["answer"]]
    return [r["sentence"] if kind == "accept" else r["text"]]


def token_cost(kind, r, sp):
    return sum(
        len(x)
        for x in sp.encode(record_lines(kind, r))
        + ([sp.encode(r["text"])] if kind == "nlg" else [])
    )


# ---------------------------------------------------------------- synthetic grammar corpus
def synthetic_lines(rng):
    G, out = lg.grammar(), set()
    dot = lambda ws, p=".": ws[:-1] + [ws[-1].rstrip(PUNCT) + p]  # noqa: E731
    add = lambda ws: out.add(" ".join(ws))  # noqa: E731
    facts = [f for b in "AB" for f in G.facts[b]]
    sentences = []
    for s, r, o in facts:
        pt = G.pos[o]
        for row in lg.ROWS:
            if row.rel == r and lg.row_ok(row, pt):
                pos, neg = G.say((s, r, o), row), G.say((s, r, o), row, neg=True)
                add(pos), add(neg)
                sentences.append(pos)
        add(G.ask_wh(s, r)), add(G.ask_wh(s, r, pron=True)), add(G.ask_yn(s, r, o, pt))
        try:
            canon = G.canon(r, pt)
        except KeyError:
            continue
        add(["yes,"] + G.say((s, r, o), canon)), add(["no,"] + G.say((s, r, o), canon, neg=True))
    for s in {f[0] for f in facts}:
        add(dot(["describe"] + G.np(s, "the"))), add(dot(["and"] + G.np(s, "the"), "?"))
        add(dot(["no,"] + G.np(s, "the")))
    add(["i", "do", "not", "know."]), add(["tell", "me", "everything."])
    for _ in range(4000):
        a, b = rng.sample(sentences, 2)
        add(dot(a, "") + ["and"] + b)
    N, O, P = lg.NAMES, lg.OBJS, lg.PLACES
    for n, p in itertools.product(N, P):
        add([n, "is", "in", "the", p + "."]), add([n, "is", "not", "in", "the", p + "."])
        add(["yes,", n, "is", "in", "the", p + "."]), add(["is", n, "in", "the", p + "?"])
        add([n, "will", "look", "in", "the", p + "."])
        for v in lg.MOVE_V:
            add([n, v, "to", "the", p + "."])
    for o, p in itertools.product(O, P):
        add(["the", o, "is", "in", "the", p + "."])
    for n, o in itertools.product(N, O):
        for v in lg.PICK_V + lg.DROP_V:
            add([n] + v.split() + ["the", o + "."])
        for p in P:
            add([n, "moved", "the", o, "to", "the", p + "."])
        add([n, "has", "the", o + "."]), add([n, "is", "carrying", "the", o + "."])
        add(["where", "will", n, "look", "for", "the", o + "?"])
    for n in N:
        (
            add(["where", "is", n + "?"]),
            add(["what", "is", n, "carrying?"]),
            add([n, "is", "carrying", "nothing."]),
        )
    for o in O:
        (
            add(["where", "is", "the", o + "?"]),
            add(["who", "has", "the", o + "?"]),
            add(["nobody", "has", "the", o + "."]),
        )
    return sorted(out)


# ---------------------------------------------------------------- tokenizer
def train_spm(corpus: Path, prefix: Path, vocab_size: int, threads: int):
    import sentencepiece as spm

    spm.SentencePieceTrainer.train(
        input=str(corpus),
        model_prefix=str(prefix),
        vocab_size=vocab_size,
        model_type="unigram",
        normalization_rule_name="nmt_nfkc_cf",
        character_coverage=0.9995,
        required_chars=PUNCT,
        byte_fallback=True,
        split_digits=True,
        split_by_whitespace=True,
        add_dummy_prefix=True,
        remove_extra_whitespaces=True,
        unk_id=0,
        bos_id=-1,
        eos_id=-1,
        pad_id=-1,
        hard_vocab_limit=False,
        max_sentence_length=8192,
        num_threads=threads,
        shuffle_input_sentence=False,
    )
    return spm.SentencePieceProcessor(model_file=str(prefix) + ".model")


def verify(sp, synthetic, external, words):
    """Hard-fail on any violation of the tokenizer contract."""
    errors = []
    unk = sp.unk_id()
    punct_pieces = set(PUNCT) | {"\u2581"}
    for name, lines, tolerance in (("synthetic", synthetic, 0.0), ("external", external, 0.005)):
        bad = 0
        for s in lines:
            ids = sp.encode(s)
            by_word = [i for w in s.split() for i in sp.encode(w)]
            if ids != by_word:
                errors.append(f"word-vs-sentence encoding differs: {s!r}")
            if sp.decode(ids) != s or unk in ids:
                bad += 1
                if name == "synthetic":
                    errors.append(f"round-trip/unk failure: {s!r}")
        if lines and bad / len(lines) > tolerance:
            errors.append(
                f"{name}: round-trip/unk failure rate {bad / len(lines):.4%} > {tolerance:.2%}"
            )
    for w in words:
        base = sp.encode(w)
        for p in PUNCT:
            ids = sp.encode(w + p)
            if ids[: len(base)] != base or not all(
                set(sp.id_to_piece(i)) <= punct_pieces for i in ids[len(base) :]
            ):
                errors.append(
                    f"punctuation merges with {w!r}+{p!r}: {[sp.id_to_piece(i) for i in ids]}"
                )
    if errors:
        print("\n".join(errors[:20]), file=sys.stderr)
        sys.exit(f"tokenizer verification FAILED ({len(errors)} violations)")


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(lt.DATA_DIR))
    for kind in KINDS:
        ap.add_argument(
            f"--{kind}", default=None, help="HF id[:config], or local json/jsonl/csv/parquet"
        )
    ap.add_argument("--map", action="append", default=[], help="e.g. nlg.text=lex")
    ap.add_argument("--limit", type=int, default=200_000, help="max source rows per corpus")
    ap.add_argument("--vocab-size", type=int, default=8192)
    ap.add_argument(
        "--threads", type=int, default=1, help="1 keeps tokenizer training deterministic"
    )
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--inspect", action="store_true")
    a = ap.parse_args()
    if not 256 < a.vocab_size <= 16384:
        sys.exit("--vocab-size must be in (256, 16384]")
    over = dict(m.split("=", 1) for m in a.map)
    sources = {k: getattr(a, k) for k in KINDS if getattr(a, k)}
    if a.inspect:
        for kind, src in sources.items():
            print(f"== {kind} ({src})")
            for row in load_rows(src, 2):
                print("  ", {k: str(v)[:80] for k, v in row.items()})
        return

    rng = random.Random(a.seed)
    external = {k: build_external(k, src, over, a.limit) for k, src in sources.items()}
    synthetic = synthetic_lines(rng)
    train_lines = [
        ln
        for kind, recs in external.items()
        for r in recs
        if r["split"] == "train"
        for ln in record_lines(kind, r)
    ]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        corpus = Path(tmp) / "corpus.txt"
        corpus.write_text(
            "\n".join(spaced(x) for x in (*synthetic, *train_lines)) + "\n", encoding="utf-8"
        )
        sp = train_spm(corpus, out / "spm", a.vocab_size, a.threads)

    G = lg.grammar()
    words = sorted(
        {w for i, w in G.word.items() if i in G.ok}
        | {lg.plural(w) for i, w in G.word.items() if i in G.ok}
        | set(lg.NAMES)
        | set(lg.OBJS)
        | set(lg.PLACES)
    )
    ext_sample = rng.sample(train_lines, min(len(train_lines), 5000))
    verify(sp, rng.sample(synthetic, min(len(synthetic), 20000)), ext_sample, words)

    counts, files = {}, {}
    for kind, recs in external.items():
        kept = [r for r in recs if token_cost(kind, r, sp) <= TOKEN_BUDGET[kind]]
        path = out / lt.EXTERNAL_FILES[kind]
        path.write_text(
            "".join(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n" for r in kept),
            encoding="utf-8",
        )
        files[path.name] = sha256(path)
        counts[kind] = {
            "read": len(recs),
            "kept": len(kept),
            "train": sum(r["split"] == "train" for r in kept),
            "test": sum(r["split"] == "test" for r in kept),
        }
    import sentencepiece

    for name in ("spm.model", "spm.vocab"):
        files[name] = sha256(out / name)
    size = sp.get_piece_size()
    manifest = {
        "schema": 1,
        "seed": a.seed,
        "text_base": TEXT_BASE,
        "text_vocab_size": size,
        "total_vocab": TEXT_BASE + size,
        "sentencepiece": {
            "version": sentencepiece.__version__,
            "model_type": "unigram",
            "requested_vocab": a.vocab_size,
            "normalization": "nmt_nfkc_cf",
            "byte_fallback": True,
            "user_pieces": PUNCT,
        },
        "stage3_manifest_sha256": _dev_lexicon.manifest_sha256(),
        "sources": sources,
        "counts": counts,
        "synthetic_lines": len(synthetic),
        "sha256": files,
        "licences_note": "UNVERIFIED: check each external source licence (share-alike) before release",
    }
    (out / "manifest.json").write_text(
        json.dumps(manifest, indent=1, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps({"text_vocab_size": size, "counts": counts}, indent=1))
    print(f"wrote {out} | total vocabulary {TEXT_BASE + size}")


if __name__ == "__main__":
    main()
