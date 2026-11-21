#!/usr/bin/env python3

import collections
import hashlib
import json
import re
import sys
from pathlib import Path

VOCAB_ID = "cstr/grundwortschatz-voc-en"
CONCEPTNET_ID = "relbert/conceptnet"
OUT = Path("src/initium/data/workspace_capability/stage3")

RELATIONS = (
    "IsA", "HasProperty", "AtLocation", "CapableOf",
    "UsedFor", "HasA", "Causes"
)

errors = []
warnings = []


def check(condition, message):
    if condition:
        print(f"  PASS  {message}")
    else:
        print(f"  FAIL  {message}")
        errors.append(message)


def warn(message):
    print(f"  WARN  {message}")
    warnings.append(message)


def normalize_word(value):
    if value is None:
        return None

    value = str(value).strip().lower()

    if value.startswith("/c/") and value.count("/") >= 3:
        value = value.split("/")[3]

    value = value.split("/")[0]

    return value if re.fullmatch(r"[a-z]{3,12}", value) else None


def normalize_relation(value):
    value = str(value).lower().replace("/r/", "")
    value = re.sub(r"[^a-z]", "", value)

    return next(
        (i for i, relation in enumerate(RELATIONS)
         if relation.lower() == value),
        None
    )


def load_vocab():
    from datasets import load_dataset

    ds = load_dataset(VOCAB_ID, data_files="words.parquet")
    return [row for split in ds.values() for row in split]


def load_conceptnet():
    from huggingface_hub import hf_hub_download

    rows = []

    for filename in (
        "dataset/train.jsonl",
        "dataset/valid.jsonl",
        "dataset/test.jsonl",
    ):
        path = hf_hub_download(
            repo_id=CONCEPTNET_ID,
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


def clean_triples(triples):
    # Same cleaning logic as the builder.
    result = {
        (h, r, t)
        for h, r, t in triples
        if (t, r, h) not in triples
    }

    by_pair = collections.defaultdict(set)

    for h, r, t in result:
        by_pair[(h, t)].add(r)

    return {
        (h, r, t)
        for h, r, t in result
        if len(by_pair[(h, t)]) == 1
    }


def audit_vocab():
    print("\n========== 1. VOCABULARY ==========")

    rows = load_vocab()

    words = {}
    missing_pos = 0
    missing_frequency = 0
    malformed_frequency = 0

    for row in rows:
        word = normalize_word(row.get("lemma") or row.get("word"))

        if not word:
            continue

        words[word] = row

        if not row.get("word_type"):
            missing_pos += 1

        if not row.get("grade_level") and not row.get("frequency_json"):
            missing_frequency += 1

        raw_frequency = row.get("frequency_json")

        if raw_frequency:
            try:
                json.loads(raw_frequency) if isinstance(
                    raw_frequency, str
                ) else raw_frequency
            except (ValueError, TypeError):
                malformed_frequency += 1

    check(len(rows) > 0, f"Loaded {len(rows):,} vocabulary rows")
    check(len(words) > 0, f"{len(words):,} usable unique lemmas")
    check(missing_pos == 0, f"Missing POS fields: {missing_pos}")
    check(
        missing_frequency == 0,
        f"Missing frequency/grade fields: {missing_frequency}"
    )
    check(
        malformed_frequency == 0,
        f"Malformed frequency JSON: {malformed_frequency}"
    )

    print("  POS distribution:")

    print(dict(collections.Counter(
        str(row.get("word_type"))
        for row in words.values()
    )))

    return words


def audit_wordnet(words):
    print("\n========== 2. WORDNET ==========")

    import nltk

    try:
        from nltk.corpus import wordnet as wn
        wn.synsets("dog")
    except LookupError:
        nltk.download("wordnet", quiet=True)
        from nltk.corpus import wordnet as wn

    covered = 0
    missing = []
    lexnames = collections.Counter()
    pos = collections.Counter()
    total_senses = 0
    senses_with_hypernyms = 0

    for word in words:
        synsets = wn.synsets(word)

        if not synsets:
            missing.append(word)
            continue

        covered += 1

        for synset in synsets:
            total_senses += 1
            lexnames[synset.lexname()] += 1
            pos[synset.pos()] += 1

            if synset.hypernyms():
                senses_with_hypernyms += 1

    coverage = covered / max(len(words), 1)

    check(covered > 0, f"WordNet covers {covered:,}/{len(words):,} words")
    check(
        coverage >= 0.70,
        f"WordNet coverage >= 70% ({coverage:.1%})"
    )
    check(total_senses > covered, f"Multiple senses available: {total_senses:,}")
    check(bool(lexnames), f"Semantic lexnames available: {len(lexnames)}")

    print(f"  Words without WordNet senses: {len(missing)}")
    if missing:
        print("  Examples:", missing[:15])

    print(f"  Total senses: {total_senses:,}")
    print(f"  Senses with hypernyms: {senses_with_hypernyms:,}")
    print(f"  POS distribution: {dict(pos)}")
    print(f"  Top lexnames: {dict(lexnames.most_common(10))}")

    return wn


def audit_conceptnet(vocab):
    print("\n========== 3. CONCEPTNET ==========")

    rows = load_conceptnet()

    triples = set()
    relation_counts = collections.Counter()
    invalid = 0
    unsupported = 0

    for row in rows:
        relation = normalize_relation(row.get("relation"))

        if relation is None:
            unsupported += 1
            continue

        head = normalize_word(row.get("head"))
        tail = normalize_word(row.get("tail"))

        if not head or not tail or head == tail:
            invalid += 1
            continue

        triples.add((head, relation, tail))

    cleaned = clean_triples(triples)

    in_vocab = {
        triple for triple in cleaned
        if triple[0] in vocab and triple[2] in vocab
    }

    relation_counts = collections.Counter(
        relation for _, relation, _ in in_vocab
    )

    check(len(rows) > 0, f"Loaded {len(rows):,} ConceptNet rows")
    check(len(triples) > 0, f"Valid normalized triples: {len(triples):,}")
    check(len(cleaned) > 0, f"Triples after cleaning: {len(cleaned):,}")
    check(
        len(in_vocab) > 0,
        f"Triples with both endpoints in vocabulary: {len(in_vocab):,}"
    )

    for i, relation in enumerate(RELATIONS):
        check(
            relation_counts[i] > 0,
            f"{relation}: {relation_counts[i]:,} in-vocabulary triples"
        )

    print(f"  Invalid endpoints/self-loops: {invalid:,}")
    print(f"  Unsupported relations: {unsupported:,}")

    return in_vocab


def audit_generated(vocab, wn, source_triples):
    print("\n========== 4. GENERATED STAGE 3 FILES ==========")

    required = [
        "lexicon.json",
        "senses.json",
        "context_graph.json",
        "word_to_idx.json",
        "idx_to_word.json",
        "manifest.json",
    ]

    for filename in required:
        check(
            (OUT / filename).is_file(),
            f"{filename} exists"
        )

    if not all((OUT / f).is_file() for f in required):
        return

    def read(filename):
        return json.loads((OUT / filename).read_text(encoding="utf-8"))

    lexicon = read("lexicon.json")
    senses = read("senses.json")
    graph = read("context_graph.json")
    word_to_idx = read("word_to_idx.json")
    idx_to_word = read("idx_to_word.json")
    manifest = read("manifest.json")

    # Verify output hashes.
    for filename, expected in manifest.get("sha256", {}).items():
        actual = hashlib.sha256((OUT / filename).read_bytes()).hexdigest()
        check(actual == expected, f"SHA-256: {filename}")

    check(
        manifest.get("sources", {}).get("wordnet") == "nltk",
        "Manifest records NLTK WordNet"
    )

    entries = {entry["idx"]: entry for entry in lexicon}
    words = [entry for entry in lexicon if entry["kind"] == "content"]
    relation_entries = [entry for entry in lexicon if entry["kind"] == "relation"]

    a_words = {e["word"] for e in words if e["band"] == "A"}
    b_words = {e["word"] for e in words if e["band"] == "B"}

    check(
        len(entries) == len(lexicon),
        "Lexicon indices are unique"
    )
    check(
        len({e["word"] for e in lexicon}) == len(lexicon),
        "Lexicon words are unique"
    )
    check(
        len(relation_entries) == 7,
        f"Seven relation tokens: {len(relation_entries)}"
    )
    check(
        {e["word"] for e in relation_entries} == set(RELATIONS),
        "All expected relation tokens are present"
    )
    check(
        len(a_words) == 57 and len(b_words) == 57,
        f"Band sizes: A={len(a_words)}, B={len(b_words)}"
    )
    check(
        not (a_words & b_words),
        "Bands A and B are disjoint"
    )
    check(
        all(w in vocab for w in a_words | b_words),
        "Every generated word exists in vocabulary"
    )

    # Check both directions of the word/index mapping.
    mapping_ok = all(
        idx_to_word.get(str(idx)) == word
        and word_to_idx.get(word) == idx
        for word, idx in word_to_idx.items()
    )

    check(mapping_ok, "word_to_idx and idx_to_word are consistent")

    # Check POS and senses.
    check(
        all(e.get("pos") in ("n", "v", "a", "r") for e in words),
        "All content words have valid POS"
    )

    polysemous = 0
    sense_errors = 0
    category_map = manifest.get("lexname_to_category", {})

    for idx, word_senses in senses.items():
        if idx not in {str(e["idx"]) for e in words}:
            sense_errors += 1
            continue

        if len(word_senses) > 2:
            sense_errors += 1

        if len(word_senses) == 2:
            polysemous += 1

        entry = entries[int(idx)]
        actual_lexnames = {
            s.lexname() for s in wn.synsets(entry["word"])
        }

        for sense in word_senses:
            if sense["lexname"] not in actual_lexnames:
                sense_errors += 1

            if category_map.get(sense["lexname"]) != sense["category"]:
                sense_errors += 1

            hypernym = sense.get("hypernym")
            if hypernym is not None:
                target = entries.get(hypernym)

                if not target or target["band"] != entry["band"]:
                    sense_errors += 1

    check(sense_errors == 0, f"Sense consistency errors: {sense_errors}")
    check(
        polysemous == manifest.get("polysemous_words"),
        f"Polysemy count: {polysemous}"
    )

    # Check graph structure and source provenance.
    graph_errors = 0
    graph_triples = set()

    for band in ("A", "B"):
        band_words = a_words if band == "A" else b_words

        for triple in graph.get(band, []):
            if len(triple) != 3:
                graph_errors += 1
                continue

            h, r, t = triple

            if h not in entries or t not in entries or r not in range(7):
                graph_errors += 1
                continue

            head = entries[h]
            tail = entries[t]

            if head["band"] != band or tail["band"] != band:
                graph_errors += 1
                continue

            if (head["word"], r, tail["word"]) not in source_triples:
                graph_errors += 1

            graph_triples.add((head["word"], r, tail["word"]))

    check(graph_errors == 0, f"Graph consistency errors: {graph_errors}")
    check(
        len(graph_triples) == sum(len(graph[b]) for b in ("A", "B")),
        "Graph triples are unique"
    )

    # Compare generated band statistics with manifest.
    for band in ("A", "B"):
        expected = manifest.get("stats", {}).get(band, {})
        actual = len(graph.get(band, []))

        check(
            actual == expected.get("triples"),
            f"Band {band} triple count matches manifest ({actual})"
        )


def main():
    print("STAGE 3 DATA VERIFICATION")
    print("=========================")

    vocab = audit_vocab()
    wn = audit_wordnet(vocab)
    triples = audit_conceptnet(set(vocab))
    audit_generated(vocab, wn, triples)

    print("\n========== FINAL RESULT ==========")

    print(f"Errors:   {len(errors)}")
    print(f"Warnings: {len(warnings)}")

    if errors:
        print("\nRESULT: FAILED — do not train yet.")
        sys.exit(1)

    print("\nRESULT: PASSED — all checks passed.")
    print("Review warnings before training.")


if __name__ == "__main__":
    main()