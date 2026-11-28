"""Build documented, deduplicated token shards from a licensed JSONL manifest.

Manifest format (one JSON object per line)::

    {"path": "corpus/source.txt", "license": "CC-BY-4.0", "source": "source-name"}

Paths are relative to the manifest unless absolute. The tool writes raw-clean
train/validation/test text, tokenized JSONL shards, tokenizer metadata, and a
build report to the output directory. It does not download corpus data.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from initium.data.language.corpus import read_manifest, stable_split
from initium.data.language.tokenizer import train_tokenizer

EMAIL = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE)
PHONE = re.compile(r"(?<!\w)(?:\+?\d[\d .()/-]{7,}\d)(?!\w)")
URL = re.compile(r"https?://\S+", re.IGNORECASE)


def clean_text(text: str, *, redact_pii: bool = True) -> tuple[str, int]:
    text = text.replace("\x00", " ")
    redactions = 0
    if redact_pii:
        for pattern in (EMAIL, PHONE):
            text, count = pattern.subn(" [REDACTED] ", text)
            redactions += count
    text = re.sub(r"[\t\r\f\v ]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    return text, redactions


def _hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def build(manifest_path: Path, output: Path, *, vocab_size: int, min_chars: int,
          max_replacement_ratio: float, shard_documents: int, keep_urls: bool,
          no_pii_redaction: bool):
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = manifest_path.resolve()
    rows, seen, report = [], set(), Counter()
    source_manifest = list(read_manifest(manifest_path))
    licenses = {}
    for entry in source_manifest:
        source_path = Path(entry["path"])
        if not source_path.is_absolute():
            source_path = manifest_path.parent / source_path
        source_path = source_path.resolve()
        licenses[entry["source"]] = entry["license"]
        with source_path.open(encoding="utf-8", errors="replace") as stream:
            for line_no, raw in enumerate(stream, 1):
                clean, redactions = clean_text(raw, redact_pii=not no_pii_redaction)
                report["pii_redactions"] += redactions
                if not keep_urls:
                    clean = URL.sub(" ", clean).strip()
                if len(clean) < min_chars:
                    report["too_short"] += 1
                    continue
                replacement_ratio = clean.count("\ufffd") / max(len(clean), 1)
                if replacement_ratio > max_replacement_ratio:
                    report["encoding_noise"] += 1
                    continue
                digest = _hash(clean)
                if digest in seen:
                    report["duplicates"] += 1
                    continue
                seen.add(digest)
                doc_id = f"{entry['source']}:{digest[:20]}"
                split = stable_split(doc_id)
                rows.append({"doc_id": doc_id, "source": entry["source"],
                             "license": entry["license"], "split": split, "text": clean,
                             "source_line": line_no})
    for split in ("train", "validation", "test"):
        (output / f"{split}.txt").write_text(
            "\n".join(row["text"] for row in rows if row["split"] == split) + "\n",
            encoding="utf-8")
    training_text = [output / "train.txt"]
    tokenizer_info = train_tokenizer(training_text, output / "tokenizer" / "tokenizer.json", vocab_size)
    from initium.data.language.tokenizer import load_tokenizer

    tokenizer = load_tokenizer(output / "tokenizer" / "tokenizer.json")
    eos_id = tokenizer.token_to_id("<eos>")
    counters = Counter()
    shard_root = output / "shards"
    shard_root.mkdir(exist_ok=True)
    shard_files = {}
    handles, shard_counts = {}, Counter()
    try:
        for split in ("train", "validation", "test"):
            split_rows = [row for row in rows if row["split"] == split]
            if not split_rows:
                continue
            shard_id = 0
            for offset in range(0, len(split_rows), shard_documents):
                path = shard_root / f"{split}-{shard_id:05d}.jsonl"
                shard_files[split] = shard_files.get(split, []) + [str(path.relative_to(output))]
                with path.open("w", encoding="utf-8") as sink:
                    for row in split_rows[offset:offset + shard_documents]:
                        ids = tokenizer.encode(row["text"], add_special_tokens=False).ids
                        ids.append(eos_id)
                        sink.write(json.dumps({"doc_id": row["doc_id"], "source": row["source"],
                                               "license": row["license"], "split": split,
                                               "input_ids": ids}, ensure_ascii=False) + "\n")
                        counters[f"{split}_tokens"] += len(ids)
                        shard_counts[f"{split}_documents"] += 1
                shard_id += 1
    finally:
        for handle in handles.values():
            handle.close()
    for row in rows:
        report[f"{row['split']}_documents"] += 1
    report["unique_documents"] = len(rows)
    report["manifest_sources"] = len(source_manifest)
    metadata = {
        "schema_version": 1,
        "manifest": str(manifest_path),
        "license_by_source": licenses,
        "split_policy": "sha256(doc_id), 98/1/1",
        "deduplication": "sha256 of normalized full document text",
        "privacy_filter": "email and phone pattern redaction" if not no_pii_redaction else "disabled",
        "tokenizer": tokenizer_info,
        "shards": shard_files,
        "token_counts": dict(counters),
        "report": dict(report),
    }
    (output / "corpus_manifest.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path, help="licensed JSONL source manifest")
    parser.add_argument("output", type=Path)
    parser.add_argument("--vocab-size", type=int, default=32_000)
    parser.add_argument("--min-chars", type=int, default=80)
    parser.add_argument("--max-replacement-ratio", type=float, default=0.01)
    parser.add_argument("--shard-documents", type=int, default=10_000)
    parser.add_argument("--keep-urls", action="store_true")
    parser.add_argument("--no-pii-redaction", action="store_true")
    args = parser.parse_args()
    if args.vocab_size < 256 or args.min_chars < 1 or args.shard_documents < 1:
        parser.error("vocab-size must be >=256; min-chars and shard-documents must be positive")
    result = build(args.manifest, args.output, vocab_size=args.vocab_size,
                   min_chars=args.min_chars, max_replacement_ratio=args.max_replacement_ratio,
                   shard_documents=args.shard_documents, keep_urls=args.keep_urls,
                   no_pii_redaction=args.no_pii_redaction)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
