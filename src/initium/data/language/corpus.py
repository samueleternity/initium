"""Manifest-backed JSONL corpus readers and deterministic streaming shards."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterator


def read_manifest(path: str | Path) -> Iterator[dict]:
    with Path(path).open(encoding="utf-8") as stream:
        for line_no, line in enumerate(stream, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            required = ("path", "license", "source")
            absent = [key for key in required if not row.get(key)]
            if absent:
                raise ValueError(f"manifest line {line_no} is missing {', '.join(absent)}")
            yield row


def iter_token_shards(shard_paths, *, split: str | None = None, max_length: int = 2048):
    """Yield packed token lists from serialized shard JSONL records."""
    for shard in shard_paths:
        with Path(shard).open(encoding="utf-8") as stream:
            for line in stream:
                if not line.strip():
                    continue
                row = json.loads(line)
                if split is not None and row.get("split") != split:
                    continue
                tokens = row["input_ids"]
                for start in range(0, len(tokens), max_length):
                    chunk = tokens[start : start + max_length]
                    if len(chunk) >= 2:
                        yield {"input_ids": chunk, "split": row["split"], "doc_id": row["doc_id"]}


def stable_split(doc_id: str, ratios=(0.98, 0.01, 0.01)) -> str:
    value = int.from_bytes(hashlib.sha256(doc_id.encode("utf-8")).digest()[:8], "big") / 2**64
    if value < ratios[0]:
        return "train"
    if value < ratios[0] + ratios[1]:
        return "validation"
    return "test"
