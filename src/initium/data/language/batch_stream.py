"""Deterministic, resumable JSONL shard batching with bounded shard memory."""

from __future__ import annotations

import json
from pathlib import Path
import random


class JsonlBatchStream:
    """Shuffle rows within one shard and shard order; checkpoint exact cursors."""

    def __init__(self, paths, *, batch_size: int, seed: int, split: str = "train", state=None):
        self.paths = [str(Path(path)) for path in paths]
        if not self.paths:
            raise ValueError("batch stream needs at least one shard")
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        self.batch_size, self.seed, self.split = int(batch_size), int(seed), split
        self.epoch = int((state or {}).get("epoch", 0))
        self.order = list((state or {}).get("order", []))
        self.file_cursor = int((state or {}).get("file_cursor", 0))
        self.row_cursor = int((state or {}).get("row_cursor", 0))
        self._rows = None
        if not self.order:
            self._new_epoch_order()

    def _new_epoch_order(self):
        self.order = list(range(len(self.paths)))
        random.Random(self.seed + self.epoch * 1_000_003).shuffle(self.order)
        self.file_cursor = 0
        self.row_cursor = 0
        self._rows = None

    def _load_current(self):
        index = self.order[self.file_cursor]
        with Path(self.paths[index]).open(encoding="utf-8") as stream:
            self._rows = [row for line in stream if line.strip()
                          if (row := json.loads(line)).get("split", "train") == self.split]
        random.Random(self.seed + self.epoch * 1_000_003 + index).shuffle(self._rows)

    def next_batch(self):
        batch = []
        visited_files = 0
        while len(batch) < self.batch_size:
            if self.file_cursor >= len(self.order):
                if not batch and visited_files >= len(self.order):
                    raise ValueError(f"no JSONL rows found for split {self.split!r}")
                self.epoch += 1
                self._new_epoch_order()
            if self._rows is None:
                self._load_current()
            if self.row_cursor >= len(self._rows):
                self.file_cursor += 1
                visited_files += 1
                self.row_cursor = 0
                self._rows = None
                continue
            take = min(self.batch_size - len(batch), len(self._rows) - self.row_cursor)
            batch.extend(self._rows[self.row_cursor:self.row_cursor + take])
            self.row_cursor += take
        return batch

    def state_dict(self):
        return {"epoch": self.epoch, "order": list(self.order),
                "file_cursor": self.file_cursor, "row_cursor": self.row_cursor}

    def load_state_dict(self, state):
        if any(index < 0 or index >= len(self.paths) for index in state.get("order", [])):
            raise ValueError("saved batch-stream order does not match the configured shard set")
        self.epoch = int(state["epoch"])
        self.order = list(state["order"])
        self.file_cursor = int(state["file_cursor"])
        self.row_cursor = int(state["row_cursor"])
        self._rows = None
