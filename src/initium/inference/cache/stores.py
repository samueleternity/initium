"""
file: inference/cache/stores.py

Storage back-ends shared by all cache types: a byte-capped RAM LRU, an optional
on-disk store (survives across CLI invocations), and a tiered wrapper. Entries are
plain dicts {"payload", "compute_ms", "nbytes", "kind"} made only of tensors /
lists / tuples / dicts / numbers, so the disk tier can use torch.load(weights_only=True)
(no arbitrary pickle execution).
"""
from __future__ import annotations

import glob
import os
from collections import OrderedDict
from dataclasses import dataclass, asdict
from typing import Optional, Tuple

import torch


@dataclass
class CacheStats:
    lookups: int = 0
    hits: int = 0
    misses: int = 0
    stores: int = 0
    ram_hits: int = 0
    disk_hits: int = 0
    bytes_stored: int = 0
    est_saved_ms: float = 0.0        # sum of recorded cold-compute time of every reused entry

    def hit(self, tier: str, saved_ms: float) -> None:
        self.hits += 1
        self.est_saved_ms += float(saved_ms)
        if tier == "disk":
            self.disk_hits += 1
        else:
            self.ram_hits += 1

    def to_dict(self) -> dict:
        d = asdict(self)
        d["hit_rate"] = 100.0 * self.hits / max(self.lookups, 1)
        return d


class MemoryStore:
    def __init__(self, max_bytes: int):
        self.max_bytes = max_bytes
        self._d: "OrderedDict[str, dict]" = OrderedDict()
        self.bytes = 0
        self.evictions = 0

    def get(self, key: str) -> Optional[dict]:
        e = self._d.get(key)
        if e is not None:
            self._d.move_to_end(key)
        return e

    def put(self, key: str, entry: dict) -> None:
        nb = int(entry.get("nbytes", 0))
        if nb > self.max_bytes:
            return
        old = self._d.pop(key, None)
        if old is not None:
            self.bytes -= int(old.get("nbytes", 0))
        self._d[key] = entry
        self.bytes += nb
        while self.bytes > self.max_bytes and self._d:
            _, ev = self._d.popitem(last=False)
            self.bytes -= int(ev.get("nbytes", 0))
            self.evictions += 1

    def __len__(self) -> int:
        return len(self._d)


class DiskStore:
    def __init__(self, root: str, max_bytes: int):
        self.root, self.max_bytes = root, max_bytes
        os.makedirs(root, exist_ok=True)

    def _path(self, key: str) -> str:
        return os.path.join(self.root, key + ".pt")

    def get(self, key: str) -> Optional[dict]:
        p = self._path(key)
        if not os.path.isfile(p):
            return None
        try:
            entry = torch.load(p, map_location="cpu", weights_only=True)
            os.utime(p, None)                    # recency for eviction
            return entry
        except Exception:                        # corrupt / incompatible -> treat as miss
            try:
                os.remove(p)
            except OSError:
                pass
            return None

    def put(self, key: str, entry: dict) -> None:
        p = self._path(key)
        tmp = f"{p}.tmp{os.getpid()}"
        try:
            torch.save(entry, tmp)
            os.replace(tmp, p)                   # atomic
        except Exception as e:
            print(f"[cache] disk write failed for {key}: {e}")
            try:
                os.remove(tmp)
            except OSError:
                pass
            return
        self._evict()

    def _evict(self) -> None:
        files = [(os.path.getmtime(f), os.path.getsize(f), f)
                 for f in glob.glob(os.path.join(self.root, "*.pt"))]
        total = sum(s for _, s, _ in files)
        for _, size, f in sorted(files):
            if total <= self.max_bytes:
                break
            try:
                os.remove(f)
                total -= size
            except OSError:
                pass

    def clear(self) -> None:
        for f in glob.glob(os.path.join(self.root, "*.pt")):
            try:
                os.remove(f)
            except OSError:
                pass


class TieredStore:
    """RAM first, disk second (disk hits are promoted to RAM). Shared by every cache
    of a run, so --cache-ram-mb / --cache-disk-mb are TOTAL budgets."""

    def __init__(self, mem: MemoryStore, disk: Optional[DiskStore] = None):
        self.mem, self.disk = mem, disk

    def get(self, key: str) -> Optional[Tuple[dict, str]]:
        e = self.mem.get(key)
        if e is not None:
            return e, "ram"
        if self.disk is not None:
            e = self.disk.get(key)
            if e is not None:
                self.mem.put(key, e)
                return e, "disk"
        return None

    def put(self, key: str, entry: dict) -> None:
        self.mem.put(key, entry)
        if self.disk is not None:
            self.disk.put(key, entry)

    def info(self) -> dict:
        return {"ram_mb": self.mem.bytes / 1e6, "ram_entries": len(self.mem),
                "ram_evictions": self.mem.evictions,
                "disk_dir": self.disk.root if self.disk else None}