"""
file: inference/cache/cache_registry.py

Single entry point used by run_inference.py (counterpart of data/dataset_registry.py):
    caches, notes, fingerprint = setup_caches(spec, ...)
Adding a cache type = subclass BaseCache + one line in CACHE_CLASSES.
"""
from __future__ import annotations

import os
from typing import List

from inference.cache.base_cache import BaseCache
from inference.cache.keys import model_fingerprint
from inference.cache.prefix_cache import PrefixStateCache
from inference.cache.result_cache import ResultCache
from inference.cache.stores import DiskStore, MemoryStore, TieredStore

CACHE_CLASSES = {"prefix": PrefixStateCache, "result": ResultCache}
_OFF = {"", "off", "none", "false", "0", "no"}


def parse_cache_spec(spec: str) -> List[str]:
    s = (spec or "off").strip().lower()
    if s in _OFF:
        return []
    if s in ("all", "on"):
        return list(CACHE_CLASSES)
    names = []
    for part in s.replace("+", ",").split(","):
        part = part.strip()
        if not part:
            continue
        if part not in CACHE_CLASSES:
            raise ValueError(f"unknown cache type {part!r}; expected off, all, or a comma list of "
                             f"{sorted(CACHE_CLASSES)}")
        if part not in names:
            names.append(part)
    return names


def setup_caches(spec, *, ckpt, device, sampled_writes, deterministic_write, ablate_memory,
                 cache_dir=None, ram_mb=512.0, disk_mb=2048.0, clear=False,
                 allow_stochastic=False):
    """-> (caches, notes, model_fingerprint|None). Returns no caches (with a note) when
    the requested caching cannot be done safely."""
    notes: List[str] = []
    names = parse_cache_spec(spec)
    if not names:
        return [], notes, None

    if sampled_writes and not allow_stochastic:
        notes.append("caching disabled: this model samples its write vectors (beta>0 without "
                     "--deterministic-write), so every forward pass is random and cached "
                     "results/states would not be reproducible. Use --deterministic-write, or "
                     "--cache-allow-stochastic to override.")
        return [], notes, None
    if sampled_writes:
        notes.append("WARNING: caching a stochastic model -- cached entries are ONE random draw "
                     "reused for every hit (and across runs if --cache-dir is set); "
                     "verification is skipped.")

    print("[cache] fingerprinting model weights (once) ...")
    fp = model_fingerprint(ckpt, {"deterministic_write": bool(deterministic_write),
                                  "sampled_writes": bool(sampled_writes),
                                  "ablate_memory": bool(ablate_memory)})
    disk = None
    if cache_dir:
        disk = DiskStore(os.path.join(cache_dir, fp), int(disk_mb * 1e6))
        if clear:
            disk.clear()
            notes.append(f"disk cache cleared: {disk.root}")
    store = TieredStore(MemoryStore(int(ram_mb * 1e6)), disk)
    caches: List[BaseCache] = [CACHE_CLASSES[n](store, fp, device) for n in names]
    return caches, notes, fp