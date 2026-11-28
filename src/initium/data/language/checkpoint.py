"""Resumable checkpoint helpers with tokenizer/config and data-position state."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import random
import tempfile

import torch


def canonical_config_hash(config: dict) -> str:
    payload = json.dumps(config, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def save_checkpoint(path, *, model, optimizer, step: int, config: dict,
                    tokenizer_path: str | Path, data_state: dict, scheduler=None,
                    extra_state: dict | None = None):
    """Atomically persist enough state to resume the same data/config run."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "step": int(step),
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "config": config,
        "config_sha256": canonical_config_hash(config),
        "tokenizer_sha256": file_sha256(tokenizer_path),
        "data_state": data_state,
        "extra_state": extra_state or {},
        "python_rng": random.getstate(),
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }
    fd, temp_name = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".tmp", dir=target.parent)
    os.close(fd)
    try:
        torch.save(payload, temp_name)
        os.replace(temp_name, target)
    finally:
        if os.path.exists(temp_name):
            os.remove(temp_name)


def load_checkpoint(path, *, model, optimizer, config: dict, tokenizer_path: str | Path,
                    scheduler=None, map_location="cpu") -> dict:
    payload = torch.load(path, map_location=map_location, weights_only=False)
    if payload.get("schema_version") != 1:
        raise ValueError("unsupported language checkpoint schema")
    if payload.get("config_sha256") != canonical_config_hash(config):
        raise ValueError("run config differs from checkpoint config; resume would not be reproducible")
    if payload.get("tokenizer_sha256") != file_sha256(tokenizer_path):
        raise ValueError("tokenizer differs from checkpoint tokenizer")
    model.load_state_dict(payload["model"])
    optimizer.load_state_dict(payload["optimizer"])
    if scheduler is not None and payload.get("scheduler") is not None:
        scheduler.load_state_dict(payload["scheduler"])
    random.setstate(payload["python_rng"])
    torch.set_rng_state(payload["torch_rng"])
    if torch.cuda.is_available() and payload.get("cuda_rng") is not None:
        torch.cuda.set_rng_state_all(payload["cuda_rng"])
    return {"step": payload["step"], "data_state": payload["data_state"],
            "extra_state": payload.get("extra_state", {})}
