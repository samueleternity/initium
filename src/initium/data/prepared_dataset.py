"""Persistence helpers for datasets after their raw sources have been prepared."""

from __future__ import annotations

import os
import pickle
import re


def _dataset_name(dataset_link: str | None, dataset_type: str) -> str:
    builtins = ("graph-traversal", "text-chain", "audio-chain", "video-chain")
    if not dataset_link or dataset_link in builtins:
        return dataset_type
    first = str(dataset_link).split("+")[0].rstrip("/\\")
    name = os.path.splitext(os.path.basename(first))[0] or dataset_type
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name)


def save_prepared_dataset(dataset, directory: str, dataset_type: str, dataset_link=None) -> str:
    """Save an initialized dataset under ``directory/<type>/<source> [prepared]``."""
    name = f"{_dataset_name(dataset_link, dataset_type)} [prepared]"
    target = os.path.join(directory, dataset_type, name)
    os.makedirs(target, exist_ok=True)
    dataset._prepared_dataset_type = dataset_type
    with open(os.path.join(target, "dataset.pkl"), "wb") as f:
        pickle.dump(dataset, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"[dataset] prepared dataset saved to {target}")
    return target


def load_prepared_dataset(directory: str, dataset_type: str):
    """Restore a saved dataset, accepting its folder or a save root with one match."""
    path = os.path.join(directory, "dataset.pkl")
    if not os.path.isfile(path):
        typed_dir = os.path.join(directory, dataset_type)
        matches = []
        if os.path.isdir(typed_dir):
            matches = [
                os.path.join(typed_dir, name, "dataset.pkl")
                for name in os.listdir(typed_dir)
                if os.path.isfile(os.path.join(typed_dir, name, "dataset.pkl"))
            ]
        if len(matches) != 1:
            raise FileNotFoundError(
                f"{directory!r} must contain dataset.pkl or exactly one prepared "
                f"{dataset_type} dataset"
            )
        path = matches[0]
    with open(path, "rb") as f:
        dataset = pickle.load(f)
    saved_type = getattr(dataset, "_prepared_dataset_type", None)
    if saved_type != dataset_type:
        raise ValueError(
            f"prepared dataset type mismatch: requested {dataset_type!r}, found {saved_type!r}"
        )
    return dataset
