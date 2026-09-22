"""
file: data/common/graph_io.py

Shared raw-edge-file loading + label-mapping for custom graph datasets -
used by BOTH the training-side graph_traversal.py (to train on a supplied
graph instead of a synthetic one) and the inference-side
graph_traversal_task.py (which used to define these functions itself).
Moved here so the two can never silently drift apart - same
train/inference-parity rationale every other data/<modality>_dataset.py
already follows relative to its inference/tasks/<modality>_task.py twin.

Format: .csv / .tsv / .txt: one edge per row, "src,dst,line" ('#' comments
and a src/source/from header row are skipped). .json: [["src","dst","line"],
...] or {"edges": [...]} (rows may also be {"src":..,"dst":..,"line":..}).
Names are mapped to distinct 0..999 labels with a fixed random.Random(seed)
unless every field is already an integer in [0,1000), in which case the
integers are used directly.
"""
from __future__ import annotations

import csv
import json
import os
import random
from typing import List, Tuple

LABEL_RANGE = 1000
_HEADER_FIRST_FIELDS = {"src", "source", "from"}


def _is_label(x: str) -> bool:
    try:
        return 0 <= int(x) < LABEL_RANGE
    except (TypeError, ValueError):
        return False


def load_raw_edges(path: str) -> List[Tuple[str, str, str]]:
    ext = os.path.splitext(path)[1].lower()
    rows = []
    if ext == ".json":
        with open(path) as f:
            data = json.load(f)
        if isinstance(data, dict):
            data = data.get("edges", data.get("raw_edges"))
        if not isinstance(data, list):
            raise ValueError(f"{path}: JSON must be a list of edges or {{'edges': [...]}}")
        rows = data
    else:
        with open(path, newline="") as f:
            for row in csv.reader(f, delimiter="\t" if ext == ".tsv" else ","):
                if not row or row[0].strip().startswith("#"):
                    continue
                rows.append(row)

    edges = []
    for i, row in enumerate(rows):
        if isinstance(row, dict):
            row = [row.get("src"), row.get("dst"), row.get("line")]
        if len(row) != 3 or any(x is None for x in row):
            raise ValueError(f"{path}: row {i} is not 'src,dst,line': {row!r}")
        row = [str(x).strip() for x in row]
        if i == 0 and row[0].lower() in _HEADER_FIRST_FIELDS:
            continue
        edges.append(tuple(row))
    if not edges:
        raise ValueError(f"{path}: no edges found")
    return edges


def build_graph_from_raw_edges(raw_edges, label_seed: int = 1234):
    """-> (edges [(src_label, edge_label, dst_label)], node_labels, adjacency)."""
    if all(_is_label(x) for e in raw_edges for x in e):
        raw_edges = [tuple(str(int(x)) for x in e) for e in raw_edges]
        numeric = True
    else:
        numeric = False

    stations = sorted({s for s, _, _ in raw_edges} | {d for _, d, _ in raw_edges})
    lines = sorted({l for _, _, l in raw_edges})

    if numeric:
        station_to_label = {s: int(s) for s in stations}
        line_to_label = {l: int(l) for l in lines}
    else:
        need = len(stations) + len(lines)
        if need > LABEL_RANGE:
            raise ValueError(f"graph has {need} distinct stations+lines; max is {LABEL_RANGE}")
        rng = random.Random(label_seed)
        all_labels = rng.sample(range(LABEL_RANGE), need)
        station_to_label = dict(zip(stations, all_labels[:len(stations)]))
        line_to_label = dict(zip(lines, all_labels[len(stations):]))

    edges = [(station_to_label[s], line_to_label[l], station_to_label[d]) for s, d, l in raw_edges]
    station_idx = {s: i for i, s in enumerate(stations)}
    adjacency = {i: [] for i in range(len(stations))}
    for s, d, l in raw_edges:
        adjacency[station_idx[s]].append((station_idx[d], line_to_label[l]))
    node_labels = [station_to_label[s] for s in stations]
    return edges, node_labels, adjacency


def load_graph(path: str, label_seed: int = 1234):
    """One-shot: raw edge file -> (edges, node_labels, adjacency)."""
    return build_graph_from_raw_edges(load_raw_edges(path), label_seed=label_seed)