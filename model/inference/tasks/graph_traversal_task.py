"""
file: inference/tasks/graph_traversal_task.py

Graph-traversal inference task. Default test graph is the built-in London
Underground (identical graph/label mapping to the training-time OOD eval).
--dataset-link may instead point to your own edge file:

  .csv / .tsv / .txt : one edge per row:  src,dst,line      ('#' comment lines and a
                       header row starting with src/source/from are skipped)
  .json              : [["src","dst","line"], ...]  or  {"edges": [...]}  (rows may also
                       be {"src":..,"dst":..,"line":..})

Names (stations/lines) are mapped to distinct 0-999 labels with a fixed
random.Random(1234) exactly like the London eval. If EVERY field in the file
is an integer in 0..999, the integers are used directly as labels instead.
Episodes are built by the dataset module's own
build_traversal_episode_from_graph(), so encoding is byte-identical to training.
"""
from __future__ import annotations

import csv
import json
import os
import random
from typing import List, Tuple

import torch

from data.graph_traversal.graph_traversal import (
    INPUT_DIM, TRIPLE_DIM, LABEL_RANGE, OOD_PATH_LENGTH_RANGE,
    build_london_underground_eval, build_traversal_episode_from_graph, decode_prediction,
    encode_triple, triple_to_digit_targets,
)
from inference.metrics import EpisodeScore
from inference.tasks.base_task import BaseInferenceTask, Episode

BUILTIN_LINKS = (None, "", "graph-traversal", "london", "london-underground")
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


class GraphTraversalTask(BaseInferenceTask):
    name = "graph-traversal"
    dataset_type = "graph"
    input_dim = INPUT_DIM
    output_dim = TRIPLE_DIM

    def __init__(self, dataset_link=None, path_length_range=None,
                 shared_context=False, num_contexts=1):
        if path_length_range is not None:
            lo, hi = path_length_range
            if lo < 1 or hi < lo:
                raise ValueError(f"invalid path length range {path_length_range}")
            self.path_length_range = (int(lo), int(hi))
        else:
            self.path_length_range = tuple(OOD_PATH_LENGTH_RANGE)

        if int(num_contexts) < 1:
            raise ValueError("num_contexts must be >= 1")
        if int(num_contexts) > 1 and not shared_context:
            raise ValueError("num_contexts > 1 requires shared_context=True")
        self.shared_context, self.num_contexts = bool(shared_context), int(num_contexts)

        if dataset_link in BUILTIN_LINKS:
            self.edges, self.node_labels, self.adjacency = build_london_underground_eval()
            self.source_desc = "London Underground (built-in)"
        else:
            if not os.path.isfile(dataset_link):
                raise FileNotFoundError(f"graph test file not found: {dataset_link}")
            self.edges, self.node_labels, self.adjacency = build_graph_from_raw_edges(
                load_raw_edges(dataset_link))
            self.source_desc = f"file: {dataset_link}"

        if not any(self.adjacency[i] for i in self.adjacency):
            raise ValueError("test graph has no outgoing edges; cannot build traversal episodes")

    def describe(self) -> str:
        shared = f" | shared context x{self.num_contexts}" if self.shared_context else ""
        return (f"{self.name} | {self.source_desc} | {len(self.node_labels)} nodes, "
                f"{len(self.edges)} edges | walk length {self.path_length_range}{shared}")

    def build_episodes(self, n: int, rng) -> List[Episode]:
        episodes, attempts = [], 0
        if self.shared_context:
            return self._build_shared_context_episodes(n, rng)
        while len(episodes) < n:
            attempts += 1
            if attempts > 100 * n + 1000:
                raise RuntimeError("could not build enough episodes from this graph")
            ep = build_traversal_episode_from_graph(
                self.edges, self.node_labels, self.adjacency, len(self.node_labels),
                self.path_length_range, rng=rng)
            if ep is None:
                continue
            input_seq, target_digits, answer_mask = ep
            episodes.append(Episode(input_seq, target_digits, answer_mask,
                                    meta={"walk_length": int(answer_mask.sum().item())}))
        return episodes

        # ---- shared-context episodes (static prefix -> cacheable) ----------------
    def _make_context(self, rng) -> torch.Tensor:
        """One fixed, shuffled edge listing: the static prefix shared by many queries."""
        shuffled = list(self.edges)
        rng.shuffle(shuffled)
        return torch.stack([encode_triple(s, e, d, 1.0 if i == 0 else 0.0, 0.0)
                            for i, (s, e, d) in enumerate(shuffled)])

    def _build_query(self, rng):
        """Walk + answer steps only (mirrors the second half of
        build_traversal_episode_from_graph). -> (inputs, target_digits, mask) or None."""
        path_length = rng.randint(*self.path_length_range)
        cur = rng.randrange(len(self.node_labels))
        walk = []
        for _ in range(path_length):
            if not self.adjacency[cur]:
                break
            dst_idx, edge_label = rng.choice(self.adjacency[cur])
            walk.append((cur, edge_label, dst_idx))
            cur = dst_idx
        if not walk:
            return None
        steps, targets, mask = [], [], []
        for i, (src_idx, edge_label, _dst) in enumerate(walk):
            src = self.node_labels[src_idx] if i == 0 else None
            steps.append(encode_triple(src, edge_label, None, 1.0 if i == 0 else 0.0, 0.0))
            targets.append([0] * 9)
            mask.append(0)
        for i, (src_idx, edge_label, dst_idx) in enumerate(walk):
            steps.append(encode_triple(None, None, None, 1.0 if i == 0 else 0.0, 1.0))
            targets.append(triple_to_digit_targets(
                self.node_labels[src_idx], edge_label, self.node_labels[dst_idx]))
            mask.append(1)
        return (torch.stack(steps), torch.tensor(targets, dtype=torch.long),
                torch.tensor(mask, dtype=torch.float32))

    def _build_shared_context_episodes(self, n: int, rng) -> List[Episode]:
        contexts = [self._make_context(rng) for _ in range(self.num_contexts)]
        episodes, attempts = [], 0
        while len(episodes) < n:
            attempts += 1
            if attempts > 100 * n + 1000:
                raise RuntimeError("could not build enough episodes from this graph")
            q = self._build_query(rng)
            if q is None:
                continue
            k = len(episodes) % self.num_contexts
            ctx_seq, (q_in, q_tgt, q_mask) = contexts[k], q
            P = ctx_seq.shape[0]
            episodes.append(Episode(
                torch.cat([ctx_seq, q_in]),
                torch.cat([torch.zeros(P, 9, dtype=torch.long), q_tgt]),
                torch.cat([torch.zeros(P), q_mask]),
                meta={"walk_length": int(q_mask.sum().item()), "context_id": k},
                cache_boundaries=[P],
            ))
        return episodes

    def score_episode(self, output, episode: Episode, verbose: bool = False) -> EpisodeScore:
        answer_idx = (episode.mask == 1).nonzero(as_tuple=True)[0]
        f = {"src": [0, 0], "edge": [0, 0], "dst": [0, 0],
             "dst|src+edge": [0, 0], "src|prev_dst_ok": [0, 0]}
        prev_dst_ok = None
        n_correct = 0
        for hop, idx in enumerate(answer_idx, start=1):
            pred = decode_prediction(output[idx])
            td = episode.target[idx].tolist()
            tgt = (int("".join(map(str, td[0:3]))),
                   int("".join(map(str, td[3:6]))),
                   int("".join(map(str, td[6:9]))))
            ok = pred == tgt
            n_correct += int(ok)
            for name, p, t in zip(("src", "edge", "dst"), pred, tgt):
                f[name][0] += int(p == t)
                f[name][1] += 1
            if pred[0] == tgt[0] and pred[1] == tgt[1]:      # lookup check
                f["dst|src+edge"][1] += 1
                f["dst|src+edge"][0] += int(pred[2] == tgt[2])
            if prev_dst_ok:                                   # chain check
                f["src|prev_dst_ok"][1] += 1
                f["src|prev_dst_ok"][0] += int(pred[0] == tgt[0])
            prev_dst_ok = pred[2] == tgt[2]
            hp = f.setdefault(f"hop{hop}", [0, 0])            # triple acc per hop position
            hp[0] += int(ok)
            hp[1] += 1
            if verbose:
                print(f"  Pred: {pred} | Target: {tgt} | Correct: {ok}")
        n = len(answer_idx)
        return EpisodeScore(
            n_items=n, n_correct=n_correct, perfect=(n_correct == n), group=n,
            fields={k: (v[0], v[1]) for k, v in f.items()},
        )