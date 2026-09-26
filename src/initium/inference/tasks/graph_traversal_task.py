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

import torch

from initium.data.common.graph_io import build_graph_from_raw_edges, load_raw_edges
from initium.data.graph_traversal.graph_traversal import (
    INPUT_DIM,
    OOD_PATH_LENGTH_RANGE,
    TRIPLE_DIM,
    build_london_underground_eval,
    build_traversal_episode_from_graph,
    decode_prediction,
    encode_triple,
    triple_to_digit_targets,
)
from initium.inference.metrics import EpisodeScore
from initium.inference.tasks.base_task import BaseInferenceTask, Episode

BUILTIN_LINKS = (None, "", "graph-traversal", "london", "london-underground")


class GraphTraversalTask(BaseInferenceTask):
    name = "graph-traversal"
    dataset_type = "graph"
    input_dim = INPUT_DIM
    output_dim = TRIPLE_DIM

    def __init__(
        self,
        dataset_link=None,
        path_length_range=None,
        shared_context=False,
        num_contexts=1,
        prepared_dataset=None,
    ):
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

        if prepared_dataset is not None:
            graph = prepared_dataset._custom_test_graph or prepared_dataset._custom_graph
            if graph is None:
                self.edges, self.node_labels, self.adjacency = build_london_underground_eval()
                self.source_desc = "London Underground (built-in prepared dataset)"
            else:
                self.edges, self.node_labels, self.adjacency = graph
                self.source_desc = "saved prepared graph"
        elif dataset_link in BUILTIN_LINKS:
            self.edges, self.node_labels, self.adjacency = build_london_underground_eval()
            self.source_desc = "London Underground (built-in)"
        else:
            # dataset_link may be a single edge file, a directory, a glob
            # pattern, or a '+'-joined list of these -- load_raw_edges
            # resolves it (data/common/real_data.py's resolve_link_paths)
            # and raises its own clear FileNotFoundError if nothing
            # matches, so no separate isfile guard is needed here.
            self.edges, self.node_labels, self.adjacency = build_graph_from_raw_edges(
                load_raw_edges(dataset_link)
            )
            self.source_desc = f"source: {dataset_link}"

        if not any(self.adjacency[i] for i in self.adjacency):
            raise ValueError("test graph has no outgoing edges; cannot build traversal episodes")

    def describe(self) -> str:
        shared = f" | shared context x{self.num_contexts}" if self.shared_context else ""
        return (
            f"{self.name} | {self.source_desc} | {len(self.node_labels)} nodes, "
            f"{len(self.edges)} edges | walk length {self.path_length_range}{shared}"
        )

    def build_episodes(self, n: int, rng, perturbation=None) -> list[Episode]:
        # perturbation: not wired for graph yet (see base_task.py) -- accepted
        # and ignored so the shared BaseInferenceTask interface stays uniform.
        episodes: list[Episode] = []
        attempts = 0
        if self.shared_context:
            return self._build_shared_context_episodes(n, rng)
        while len(episodes) < n:
            attempts += 1
            if attempts > 100 * n + 1000:
                raise RuntimeError("could not build enough episodes from this graph")
            ep = build_traversal_episode_from_graph(
                self.edges,
                self.node_labels,
                self.adjacency,
                len(self.node_labels),
                self.path_length_range,
                rng=rng,
            )
            if ep is None:
                continue
            input_seq, target_digits, answer_mask = ep
            episodes.append(
                Episode(
                    input_seq,
                    target_digits,
                    answer_mask,
                    meta={"walk_length": int(answer_mask.sum().item())},
                )
            )
        return episodes

        # ---- shared-context episodes (static prefix -> cacheable) ----------------

    def _make_context(self, rng) -> torch.Tensor:
        """One fixed, shuffled edge listing: the static prefix shared by many queries."""
        shuffled = list(self.edges)
        rng.shuffle(shuffled)
        return torch.stack(
            [
                encode_triple(s, e, d, 1.0 if i == 0 else 0.0, 0.0)
                for i, (s, e, d) in enumerate(shuffled)
            ]
        )

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
            targets.append(
                triple_to_digit_targets(
                    self.node_labels[src_idx], edge_label, self.node_labels[dst_idx]
                )
            )
            mask.append(1)
        return (
            torch.stack(steps),
            torch.tensor(targets, dtype=torch.long),
            torch.tensor(mask, dtype=torch.float32),
        )

    def _build_shared_context_episodes(self, n: int, rng) -> list[Episode]:
        contexts = [self._make_context(rng) for _ in range(self.num_contexts)]
        episodes: list[Episode] = []
        attempts = 0
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
            episodes.append(
                Episode(
                    torch.cat([ctx_seq, q_in]),
                    torch.cat([torch.zeros(P, 9, dtype=torch.long), q_tgt]),
                    torch.cat([torch.zeros(P), q_mask]),
                    meta={"walk_length": int(q_mask.sum().item()), "context_id": k},
                    cache_boundaries=[P],
                )
            )
        return episodes

    def score_episode(self, output, episode: Episode, verbose: bool = False) -> EpisodeScore:
        answer_idx = (episode.mask == 1).nonzero(as_tuple=True)[0]
        f = {
            "src": [0, 0],
            "edge": [0, 0],
            "dst": [0, 0],
            "dst|src+edge": [0, 0],
            "src|prev_dst_ok": [0, 0],
        }
        prev_dst_ok = None
        n_correct = 0
        for hop, idx in enumerate(answer_idx, start=1):
            pred = decode_prediction(output[idx])
            td = episode.target[idx].tolist()
            tgt = (
                int("".join(map(str, td[0:3]))),
                int("".join(map(str, td[3:6]))),
                int("".join(map(str, td[6:9]))),
            )
            ok = pred == tgt
            n_correct += int(ok)
            for name, p, t in zip(("src", "edge", "dst"), pred, tgt):
                f[name][0] += int(p == t)
                f[name][1] += 1
            if pred[0] == tgt[0] and pred[1] == tgt[1]:  # lookup check
                f["dst|src+edge"][1] += 1
                f["dst|src+edge"][0] += int(pred[2] == tgt[2])
            if prev_dst_ok:  # chain check
                f["src|prev_dst_ok"][1] += 1
                f["src|prev_dst_ok"][0] += int(pred[0] == tgt[0])
            prev_dst_ok = pred[2] == tgt[2]
            hp = f.setdefault(f"hop{hop}", [0, 0])  # triple acc per hop position
            hp[0] += int(ok)
            hp[1] += 1
            if verbose:
                print(f"  Pred: {pred} | Target: {tgt} | Correct: {ok}")
        n = len(answer_idx)
        return EpisodeScore(
            n_items=n,
            n_correct=n_correct,
            perfect=(n_correct == n),
            group=n,
            fields={k: (v[0], v[1]) for k, v in f.items()},
        )
