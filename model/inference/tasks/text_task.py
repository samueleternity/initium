"""
file: inference/tasks/text_task.py

Text-modality inference task. Delegates episode-building/decoding to
data.text.text_dataset.TextChainDataset (same pattern graph_traversal_task.py
uses relative to graph_traversal.py - training and inference can never
silently drift apart).
"""
from __future__ import annotations

from typing import List

import torch

from data.text.text_dataset import TextChainDataset
from inference.metrics import EpisodeScore
from inference.tasks.base_task import BaseInferenceTask, Episode

BUILTIN_LINKS = (None, "", "text-chain")


class TextChainTask(BaseInferenceTask):
    name = "text-chain"
    dataset_type = "text"

    def __init__(self, dataset_link=None, path_length_range=None, **kwargs):
        # Inference always tests a trained model, so --dataset-link here plays the
        # role of test_dataset_link on the training-side dataset class: a real
        # text file used as the held-out fact source, instead of the synthetic
        # seeded table. Omit (or pass one of BUILTIN_LINKS) for that default.
        link = None if dataset_link in BUILTIN_LINKS else dataset_link
        self._ds = TextChainDataset(test_dataset_link=link)
        self.input_dim, self.output_dim = self._ds.input_dim, self._ds.output_dim
        self.query_range = tuple(path_length_range) if path_length_range else self._ds.ood_query_range
        self.facts = self._ds.build_ood_facts()

    def describe(self) -> str:
        return f"{self.name} | {len(self.facts)} facts | queries {self.query_range}"

    def build_episodes(self, n: int, rng, perturbation=None) -> List[Episode]:
        severity = (perturbation or {}).get("severity", 0.0) if isinstance(perturbation, dict) else 0.0
        keys, vals_map = [k for k, _ in self.facts], dict(self.facts)
        NF = self._ds.codec.num_digits
        episodes = []
        for _ in range(n):
            nq = rng.randint(*self.query_range)
            query_keys = [rng.choice(keys) for _ in range(nq)]
            facts = list(self.facts); rng.shuffle(facts)
            inputs, targets, mask = [], [], []
            for i, (k, v) in enumerate(facts):
                inputs.append(self._ds._encode_fact(k, v, 1.0 if i == 0 else 0.0,
                                                     perturb=(severity or None), rng=rng))
                targets.append([0] * (2 * NF)); mask.append(0)
            for i, k in enumerate(query_keys):
                inputs.append(self._ds._encode_query(k, 1.0 if i == 0 else 0.0))
                targets.append([0] * (2 * NF)); mask.append(0)
            cumsum = 0
            for i, k in enumerate(query_keys):
                cumsum = (cumsum + vals_map[k]) % self._ds.label_range
                inputs.append(self._ds._encode_answer(1.0 if i == 0 else 0.0))
                targets.append(self._ds.codec.label_to_digit_targets(vals_map[k]) +
                                self._ds.codec.label_to_digit_targets(cumsum))
                mask.append(1)
            episodes.append(Episode(torch.stack(inputs), torch.tensor(targets, dtype=torch.long),
                                     torch.tensor(mask, dtype=torch.float32), meta={"num_queries": nq}))
        return episodes

    def score_episode(self, output, episode: Episode, verbose: bool = False) -> EpisodeScore:
        codec, D = self._ds.codec, self._ds.codec.label_dim
        answer_idx = (episode.mask == 1).nonzero(as_tuple=True)[0]
        f = {"value": [0, 0], "cumsum": [0, 0]}
        n_correct = 0
        for idx in answer_idx:
            v_pred, c_pred = codec.decode_field(output[idx][0:D]), codec.decode_field(output[idx][D:2 * D])
            td = episode.target[idx].tolist(); nd = codec.num_digits
            v_tgt = int("".join(map(str, td[0:nd]))); c_tgt = int("".join(map(str, td[nd:2 * nd])))
            v_ok, c_ok = v_pred == v_tgt, c_pred == c_tgt
            f["value"][0] += int(v_ok); f["value"][1] += 1
            f["cumsum"][0] += int(c_ok); f["cumsum"][1] += 1
            n_correct += int(v_ok) + int(c_ok)
            if verbose:
                print(f"  value {v_pred}/{v_tgt} ok={v_ok} | cumsum {c_pred}/{c_tgt} ok={c_ok}")
        n = 2 * len(answer_idx)
        return EpisodeScore(n_items=n, n_correct=n_correct, perfect=(n_correct == n),
                             group=len(answer_idx), fields={k: (v[0], v[1]) for k, v in f.items()})