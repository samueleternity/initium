"""
file: inference/tasks/multimodal_task.py

Multimodal inference task: delegates to data.multimodal.multimodal_dataset.
MultimodalDataset the same way text/audio/video tasks delegate to their
training-side dataset classes.
"""

from __future__ import annotations

from data.multimodal.multimodal_dataset import MultimodalDataset

from inference.metrics import EpisodeScore
from inference.tasks.base_task import BaseInferenceTask, Episode


class MultimodalTask(BaseInferenceTask):
    dataset_type = "multimodal"

    def __init__(self, modalities, path_length_range=None, **kwargs):
        # Inference always tests a trained model, so each per-modality real
        # source given via --dataset-link (e.g. "video:clip.mp4+audio:clip.mp4")
        # is wired in as that modality's test_dataset_link, matching every
        # other *_task.py's own convention (see e.g. text_task.py).
        self._ds = MultimodalDataset(modalities, link_role="test_dataset_link")
        self.name = self._ds.name
        self.input_dim, self.output_dim = self._ds.input_dim, self._ds.output_dim
        self.query_range = (
            tuple(path_length_range) if path_length_range else self._ds.primary.ood_query_range
        )

    def describe(self) -> str:
        real = [
            m for m, s in zip(self._ds.modalities, self._ds.subs) if s._test_fact_pool is not None
        ]
        src = f" | real data: {', '.join(real)}" if real else ""
        return f"{self.name} | queries {self.query_range}{src}"

    def build_episodes(self, n: int, rng, perturbation=None) -> list[Episode]:
        episodes = []
        for _ in range(n):
            inp, tgt, mask, nq = self._ds._build_fused_episode(
                (15, 20), self.query_range, rng=rng, use_test_pool=True
            )
            episodes.append(Episode(inp, tgt, mask, meta={"num_queries": nq}))
        return episodes

    def score_episode(self, output, episode: Episode, verbose: bool = False) -> EpisodeScore:
        codec, D = self._ds.primary.codec, self._ds.primary.codec.label_dim
        answer_idx = (episode.mask == 1).nonzero(as_tuple=True)[0]
        f = {"value": [0, 0], "cumsum": [0, 0]}
        n_correct = 0
        for idx in answer_idx:
            v_pred, c_pred = (
                codec.decode_field(output[idx][0:D]),
                codec.decode_field(output[idx][D : 2 * D]),
            )
            td = episode.target[idx].tolist()
            nd = codec.num_digits
            v_tgt = int("".join(map(str, td[0:nd])))
            c_tgt = int("".join(map(str, td[nd : 2 * nd])))
            v_ok, c_ok = v_pred == v_tgt, c_pred == c_tgt
            f["value"][0] += int(v_ok)
            f["value"][1] += 1
            f["cumsum"][0] += int(c_ok)
            f["cumsum"][1] += 1
            n_correct += int(v_ok) + int(c_ok)
        n = 2 * len(answer_idx)
        return EpisodeScore(
            n_items=n,
            n_correct=n_correct,
            perfect=(n_correct == n),
            group=len(answer_idx),
            fields={k: (v[0], v[1]) for k, v in f.items()},
        )
