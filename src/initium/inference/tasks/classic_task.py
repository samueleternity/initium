"""Inference adapter for the additive long-window classic task family."""

from __future__ import annotations

import torch

from initium.data.dataset_registry import get_dataset
from initium.config.classic_config import (
    CLASSIC_PROBE_DISTANCES,
    CLASSIC_PROBE_GAMMA,
    CLASSIC_WINDOW_SIZE,
)
from initium.inference.metrics import EpisodeScore
from initium.inference.tasks.base_task import BaseInferenceTask, Episode


class ClassicInferenceTask(BaseInferenceTask):
    def __init__(
        self,
        dataset_type,
        dataset_link=None,
        prepared_dataset=None,
        classic_modalities="text+audio",
        classic_window=CLASSIC_WINDOW_SIZE,
        probe_distances=CLASSIC_PROBE_DISTANCES,
        probe_gamma=CLASSIC_PROBE_GAMMA,
        test_dataset_link=None,
    ):
        self.dataset_type = dataset_type
        if dataset_type == "multimodal-classic":
            modalities = [m for m in classic_modalities.split("+") if m]
        else:
            modalities = [dataset_type.removesuffix("-classic")]
        # In inference the supplied source is a held-out source. Passing it as
        # both links constructs the same shared ClassicDataset class while its
        # test split is the complete supplied stream.
        self._ds = prepared_dataset or get_dataset(
            dataset_type,
            dataset_link,
            test_dataset_link=test_dataset_link or dataset_link,
            modalities=modalities,
            window_size=classic_window,
            probe_distances=probe_distances,
            probe_gamma=probe_gamma,
        )
        if prepared_dataset is not None:
            self._ds.configure(
                window_size=classic_window,
                probe_distances=probe_distances,
                probe_gamma=probe_gamma,
            )
        self.name = self._ds.name
        self.input_dim, self.output_dim = self._ds.input_dim, self._ds.output_dim

    def describe(self):
        return (
            f"{self.name} | window {self._ds.window_size} | probe distances "
            f"{self._ds.probe_distances} | gamma {self._ds.probe_gamma:g}"
        )

    def build_episodes(self, n: int, rng, perturbation=None):
        episodes = []
        for _ in range(n):
            x, target, mask, meta = self._ds._episode(self._ds.test_sequences, rng)
            episodes.append(Episode(x, target, mask, meta=meta))
        return episodes

    def score_episode(self, output, episode: Episode, verbose=False):
        pos = int(episode.meta["probe_pos"])
        expected = self._ds.codec.decode_field(episode.target[pos, 1].float())
        predicted = self._ds.codec.decode_field(output[pos])
        correct = int(predicted == expected)
        if verbose:
            print(
                f"  probe distance {episode.meta['distance']} "
                f"({episode.meta['modality']}): predicted {predicted}, target {expected}, "
                f"correct={bool(correct)}"
            )
        return EpisodeScore(
            n_items=1,
            n_correct=correct,
            perfect=bool(correct),
            group=1,
            fields={"probe": (correct, 1)},
        )

    def encode_generated_token(self, token, position=0):
        return self._ds.encode_generated_token(token, position)

    def decode_logits(self, logits):
        return self._ds.codec.decode_field(logits)

    def sample_token(self, logits, temperature=1.0, top_k=0):
        return self._ds.sample_token(logits, temperature, top_k)

    def generation_prompts(self, n, rng):
        prompts = []
        for _ in range(n):
            x, target, meta = self._ds.generation_probe(rng)
            end = int(meta["probe_pos"]) + 1
            prompts.append((x[:end], target, meta))
        return prompts
