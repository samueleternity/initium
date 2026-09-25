"""
file: data/audio/audio_dataset.py

Audio modality dataset: AudioChainDataset(KVChainDataset). Default (no
dataset_link): synthetic frequency-bin(key)->amplitude-bin(value)
tone-event facts, drawn fresh each episode (unchanged behavior). With
dataset_link (a path to a real audio file, needs torchaudio): each STFT
frame is quantized to a dominant-frequency-bin id (data/common/real_data.py
audio_token_stream) and turned into a real fact pool, split into a
disjoint TRAIN/TEST pool the same way text_dataset.py does -- the audio
analogue of the graph dataset's held-out London Underground graph.
test_dataset_link (a second audio file) can be given to use a genuinely
different recording as the test set instead of a held-out slice of the
same one.

Depth axis: number of tone-events (utterance-duration analogue). Fields:
value (per-event recognition), cumsum (chained running-energy state).
Robustness axis: additive noise -- softly blends the clean value one-hot
with a second random one-hot, weighted by severity, instead of a hard
digit flip.
"""

import random

import torch

from data.common.chain_task import KVChainDataset
from data.common.real_data import build_audio_kv_pool, split_train_test_facts

AUDIO_CURRICULUM = [
    (3, 2),
    (3, 3),
    (5, 3),
    (5, 4),
    (8, 4),
    (8, 5),
    (10, 5),
    (10, 6),
    (12, 6),
    (12, 7),
    (15, 8),
    (15, 9),
    (20, 10),
    (20, 12),
]
AUDIO_LESSON_NR_CELLS = [128, 128, 128, 128, 160, 160, 160, 160, 192, 192, 192, 192, 256, 256]
assert len(AUDIO_CURRICULUM) == len(AUDIO_LESSON_NR_CELLS)


class AudioChainDataset(KVChainDataset):
    name = "audio-chain"

    def __init__(self, dataset_link=None, test_dataset_link=None):
        # dataset_link / test_dataset_link each accept a single file, a
        # directory (an entire folder of clips), a glob pattern
        # ("/data/audio_clips/*.wav"), or a '+'-joined list of these -- see
        # data/common/real_data.py's resolve_link_paths. Each matched file
        # contributes its own facts (e.g. 5 clips with different
        # frequency/amplitude profiles) to one combined pool.
        self._table = AUDIO_CURRICULUM
        self._lesson_nr_cells = AUDIO_LESSON_NR_CELLS
        self._fact_pool = None
        self._test_fact_pool = None
        if dataset_link is not None:
            pool = build_audio_kv_pool(dataset_link, self.label_range)
            if test_dataset_link is not None:
                self._fact_pool = pool
                self._test_fact_pool = build_audio_kv_pool(test_dataset_link, self.label_range)
            else:
                self._fact_pool, self._test_fact_pool = split_train_test_facts(pool)
            print(
                f"[audio-chain] {len(self._fact_pool)} train facts from {dataset_link} | "
                f"{len(self._test_fact_pool)} test facts"
                + (f" from {test_dataset_link}" if test_dataset_link else " (held-out split)")
            )
        elif test_dataset_link is not None:
            # Inference-only: no training pool needed, just a fixed real
            # test pool -- every inference/tasks/*_task.py passes
            # --dataset-link through as test_dataset_link.
            self._test_fact_pool = build_audio_kv_pool(test_dataset_link, self.label_range)
            print(f"[audio-chain] {len(self._test_fact_pool)} test facts from {test_dataset_link}")

    def _synthetic_ood_facts(self, n_facts: int = 20):
        rng = random.Random(9001)  # different seed -> different "speaker profile"
        keys = rng.sample(range(self.label_range), n_facts)
        vals = [rng.randrange(self.label_range) for _ in range(n_facts)]
        return list(zip(keys, vals))

    def perturb_fact(self, key_vec: torch.Tensor, value_vec: torch.Tensor, rng, severity: float):
        if severity <= 0:
            return key_vec, value_vec
        noise_vec = torch.zeros_like(value_vec)
        nd, db = self.codec.num_digits, self.codec.digit_base
        for d in range(nd):
            noise_vec[d * db + rng.randrange(db)] = 1.0
        return key_vec, (1.0 - severity) * value_vec + severity * noise_vec
