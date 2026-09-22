"""
file: data/audio/audio_dataset.py

Audio modality dataset: AudioChainDataset(KVChainDataset). "Facts" are
frequency-bin(key)->amplitude-bin(value) tone-event associations; "queries"
are a sequence of tone identities the model must recognize and accumulate a
running (mod-1000) signal-energy checksum over.

Depth axis: number of tone-events (utterance-duration analogue).
Fields: value (per-event recognition, ~frame/phoneme accuracy), cumsum
(chained running-energy state).
OOD axis: a fixed held-out "speaker profile" (a different fixed seed's
frequency-to-amplitude table) -- the audio analogue of an unseen speaker.
Robustness axis: additive noise -- softly blends the clean value one-hot
with a second random one-hot, weighted by severity, instead of a hard digit
flip (a better analogue of acoustic noise than token corruption).
"""
import random

import torch

from data.common.chain_task import KVChainDataset

AUDIO_CURRICULUM = [
    (3, 2), (3, 3), (5, 3), (5, 4), (8, 4), (8, 5),
    (10, 5), (10, 6), (12, 6), (12, 7), (15, 8), (15, 9), (20, 10), (20, 12),
]
AUDIO_LESSON_NR_CELLS = [128, 128, 128, 128, 160, 160, 160, 160, 192, 192, 192, 192, 256, 256]
assert len(AUDIO_CURRICULUM) == len(AUDIO_LESSON_NR_CELLS)


class AudioChainDataset(KVChainDataset):
    name = "audio-chain"

    def __init__(self):
        self._table = AUDIO_CURRICULUM
        self._lesson_nr_cells = AUDIO_LESSON_NR_CELLS

    def build_ood_facts(self, n_facts: int = 20):
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