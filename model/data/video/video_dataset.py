"""
file: data/video/video_dataset.py

Video modality dataset: VideoChainDataset(KVChainDataset). "Facts" are
object-identity(key)->action/state-identity(value) associations (object-
action binding, per frame); "queries" are a sequence of object identities
the model must recall the action/state of and accumulate a running
(mod-1000) event checksum over.

Depth axis: number of queried events (event-chain-length / temporal-horizon
analogue). Fields: value (object-action binding accuracy), cumsum (chained
event-ordering state).
OOD axis: a fixed held-out object/action table (unseen object identities).
Robustness axis: frame dropping / occlusion -- the WHOLE fact observation
(key AND value) is zeroed with prob=severity, rather than corrupted in
place (closer to "this frame was dropped/occluded" than to token noise).
"""
import random

import torch

from data.common.chain_task import KVChainDataset

VIDEO_CURRICULUM = [
    (3, 2), (3, 3), (5, 3), (5, 4), (8, 4), (8, 5),
    (10, 5), (10, 6), (12, 6), (12, 7), (15, 8), (15, 9), (20, 10), (20, 12),
]
VIDEO_LESSON_NR_CELLS = [128, 128, 128, 128, 160, 160, 160, 160, 192, 192, 192, 192, 256, 256]
assert len(VIDEO_CURRICULUM) == len(VIDEO_LESSON_NR_CELLS)


class VideoChainDataset(KVChainDataset):
    name = "video-chain"

    def __init__(self):
        self._table = VIDEO_CURRICULUM
        self._lesson_nr_cells = VIDEO_LESSON_NR_CELLS

    def build_ood_facts(self, n_facts: int = 20):
        rng = random.Random(31337)  # unseen object/action table
        keys = rng.sample(range(self.label_range), n_facts)
        vals = [rng.randrange(self.label_range) for _ in range(n_facts)]
        return list(zip(keys, vals))

    def perturb_fact(self, key_vec: torch.Tensor, value_vec: torch.Tensor, rng, severity: float):
        if rng.random() < severity:
            return torch.zeros_like(key_vec), torch.zeros_like(value_vec)
        return key_vec, value_vec