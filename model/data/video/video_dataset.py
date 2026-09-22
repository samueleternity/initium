"""
file: data/video/video_dataset.py

Video modality dataset: VideoChainDataset(KVChainDataset). Default (no
dataset_link): synthetic object-identity(key)->action/state-identity(value)
facts, drawn fresh each episode (unchanged behavior). With dataset_link (a
path to a real video file, needs opencv-python): each frame is downsampled
and quantized to an intensity-bin id (data/common/real_data.py
video_token_stream) and turned into a real fact pool, split into a
disjoint TRAIN/TEST pool the same way text_dataset.py does -- the video
analogue of the graph dataset's held-out London Underground graph.
test_dataset_link (a second video file) can be given to use a genuinely
different clip as the test set instead of a held-out slice of the same
one.

Depth axis: number of queried events (event-chain-length / temporal-
horizon analogue). Fields: value (object-action binding accuracy), cumsum
(chained event-ordering state).
Robustness axis: frame dropping / occlusion -- the WHOLE fact observation
(key AND value) is zeroed with prob=severity, rather than corrupted in
place (closer to "this frame was dropped/occluded" than to token noise).
"""
import random

import torch

from data.common.chain_task import KVChainDataset
from data.common.real_data import build_video_kv_pool, split_train_test_facts

VIDEO_CURRICULUM = [
    (3, 2), (3, 3), (5, 3), (5, 4), (8, 4), (8, 5),
    (10, 5), (10, 6), (12, 6), (12, 7), (15, 8), (15, 9), (20, 10), (20, 12),
]
VIDEO_LESSON_NR_CELLS = [128, 128, 128, 128, 160, 160, 160, 160, 192, 192, 192, 192, 256, 256]
assert len(VIDEO_CURRICULUM) == len(VIDEO_LESSON_NR_CELLS)


class VideoChainDataset(KVChainDataset):
    name = "video-chain"

    def __init__(self, dataset_link: str = None, test_dataset_link: str = None):
        self._table = VIDEO_CURRICULUM
        self._lesson_nr_cells = VIDEO_LESSON_NR_CELLS
        self._fact_pool = None
        self._test_fact_pool = None
        if dataset_link is not None:
            pool = build_video_kv_pool(dataset_link, self.label_range)
            if test_dataset_link is not None:
                self._fact_pool = pool
                self._test_fact_pool = build_video_kv_pool(test_dataset_link, self.label_range)
            else:
                self._fact_pool, self._test_fact_pool = split_train_test_facts(pool)
            print(f"[video-chain] {len(self._fact_pool)} train facts from {dataset_link} | "
                  f"{len(self._test_fact_pool)} test facts"
                  + (f" from {test_dataset_link}" if test_dataset_link else " (held-out split)"))
        elif test_dataset_link is not None:
            # Inference-only real test pool -- see text_dataset.py's twin fix.
            self._test_fact_pool = build_video_kv_pool(test_dataset_link, self.label_range)
            print(f"[video-chain] {len(self._test_fact_pool)} test facts from {test_dataset_link}")

    def _synthetic_ood_facts(self, n_facts: int = 20):
        rng = random.Random(31337)  # unseen object/action table
        keys = rng.sample(range(self.label_range), n_facts)
        vals = [rng.randrange(self.label_range) for _ in range(n_facts)]
        return list(zip(keys, vals))

    def perturb_fact(self, key_vec: torch.Tensor, value_vec: torch.Tensor, rng, severity: float):
        if rng.random() < severity:
            return torch.zeros_like(key_vec), torch.zeros_like(value_vec)
        return key_vec, value_vec