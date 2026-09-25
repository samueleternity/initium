"""
file: data/text/text_dataset.py

Text modality dataset: TextChainDataset(KVChainDataset). Default (no
dataset_link): "facts" are token(key)->token(value) associations drawn
fresh each episode from the full label space (unchanged synthetic
behavior). With dataset_link (a path to a real text file): a byte-level
BPE tokenizer (data/common/real_data.py) is trained on the file, the
resulting token stream is turned into a real (key,value) fact pool, and
that pool is split into a disjoint TRAIN pool (used by build_episode
instead of random draws) and TEST pool (used by build_ood_facts instead of
the synthetic seeded table below) -- the text analogue of the graph
dataset's "train on synthetic graphs, test on the held-out London
Underground" split. test_dataset_link (a second, separate text file) can
be given to use an entirely different source as the test set instead of a
held-out slice of the same file.

Depth axis: number of queries (reasoning-chain-length / context-length
analogue). Fields: value (lookup), cumsum (chained state).
"""

import random

from data.common.chain_task import KVChainDataset
from data.common.real_data import build_text_kv_pool, split_train_test_facts

TEXT_CURRICULUM = [
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
TEXT_LESSON_NR_CELLS = [128, 128, 128, 128, 160, 160, 160, 160, 192, 192, 192, 192, 256, 256]
assert len(TEXT_CURRICULUM) == len(TEXT_LESSON_NR_CELLS)


class TextChainDataset(KVChainDataset):
    name = "text-chain"

    def __init__(self, dataset_link: str = None, test_dataset_link: str = None):
        self._table = TEXT_CURRICULUM
        self._lesson_nr_cells = TEXT_LESSON_NR_CELLS
        self._fact_pool = None
        self._test_fact_pool = None
        if dataset_link is not None:
            pool = build_text_kv_pool(dataset_link, self.label_range)
            if test_dataset_link is not None:
                self._fact_pool = pool
                self._test_fact_pool = build_text_kv_pool(test_dataset_link, self.label_range)
            else:
                self._fact_pool, self._test_fact_pool = split_train_test_facts(pool)
            print(
                f"[text-chain] {len(self._fact_pool)} train facts from {dataset_link} | "
                f"{len(self._test_fact_pool)} test facts"
                + (f" from {test_dataset_link}" if test_dataset_link else " (held-out split)")
            )
        elif test_dataset_link is not None:
            # Inference-only: no training pool needed, just a fixed real test
            # pool -- every inference/tasks/*_task.py passes --dataset-link
            # through as test_dataset_link, since inference only ever tests
            # an already-trained model (see this class's own docstring).
            self._test_fact_pool = build_text_kv_pool(test_dataset_link, self.label_range)
            print(f"[text-chain] {len(self._test_fact_pool)} test facts from {test_dataset_link}")

    def _synthetic_ood_facts(self, n_facts: int = 20):
        rng = random.Random(4242)  # fixed held-out vocabulary
        keys = rng.sample(range(self.label_range), n_facts)
        vals = [rng.randrange(self.label_range) for _ in range(n_facts)]
        return list(zip(keys, vals))
