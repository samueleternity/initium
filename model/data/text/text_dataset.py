"""
file: data/text/text_dataset.py

Text modality dataset: TextChainDataset(KVChainDataset). "Facts" are
token(key)->token(value) associations (a synthetic glossary); "queries" are
a sequence of tokens the model must look up and accumulate a running
(mod-1000) checksum over - see data/common/chain_task.py's module
docstring for the shared mechanics.

Depth axis: number of queries (reasoning-chain-length / context-length
analogue). Fields: value (lookup), cumsum (chained state).
OOD axis: a fixed held-out key/value vocabulary (unseen at training time).
Robustness axis: per-fact digit-slot corruption (inherited unmodified from
KVChainDataset.perturb_fact - already exactly "token corruption").
"""
import random

from data.common.chain_task import KVChainDataset

TEXT_CURRICULUM = [
    (3, 2), (3, 3), (5, 3), (5, 4), (8, 4), (8, 5),
    (10, 5), (10, 6), (12, 6), (12, 7), (15, 8), (15, 9), (20, 10), (20, 12),
]
TEXT_LESSON_NR_CELLS = [128, 128, 128, 128, 160, 160, 160, 160, 192, 192, 192, 192, 256, 256]
assert len(TEXT_CURRICULUM) == len(TEXT_LESSON_NR_CELLS)


class TextChainDataset(KVChainDataset):
    name = "text-chain"

    def __init__(self):
        self._table = TEXT_CURRICULUM
        self._lesson_nr_cells = TEXT_LESSON_NR_CELLS

    def build_ood_facts(self, n_facts: int = 20):
        rng = random.Random(4242)  # fixed held-out vocabulary
        keys = rng.sample(range(self.label_range), n_facts)
        vals = [rng.randrange(self.label_range) for _ in range(n_facts)]
        return list(zip(keys, vals))