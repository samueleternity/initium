import numpy as np
from data.common.real_data import _pool_from_bigram_matrix, split_train_test_facts


def test_bigram_pool_and_split_are_deterministic():
    matrix = np.zeros((25,), dtype=np.int64)
    for key, value in enumerate([1, 2, 3, 4, 0]):
        matrix[key * 5 + value] = 2
    facts = _pool_from_bigram_matrix(matrix, label_range=5)
    assert facts == [(0, 1), (1, 2), (2, 3), (3, 4), (4, 0)]
    train_a, test_a = split_train_test_facts(facts, test_frac=0.2, seed=9)
    train_b, test_b = split_train_test_facts(facts, test_frac=0.2, seed=9)
    assert train_a == train_b
    assert test_a == test_b
    assert set(train_a).isdisjoint(test_a)
