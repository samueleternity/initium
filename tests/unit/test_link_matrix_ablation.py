import torch

from dnc.memory import Memory
from memory_manipulation.link_matrix_ablation import (
    AblatableSparseLinkMemory,
    patch_link_matrix,
)


def _memory():
    return Memory(input_size=8, nr_cells=8, cell_size=4, read_heads=1, independent_linears=True)


def test_dense_and_ablated_modes():
    memory = _memory()
    link = torch.rand(1, 1, 8, 8)
    write = torch.rand(1, 1, 8)
    precedence = torch.rand(1, 1, 8)
    expected = Memory.get_link_matrix(memory, link, write, precedence)
    memory.__class__ = AblatableSparseLinkMemory
    memory.link_matrix_mode = "dense"
    torch.testing.assert_close(memory.get_link_matrix(link, write, precedence), expected)
    memory.link_matrix_mode = "ablated"
    assert memory.get_link_matrix(link, write, precedence) is link


def test_sparse_topk_keeps_largest_magnitude_per_row():
    memory = _memory()
    memory.__class__ = AblatableSparseLinkMemory
    memory.link_matrix_mode = "sparse_topk"
    memory.link_matrix_topk = 3
    link = torch.zeros(1, 1, 8, 8)
    write = torch.rand(1, 1, 8)
    precedence = torch.rand(1, 1, 8)
    updated = Memory.get_link_matrix(memory, link, write, precedence)
    actual = memory.get_link_matrix(link, write, precedence)
    expected = updated * torch.zeros_like(updated).scatter(
        -1, updated.abs().topk(3, dim=-1).indices, 1
    )
    torch.testing.assert_close(actual, expected)
    assert (actual != 0).sum(-1).max().item() <= 3


def test_patch_layer_selection():
    class Model:
        memories = [_memory(), _memory()]

    model = Model()
    patch_link_matrix(model, "ablated", layer=1)
    assert not isinstance(model.memories[0], AblatableSparseLinkMemory)
    assert isinstance(model.memories[1], AblatableSparseLinkMemory)
