"""
Static Option 2: rebuild a dnc.DNC's Memory submodule at a new nr_cells,
transplanting every N-independent learned sublayer so no trained weight is
lost. Safe because Memory's per-episode state (memory/link_matrix/
precedence/usage_vector/read_weights/write_weights) is rebuilt from scratch
by Memory.reset() every forward call regardless -- there is no cross-episode
memory *content* to preserve, only trained *weights*, and none of those are
sized by nr_cells.

Also transplants Option 1's link-matrix patch (class + mode) onto the
rebuilt Memory when the old one was patched -- see the isinstance check
below and link_matrix_ablation.py's module docstring for why this is
needed: without it, a resize event mid-run would silently revert an active
link-matrix ablation/sparsification back to dense behavior.
"""
import torch.nn as nn
from dnc.memory import Memory
from link_matrix_ablation import AblatableSparseLinkMemory


def resize_memory(model, new_nr_cells: int, device=None, layer: int = 0):
    old_memory: Memory = model.memories[layer]          # VERIFY attribute name
    if old_memory.nr_cells == new_nr_cells:              # VERIFY kwarg/attr name
        return old_memory

    new_memory = Memory(
        input_size=old_memory.input_size,
        nr_cells=new_nr_cells,                           # VERIFY kwarg name
        cell_size=old_memory.cell_size,
        read_heads=old_memory.read_heads,
        independent_linears=True,
    ).to(device)

    # Transplant every N-independent learned sublayer except
    # write_vector_transform (that one is a StochasticWriteHead, not a
    # plain Linear, on old_memory -- move the module object itself instead
    # of trying to load_state_dict it).
    old_write_head = old_memory.write_vector_transform
    old_memory.write_vector_transform = nn.Identity()   # placeholder so it's excluded below
    new_memory.load_state_dict(old_memory.state_dict(), strict=False)
    new_memory.write_vector_transform = old_write_head    # move the actual module

    # Static Option 1 interaction: if old_memory was patched by
    # link_matrix_ablation.patch_link_matrix (its __class__ reassigned to
    # AblatableSparseLinkMemory, carrying link_matrix_mode / optionally
    # link_matrix_topk as plain instance attributes), new_memory above is a
    # *different* freshly-constructed plain dnc.memory.Memory object --
    # load_state_dict only copies tensor weights, never the Python class or
    # plain instance attributes, so the patch would silently be dropped
    # right here, reverting new_memory to baseline dense link-matrix
    # behavior with no error or warning. Re-apply it explicitly so a resize
    # event (Option 2) can never silently undo an active link-matrix
    # ablation/sparsification (Option 1) mid-run.
    if isinstance(old_memory, AblatableSparseLinkMemory):
        new_memory.__class__ = AblatableSparseLinkMemory
        new_memory.link_matrix_mode = old_memory.link_matrix_mode
        if hasattr(old_memory, "link_matrix_topk"):
            new_memory.link_matrix_topk = old_memory.link_matrix_topk

    model.memories[layer] = new_memory
    return new_memory