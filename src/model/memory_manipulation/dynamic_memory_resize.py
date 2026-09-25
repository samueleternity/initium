"""
file: dynamic_memory_resize.py

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

--- optimizer resync (added for Dynamic-N) --------------------------------
`model.memories[layer] = new_memory` swaps the module tree, but the
training-loop optimizer (`torch.optim.Adam(list(rnn.parameters()) + ...)`)
was built ONCE, before this function was ever called, from the Parameter
OBJECTS live at that time. load_state_dict() above only copies tensor
*values* into new_memory's freshly-constructed Parameters -- it never makes
optimizer.param_groups aware of them. So without the `optimizer=` arg
below, every Memory sublayer except write_vector_transform (moved over as
the same object, so already tracked) silently stops receiving Adam updates
for the rest of the run after ANY live mid-run call to this function:
gradients still compute (they're in the forward graph), but
optimizer.step() only updates params actually present in a param_group,
and the newly-constructed ones never are.

This was already true for static Option 2's curriculum-indexed resize
(TraversalCurriculum.maybe_advance() calls this function without
`optimizer=`, ~13 times over a 120k-step run) -- it just wasn't obvious in
aggregate metrics from a handful of one-time freezes near a lesson
boundary. It's NOT a problem on a checkpoint *resume*, because
`optimizer.load_state_dict(ckpt["optimizer_state_dict"])` is called right
after resize_memory() there and rebuilds `state` keyed to the CURRENT
param objects, matched positionally against the checkpoint's saved
param_groups (same architecture => same order => this just works). It's
specifically live, same-process, mid-run resize calls that orphan params --
which is exactly what Dynamic-N's whole premise is: resizing far more
often, live, mid-run. So this fix ships as part of Dynamic-N; the existing
static-Option-2 call site can opt in by passing `optimizer=optimizer` too
(not changed automatically here, since that changes reproducibility of
already-running static-Option-2 experiments).
"""

import torch.nn as nn
from dnc.memory import Memory
from memory_manipulation.link_matrix_ablation import AblatableSparseLinkMemory


def _resync_optimizer_after_resize(optimizer, old_named_params: dict, new_memory) -> None:
    """Repoint optimizer.param_groups at new_memory's Parameter objects
    wherever a same-named old parameter existed, and migrate that
    parameter's Adam state (exp_avg, exp_avg_sq, step) onto the new object
    so momentum isn't reset by a resize. Safe to do unconditionally here
    because no Memory Linear sublayer's shape depends on nr_cells (see
    module docstring) -- every migrated tensor is the same shape before and
    after.

    write_vector_transform is a no-op under this function: `new_p is
    old_p` for its parameters (the module object itself was moved, not
    rebuilt -- see resize_memory() below), so it's skipped by the `is not`
    check and stays exactly as-is in both param_groups and optimizer.state.
    """
    new_named_params = dict(new_memory.named_parameters())
    replace_map = {}  # old Parameter object -> new Parameter object
    for name, old_p in old_named_params.items():
        new_p = new_named_params.get(name)
        if new_p is not None and new_p is not old_p:
            replace_map[old_p] = new_p

    for old_p, new_p in replace_map.items():
        if old_p in optimizer.state:
            optimizer.state[new_p] = optimizer.state.pop(old_p)

    replace_ids = {id(old_p): new_p for old_p, new_p in replace_map.items()}
    for group in optimizer.param_groups:
        group["params"] = [replace_ids.get(id(p), p) for p in group["params"]]


def resize_memory(model, new_nr_cells: int, device=None, layer: int = 0, optimizer=None):
    old_memory: Memory = model.memories[layer]  # VERIFY attribute name
    if old_memory.nr_cells == new_nr_cells:  # VERIFY kwarg/attr name
        return old_memory

    # Captured before write_vector_transform is swapped to a placeholder
    # below, so this reflects old_memory's real (pre-mutation) structure --
    # used by the optimizer resync at the very end of this function.
    old_named_params = dict(old_memory.named_parameters())

    new_memory = Memory(
        input_size=old_memory.input_size,
        nr_cells=new_nr_cells,  # VERIFY kwarg name
        cell_size=old_memory.cell_size,
        read_heads=old_memory.read_heads,
        independent_linears=True,
        device=device,  # VERIFY kwarg name -- see note below
    ).to(device)

    # Transplant every N-independent learned sublayer except
    # write_vector_transform (that one is a StochasticWriteHead, not a
    # plain Linear, on old_memory -- move the module object itself instead
    # of trying to load_state_dict it).
    old_write_head = old_memory.write_vector_transform
    old_memory.write_vector_transform = nn.Identity()  # placeholder so it's excluded below
    new_memory.load_state_dict(old_memory.state_dict(), strict=False)
    new_memory.write_vector_transform = old_write_head  # move the actual module

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
    # model.memories is a plain python list, not an nn.ModuleList -- the
    # line above rebinds the list slot but never touches whatever attribute
    # PyTorch's own submodule registry (_modules) actually points at
    # (rnn_layer_memory_shared, or rnn_layer_memory_<layer> when memories
    # aren't shared -- set once via setattr() at model construction, see
    # mamba_controller.py / dnc.DNC.__init__). Without this, that attribute
    # stays pointed at old_memory forever -- the same object whose
    # write_vector_transform was just swapped to nn.Identity() above as a
    # transplant placeholder -- so rnn.state_dict()/rnn.parameters()/
    # rnn.load_state_dict() silently keep walking a stale, Identity-headed
    # module while forward() (which reads self.memories[layer] directly)
    # correctly uses new_memory. Found via a resume-time crash: dynamic-N's
    # floor-start construction forces a resize on resume that a static-only
    # workflow never used to trigger, which is what surfaced this -- but
    # the same corruption happens on any LIVE resize, including every
    # static-Option-2 lesson advance that ever fired.
    for attr_name, submodule in list(model._modules.items()):
        if submodule is old_memory:
            setattr(model, attr_name, new_memory)

    if optimizer is not None:
        _resync_optimizer_after_resize(optimizer, old_named_params, new_memory)

    return new_memory
