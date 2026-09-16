"""
file: link_matrix_ablation.py

Static Option 1: ablate or sparsify the DNC's temporal link matrix at fixed
N=256, per Session 009's synthesis finding that temporal-linkage addressing
is near-unused in QA-style (single-hop, content-addressed) tasks. Graph
traversal chains multiple hops per episode, which is exactly the case that
finding doesn't directly cover -- see the hop-breakdown logging in
Alter_PHASE3_mamba.py (TraversalCurriculum.maybe_advance, plus the periodic
and terminal OOD evaluate_traversal calls) for the diagnostic that watches
for a hop-count-correlated accuracy collapse this synthesis finding
wouldn't predict.

This does NOT touch nr_cells (that's Option 2 / dynamic_memory_resize.py)
and does NOT resize any tensor -- it only changes what get_link_matrix()
computes at whatever N the Memory was already built at. Applied by
reassigning __class__ on an already-constructed dnc.memory.Memory instance
(see patch_link_matrix() below), not by editing the installed dnc package --
same external-library-as-black-box precedent as dynamic_memory_resize.py's
weight-transplant approach, just simpler here since no weights move.

INTERACTION WITH OPTION 2: dynamic_memory_resize.resize_memory() builds a
brand-new plain dnc.memory.Memory instance and swaps it into
model.memories[layer]. Its own weight-transplant (load_state_dict) has no
notion of link-matrix mode, so without extra handling a resize event would
silently drop back to "dense" mid-run. resize_memory() now checks
isinstance(old_memory, AblatableSparseLinkMemory) and re-applies the class
and link_matrix_mode/link_matrix_topk onto the freshly-built new_memory --
see that file. This module doesn't need to do anything for that to work;
it's noted here so the coupling isn't a surprise when reading either file
in isolation.
"""
import torch
from dnc.memory import Memory


class AblatableSparseLinkMemory(Memory):
    """Drop-in replacement for get_link_matrix() only. Every other Memory
    method (write, read, content_weightings, directional_weightings, ...)
    is inherited unchanged, and this class adds no new nn.Module state --
    so it's safe to attach to an already-constructed Memory instance via
    __class__ reassignment (see patch_link_matrix) instead of
    reconstructing/reloading weights.

    Mode is read from instance attributes (set by patch_link_matrix, not
    passed as get_link_matrix() args), so this overrides cleanly against
    the existing call signature in Memory.write() -- no caller changes
    needed anywhere else in dnc.py or memory.py.

    link_matrix_mode:
      "dense"   -- unchanged Memory behavior (baseline; same as not
                   patching at all).
      "ablated" -- link matrix is never updated; it stays at its initial
                   zero value (from Memory.new()/reset()) for the whole
                   episode. directional_weightings() (unchanged, inherited)
                   then always returns zero forward/backward weightings, so
                   read_weightings' content_mode term is the only nonzero
                   contribution -- content-based addressing continues to
                   work exactly as before, temporal addressing contributes
                   nothing. This is the "ablate" half of the roadmap's
                   "ablate or aggressively sparsify."
      "sparse_topk" -- link matrix is updated as normal every write() call,
                   but immediately after each update, every row keeps only
                   its link_matrix_topk largest-magnitude entries and zeros
                   the rest. This is the "sparsify" half -- reduces the
                   O(N^2) footprint the roadmap's citation flags, while
                   still letting the model use *some* temporal signal,
                   unlike full ablation. Recomputed fresh every step (not a
                   one-time structural prune), so it tracks whichever links
                   the model is actually relying on at that point in the
                   episode rather than freezing an initial guess.
    """

    def get_link_matrix(
        self, link_matrix: torch.Tensor, write_weights: torch.Tensor, precedence: torch.Tensor
    ) -> torch.Tensor:
        mode = getattr(self, "link_matrix_mode", "dense")

        if mode == "dense":
            return super().get_link_matrix(link_matrix, write_weights, precedence)

        if mode == "ablated":
            return link_matrix  # never updated -- stays at reset()'s zero value

        if mode == "sparse_topk":
            updated = super().get_link_matrix(link_matrix, write_weights, precedence)
            k = getattr(self, "link_matrix_topk", None)
            if k is None or k >= self.nr_cells:
                return updated  # no-op if topk isn't set or wouldn't actually sparsify anything
            # updated: (batch, 1, N, N). Keep each row's top-k
            # largest-magnitude entries, zero the rest.
            magnitudes = updated.abs()
            _, topk_idx = magnitudes.topk(k, dim=-1)
            mask = torch.zeros_like(updated)
            mask.scatter_(-1, topk_idx, 1.0)
            return updated * mask

        raise ValueError(f"Unknown link_matrix_mode: {mode!r}")


def patch_link_matrix(model, mode: str, topk: int | None = None, layer: int | None = None):
    """Apply Static Option 1 to an already-constructed dnc.DNC (or MambaDNC,
    which subclasses DNC and builds model.memories identically -- see
    mamba_controller.py) instance, right after model construction.

    mode: "dense" | "ablated" | "sparse_topk" -- see AblatableSparseLinkMemory
        docstring. "dense" is accepted (as a no-op) so a run can be flipped
        back to baseline behavior via the same config/CLI path, for the
        roadmap's required baseline-vs-isolated-ablation comparison,
        without touching any other run() plumbing.
    topk: required (and only used) when mode == "sparse_topk".
    layer: which model.memories[i] to patch. None (default) patches every
        entry in model.memories -- correct for share_memory_between_layers
        =True (this project's setup: a single shared Memory, so memories
        has exactly one entry) and still correct if share_memory_between_
        layers is ever False (patches every per-layer Memory identically).
        Pass an int to patch only one layer's Memory when memories are NOT
        shared and you want just one layer's link matrix affected.
    """
    if mode == "sparse_topk" and topk is None:
        raise ValueError("patch_link_matrix: topk is required when mode='sparse_topk'")

    targets = model.memories if layer is None else [model.memories[layer]]
    for memory in targets:
        memory.__class__ = AblatableSparseLinkMemory
        memory.link_matrix_mode = mode
        if topk is not None:
            memory.link_matrix_topk = topk
    return model