"""
chunked_parallel_dnc.py -- v1 (new file)

Alternate Phase 3, Step 2 (see Experiment-Roadmap.md, "Attempt to modify DNC
in a way that it can in one way or another simulate SSMs parallelism ...
The 3 options from that section should be analyzed ... The faithfulness to
DNC is not an issue, we need to keep the benefits of DNC and its
functionality while adding even better functionality to it.").

--------------------------------------------------------------------------
Analysis of the roadmap's three options (required before implementing)
--------------------------------------------------------------------------
Recap of the roadmap's own framing: DNC's memory update
    M_t = M_{t-1} ⊙ (1 − w_t e_t^T) + w_t v_t^T
is a rank-1 affine recurrence on a matrix state, structurally identical to
an SSM transition x_t = A_t x_{t-1} + B_t u_t. It stays *sequential* (not
scannable like Mamba) because DNC's write weighting w_t is itself a
function of M_{t-1} (content-based cosine-similarity addressing against
the *current* memory) plus a usage/allocation vector and a temporal link
matrix that both evolve from memory's own history -- i.e. the recurrence's
own "A_t, B_t" depend on the running state, not just the external input
stream, which breaks the associativity trick that makes Mamba's scan work.

  Option 1 (input-only addressing, exact scan). Rejected. Forcing w_t to be
  a function of the external input/controller state ALONE -- never of
  M_{t-1} -- removes exactly the mechanism (content-based lookup of the
  *current* memory contents) that distinguishes DNC from a plain SSM in
  the first place. The roadmap's own text is blunt about this ("you get a
  fast, parallel, DNC-flavored architecture, not DNC"), and this project's
  explicit instruction is to KEEP DNC's benefits/functionality while adding
  parallelism, not trade one for the other. Also, concretely: the KL-prior
  write head (stochastic_write_head_v2.py) already depends on `Memory`
  computing write weightings the normal (content+allocation) way -- an
  input-only-addressing DNC would still be compatible with the write head
  mechanically, but would forfeit the associative-recall behavior that made
  the London Underground OOD generalization test meaningful in the first
  place (Graves et al. 2016's own headline result is precisely that
  content-based *and* temporal-link addressing generalizes to unseen
  graphs -- an input-only-addressing variant has no graph-traversal claim
  left to test). Rejected on "faithfulness to DNC's functionality" grounds,
  which the roadmap explicitly says matters more than raw speed here.

  Option 3 (DEER/ELK: full nonlinear-recurrence Newton-Raphson parallel
  solve). Considered and rejected for THIS phase, not dismissed in
  principle. DEER (Lim et al., NeurIPS 2023) is real, general, and would in
  principle cover DNC's full memory+controller feedback loop, addressing
  included, with no approximation error at convergence (unlike Option 2).
  But it requires: (a) linearizing the ENTIRE per-step DNC transition
  (controller forward pass + content-addressing softmax + allocation
  sort/cumprod + link-matrix update) at every Newton iteration, i.e. a
  full Jacobian-vector-product capability through all of that, which this
  codebase has no existing scaffolding for and which the roadmap's own
  caveat flags as numerically risky specifically where this project's
  addressing lives (the cosine-similarity content softmax has the same
  flat/low-gradient-region character the roadmap warns will make Newton's
  method converge slowly or unstably); and (b) the roadmap's own explicit
  "honest caveat" that at this project's actual episode lengths (~8-260
  steps, not "thousands+"), a well-optimized *sequential* per-step SSM
  controller would likely close most of the LSTM-vs-SNN speed gap on its
  own -- i.e. DEER's O(log T) sequential-Newton-rounds payoff is aimed at a
  regime (T in the thousands+) this project is not currently operating in.
  Given that, DEER is real future work (documented as such in the
  standalone architecture writeup, Section 6) but a poor fit for what
  "Step 2, implemented now, at this project's actual scale" calls for.

  Option 2 (chunked/blockwise approximate parallelism). SELECTED. This is
  the option the roadmap itself calls "the practical middle ground ...
  real, proven ... you could genuinely build here", citing exactly the
  pattern this file implements: freeze the memory state (here: just the
  read vector, the minimal thing needed to decouple the controller from
  the memory's cross-timestep dependency) for a short chunk, run the
  controller in parallel within the chunk under that frozen context, then
  reconcile memory with one cheap, EXACT, fully-sequential pass per chunk.
  It trades a tunable, chunk-size-controlled approximation error (stale
  reads within a chunk) for real, measurable wall-clock parallelism, using
  exactly the same pattern chunked linear attention / GLA / Mamba-2's own
  hardware-efficient scan already use in production (the roadmap's own
  citation). It is the only one of the three options that (a) keeps DNC's
  actual content/allocation/link-matrix addressing running, unmodified,
  every single real timestep (not an approximation of addressing itself --
  see "What stays exact" below), (b) requires no new numerical-optimization
  machinery this codebase doesn't already have, and (c) degrades gracefully
  to the *exact* original sequential DNC at chunk_size=1 (see "Exactness at
  chunk_size=1" below) -- an explicit, checkable regression test, not just
  an assertion.

An additional option, not named in the roadmap, was considered and also
rejected: (4) truncated-BPTT chunking (detach the controller state AND the
read vector at every chunk boundary, training each chunk as an
independent short sequence). This would be simpler to implement than what
follows, but was rejected because it would silently change what the model
is being trained to do -- long-range credit assignment through the KL-prior
mechanism, the temporal link matrix, and the controller's own recurrent
memory would all be truncated at the chunk boundary, which is a strictly
different (and, for this project's graph-traversal task, likely much
weaker) learning signal than what Phase 1/2 validated, not merely a faster
way to compute the same gradient. This file does NOT detach anything at
chunk boundaries -- see "Full BPTT is preserved" below.

--------------------------------------------------------------------------
What this file actually does
--------------------------------------------------------------------------
`ChunkedParallelDNC` subclasses `MambaDNC` (mamba_controller.py, v7)
UNMODIFIED -- `__init__` is inherited verbatim for BOTH controller types
(`rnn_type='lstm'` and `rnn_type='mamba'`), so Memory construction, the
output projection, and every existing attribute are built exactly as
Phase 2 / Alternate-Phase-3-Step-1 already build them. The only thing this
subclass does after `__init__` runs is, for `rnn_type == 'mamba'` only,
transplant each `MambaControllerWrapper`'s trained parameters into a
`MambaChunkControllerWrapper` (mamba_chunk_controller.py) via a literal
`state_dict()` copy -- the two classes have identical submodule trees (the
chunk-capable cell only ADDS a method, `forward_chunk`, never a new
parameter -- see mamba_chunk_controller.py's module docstring), so this
transplant is guaranteed to reproduce the exact same trained weights, not
an independently-initialized approximation of them.

The only new logic is `forward()`, which processes the sequence in chunks
of `self.chunk_size` real timesteps:

  for each chunk of length C (<= chunk_size):
    1. Freeze the controller's read-vector INPUT for every one of this
       chunk's C real timesteps to `last_read` -- the TRUE read vector
       computed at the end of the PREVIOUS chunk (or zeros, at sequence
       start). This is the ONE approximation this file introduces (see
       "What is approximated" below).
    2. Run the controller ONCE over the whole (B, C, nn_input_size) chunk:
         - `rnn_type == 'mamba'`: `MambaChunkControllerWrapper.forward_chunk`
           (mamba_chunk_controller.py) -- a real O(log C)-depth parallel
           scan, not a Python loop.
         - `rnn_type in {'lstm','gru','rnn'}`: the stock `nn.LSTM`/`nn.GRU`/
           `nn.RNN` instance's own native (B, C, *) forward call -- these
           already accept a whole sequence in one call (that is their
           default mode; DNC's own `_layer_forward` is the thing that
           artificially restricts them to one step at a time via
           `input.unsqueeze(1)`), so no new class is needed for this path
           at all, and it gets the exact same cuDNN-fused-kernel speedup
           this file's introduction claims for the whole design.
       Either way, this produces C interface vectors (ξ_1 .. ξ_C) for the
       chunk IN ONE CALL.
    3. Reconcile memory EXACTLY and SEQUENTIALLY, one real timestep at a
       time, C cheap steps: call `self.memories[0].forward(ξ_i, mhx)`
       (`dnc.memory.Memory`, completely unmodified -- content lookup,
       allocation, the link matrix, read modes, ALL of it, run exactly as
       the stock library does every single time) for i = 1..C in order.
       This is real, unapproximated DNC addressing for every real
       timestep -- what's approximated is only what the CONTROLLER saw as
       "the previous read vector" while producing ξ_1..ξ_C, not what
       memory itself computes from those ξ's.
    4. Set `last_read` to the TRUE final read vector this chunk's
       reconciliation pass actually produced (i.e. the read after the
       LAST real timestep in the chunk) -- this becomes the frozen context
       for the NEXT chunk, so staleness never compounds beyond one chunk's
       width; every chunk starts from a fully genuine, freshly-reconciled
       read vector.

`chunk_size=1` makes every "chunk" exactly one real timestep, at which
point step 1's freeze introduces NO staleness at all (the frozen value IS
the true previous-timestep read vector, by construction, since there is
no gap to be stale over) -- see "Exactness at chunk_size=1" below for the
full argument. Every increase in `chunk_size` beyond 1 trades a larger
approximation window for fewer, larger (more parallel) controller calls.

--------------------------------------------------------------------------
What stays exact regardless of chunk_size
--------------------------------------------------------------------------
  - Memory addressing itself (content-based read/write key lookup,
    allocation-gate/usage-vector bookkeeping, the temporal link matrix,
    read-mode interpolation): `dnc.memory.Memory.write()`/`.read()`/
    `.forward()` are called completely unmodified, once per REAL timestep,
    never batched, never approximated, never skipped. This file does not
    contain a single line that reimplements or alters that math.
  - The stochastic write head / KL-prior mechanism
    (`stochastic_write_head_v2.py`, Phase 1/2, completely unmodified,
    imported and installed exactly as `Alter_PHASE3_mamba.py` already
    does): because step 3 above calls `Memory.forward(ξ_i, mhx)` exactly
    once per real timestep -- the same call `Memory.write_vector_transform`
    (the installed `StochasticWriteHead`) is invoked from, the same number
    of times, with the same per-timestep-only inputs it always received --
    the KL accumulation (`self._kl_terms`), the periodic-snapshot prior
    fitting (`self._recent_writes`), and the `pop_total_kl()`/
    `update_all_prior_snapshots()` training-script call sites all keep
    working with ZERO code change and zero behavior change. The write
    head has no way to observe or depend on whether its caller arrived via
    a chunked or fully-sequential forward pass -- it only ever sees "one
    timestep's `x`, called once", which is exactly what it still gets.
  - Full BPTT is preserved: nothing in this file calls `.detach()` on the
    controller state OR on `last_read` at a chunk boundary. Gradients flow
    backward through the full, true sequence exactly as they did in the
    unmodified sequential DNC forward pass -- the only thing that changes
    is which VALUE (the true previous-chunk-final read vs. a step-by-step-
    updated read) a given controller call's input depended on forward, not
    whether gradient can reach earlier timesteps through it.

--------------------------------------------------------------------------
What IS approximated
--------------------------------------------------------------------------
Exactly one thing: for chunk_size > 1, the controller at real timestep t
(for t inside a chunk but not the chunk's first step) sees the read vector
from the START of that chunk, not from t-1. Concretely, for a chunk
covering real timesteps [t0, t0+C), the controller's read-vector input at
every one of those C steps is `read_vector_after(t0-1)`, not
`read_vector_after(t-1)`. This is precisely the "stale reads within a
chunk" tradeoff the roadmap names explicitly for Option 2, tunable via
`chunk_size` (C=1 removes it entirely; larger C trades more staleness for
fewer, more-parallel controller calls). Everything else (memory contents,
addressing, the KL/prior mechanism, the controller's own recurrent state)
evolves exactly, every real timestep, with no approximation.

--------------------------------------------------------------------------
Exactness at chunk_size=1 (the mandatory regression check)
--------------------------------------------------------------------------
At `chunk_size=1`, every chunk has C=1, so:
  - step 1's "frozen" read vector for that single real timestep IS
    `last_read` from the immediately-preceding real timestep -- there is
    no earlier timestep within the (length-1) chunk it could have staled
    relative to, so this is not an approximation, it is simply the correct
    value.
  - step 2's controller call processes exactly one real timestep. For
    `rnn_type == 'mamba'`, `MambaChunkControllerWrapper.forward_chunk` at
    C=1 is provably identical to `mamba_controller.MambaControllerCell.
    step()` (see mamba_chunk_controller.py's module docstring, "Exactness
    of forward_chunk() vs. step() x C", point-by-point). For
    `rnn_type in {'lstm','gru','rnn'}`, calling the stock recurrent module
    with a length-1 sequence is definitionally identical to calling it
    with `input.unsqueeze(1)` the way `dnc.dnc.DNC._layer_forward` already
    does -- there is no algorithmic difference, only whether the Python
    call site wrote `.unsqueeze(1)` explicitly or the chunk slice already
    has a length-1 middle dimension.
  - step 3 reconciles memory for exactly that one real timestep, calling
    the same `Memory.forward()` the stock sequential loop would have
    called at that timestep, with the same `ξ` (since step 2 produced the
    same `ξ` as the sequential path would have -- see above) and the same
    `mhx` (memory hidden state), carried forward identically chunk-to-
    chunk / step-to-step either way.

Therefore `ChunkedParallelDNC(..., chunk_size=1).forward(...)` computes the
SAME function as running the model's ordinary, fully-sequential forward
pass one real timestep at a time -- not an approximation of it, an
algebraically equivalent re-expression of it (up to ordinary floating-
point operation-order non-associativity, the same caveat that already
applies to e.g. `nn.LSTM`'s fused cuDNN kernel vs. a hand-written
per-step LSTM cell, which is not specific to anything in this file). This
is the load-bearing regression test for "did wiring in the chunked
codepath at chunk_size=1 change any existing Phase 1/2/Step-1 behavior" --
it should not, by the argument above, and this should be verified once,
directly, in the real training environment (GPU + `mamba-ssm` installed)
before trusting any chunk_size>1 result, exactly the way every prior
version bump in this project (v2 through v7) was validated against its
immediate predecessor before being trusted. See this repository's
`test_chunked_parallel_dnc.py` for the runnable version of this check
(skips gracefully, rather than silently passing, when torch/mamba-ssm
are not available in the environment it's run in).

--------------------------------------------------------------------------
Scope restriction (explicit, not silent)
--------------------------------------------------------------------------
This class supports exactly the configuration `Alter_PHASE3_mamba.py`
actually constructs: `num_layers=1`, `share_memory_between_layers=True`,
`batch_first=True`, dense `torch.Tensor` input (not `PackedSequence`),
`debug=False`. `__init__` does not need to check these (they're inherited
from `MambaDNC`/`DNC` as configured by the caller), but `forward()`
explicitly asserts them and raises `NotImplementedError` with a clear
message otherwise, rather than silently computing something wrong for a
configuration this file was never designed or reasoned about for.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from model.mamba_controller import MambaDNC
from Chunked_parallel.mamba_chunk_controller import MambaChunkControllerWrapper


class ChunkedParallelDNC(MambaDNC):
    """`MambaDNC` subclass adding a chunked-parallel `forward()`. See
    module docstring for the full design rationale, the three-option
    analysis, and the exactness argument for `chunk_size=1`.

    Args (beyond `MambaDNC`'s own, all passed straight through unchanged):
        chunk_size: number of REAL timesteps processed per controller call.
            `chunk_size=1` reproduces the exact sequential DNC forward pass
            (see module docstring). Must be a positive int. Default 1, so
            constructing this class with no extra argument is a no-op
            relative to `MambaDNC` (consistent with this project's existing
            convention of every new knob defaulting to "reproduce the
            previous behavior exactly" -- see `Alter_PHASE3_mamba.py`'s
            v3-v7 header notes for the same pattern applied to
            `CONTROLLER_TYPE`, `PRIOR_SNAPSHOT_EVERY`, etc.).
    """

    def __init__(self, *args, chunk_size: int = 1, **kwargs):
        if not isinstance(chunk_size, int) or chunk_size < 1:
            raise ValueError(f"chunk_size must be a positive int, got {chunk_size!r}")

        # Build everything exactly as MambaDNC/DNC already do, for either
        # controller type -- Memory construction, the output projection,
        # self.rnns[layer] as a MambaControllerWrapper (mamba) or
        # nn.LSTM/GRU/RNN (otherwise). Nothing about this call is
        # different from what Alter_PHASE3_mamba.py's existing
        # `MambaDNC(...)` construction call already does.
        super().__init__(*args, **kwargs)
        self.chunk_size = chunk_size

        if self.rnn_type.lower() == "mamba":
            # Upgrade each MambaControllerWrapper -> MambaChunkControllerWrapper
            # via a literal state_dict() transplant, NOT independent
            # reconstruction -- see module docstring for why this
            # guarantees parameter identity rather than merely aiming for
            # it. Both classes have identical submodule trees
            # (`in_adapter`, `blocks.<i>.norm`, `blocks.<i>.cell.mamba.*`)
            # because MambaChunkControllerCell subclasses
            # MambaControllerCell and adds only a method, never a
            # parameter -- see mamba_chunk_controller.py.
            for layer in range(self.num_layers):
                old_wrapper = self.rnns[layer]
                new_wrapper = MambaChunkControllerWrapper(
                    in_dim=(self.nn_input_size if layer == 0 else self.nn_output_size),
                    d_model=self.output_size,
                    num_blocks=self.num_hidden_layers,
                    d_state=self.mamba_d_state,
                    d_conv=self.mamba_d_conv,
                    expand=self.mamba_expand,
                    device=self.device,
                )
                new_wrapper.load_state_dict(old_wrapper.state_dict())
                if self.device is not None and self.device.type == "cuda":
                    new_wrapper = new_wrapper.to(self.device)
                self.rnns[layer] = new_wrapper
                # setattr, matching MambaDNC.__init__'s own convention --
                # this is what actually re-registers the submodule for
                # autograd/optimizer.parameters()/`.to(device)`; the
                # `self.rnns` list itself is a plain Python list, not an
                # nn.ModuleList (see MambaDNC.__init__ / dnc.dnc.DNC.__init__).
                setattr(self, "mamba_layer_" + str(layer), new_wrapper)

    def forward(
        self,
        input_data: torch.Tensor,
        hx,
        reset_experience: bool = False,
        pass_through_memory: bool = True,
    ):
        """Chunked-parallel forward pass. See module docstring for the
        full step-by-step description and the exactness argument for
        `chunk_size=1`.

        Args / Returns: identical contract to `dnc.dnc.DNC.forward()` /
        `mamba_controller.MambaDNC` (inherited, not overridden) for the
        scope this class supports (see "Scope restriction" above) --
        `input_data`: (B, T, input_size) since this class requires
        `batch_first=True` (checked below). Returns `(output, hidden)`
        with `output` TIME-MAJOR, `(T, B, input_size)` -- matching this
        exact library version's own `forward()` convention as already
        consumed by `Alter_PHASE3_mamba.py`'s training loop (which calls
        `output.transpose(0, 1)` itself immediately after every `rnn(...)`
        call, for both the stock sequential path and this one).
        """
        if self.num_layers != 1 or not self.share_memory_between_layers:
            raise NotImplementedError(
                "ChunkedParallelDNC.forward only supports num_layers=1, "
                "share_memory_between_layers=True (the only configuration "
                "Alter_PHASE3_mamba.py actually constructs) -- got "
                f"num_layers={self.num_layers}, "
                f"share_memory_between_layers={self.share_memory_between_layers}. "
                "See this file's module docstring, 'Scope restriction'."
            )
        if not self.batch_first:
            raise NotImplementedError(
                "ChunkedParallelDNC.forward only supports batch_first=True. "
                "See this file's module docstring, 'Scope restriction'."
            )
        if self.debug:
            raise NotImplementedError(
                "ChunkedParallelDNC.forward does not implement the debug/viz "
                "path -- construct with debug=False. See this file's module "
                "docstring, 'Scope restriction'."
            )
        if not torch.is_tensor(input_data):
            raise NotImplementedError(
                "ChunkedParallelDNC.forward only supports a dense torch.Tensor "
                "input_data (not PackedSequence). See this file's module "
                "docstring, 'Scope restriction'."
            )

        input = input_data
        batch_size = input.size(0)
        max_length = input.size(1)

        controller_hidden, mem_hidden, last_read = self._init_hidden(hx, batch_size, reset_experience)
        chx = controller_hidden[0]
        mhx = mem_hidden[0]
        memory = self.memories[0]

        is_mamba = self.rnn_type.lower() == "mamba"
        rw = self.w * self.r  # read-vector width, matches dnc.DNC's own `self.w * self.r`

        step_outputs: list[torch.Tensor] = []
        t = 0
        while t < max_length:
            chunk_len = min(self.chunk_size, max_length - t)
            x_chunk = input[:, t : t + chunk_len, :]  # (B, chunk_len, input_size)

            # ---- step 1: freeze the read-vector INPUT for this whole chunk ----
            # `last_read` here is the TRUE read vector at the end of the
            # previous chunk (or the true zero/initial read at sequence
            # start) -- see module docstring, "What IS approximated". Not
            # detached: gradient flows back through this exactly like any
            # other tensor in the graph (see "Full BPTT is preserved").
            frozen_read = last_read.unsqueeze(1).expand(-1, chunk_len, -1)  # (B, chunk_len, rw)
            controller_in_chunk = torch.cat([x_chunk, frozen_read], dim=-1)  # (B, chunk_len, nn_input_size)

            # ---- step 2: ONE controller call for the whole chunk --------------
            if is_mamba:
                ctrl_out_chunk, chx = self.rnns[0].forward_chunk(controller_in_chunk, chx)
            else:
                ctrl_out_chunk, chx = self.rnns[0](controller_in_chunk, chx)
            # ctrl_out_chunk: (B, chunk_len, output_size)

            if self.clip != 0:
                ctrl_out_chunk = torch.clamp(ctrl_out_chunk, -self.clip, self.clip)

            # ---- step 3: EXACT, sequential, per-real-timestep memory reconciliation ----
            # dnc.memory.Memory.forward() is called completely unmodified,
            # once per real timestep -- see module docstring, "What stays
            # exact regardless of chunk_size". This is a Python loop of
            # length chunk_len, but each iteration is cheap (Memory ops
            # only -- no controller recompute), unlike the fully-sequential
            # baseline's loop, where EVERY iteration also re-runs the full
            # controller forward pass.
            for i in range(chunk_len):
                xi = ctrl_out_chunk[:, i, :]  # (B, output_size) -- this real timestep's interface vector
                if pass_through_memory:
                    read_vecs, mhx = memory(xi, mhx)
                    read_vectors = read_vecs.reshape(batch_size, rw)
                else:
                    read_vectors = xi.new_zeros(batch_size, rw)
                step_outputs.append(torch.cat([xi, read_vectors], dim=1))
                last_read = read_vectors  # step 4: true final read carried forward

            t += chunk_len

        controller_hidden[0] = chx
        mem_hidden[0] = mhx

        outputs_stacked = torch.stack(step_outputs, dim=0)  # (T, B, nn_output_size) -- time-major
        outputs_final = self.output(outputs_stacked)  # (T, B, input_size)

        return outputs_final, (controller_hidden, mem_hidden, last_read)
