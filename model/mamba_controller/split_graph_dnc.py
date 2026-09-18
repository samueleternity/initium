"""
file: split_graph_dnc.py

Alternate Phase 3, Step 2, Option 5 -- "A genuine, higher-payoff Idea
candidate: split the compute graph, don't try to parallelize the addressing
itself" (see Experiment-Roadmap.md, Idea 5 and its failure-mode table).

DISABLED BY DEFAULT AT EVERY LEVEL (module has no global toggle of its own
-- it's simply not constructed unless the training script's
SPLIT_GRAPH_ENABLED / --split-graph is explicitly set). The roadmap
explicitly flags this mechanism as having NO corpus-established failure
mode yet ("Honesty check first... there is no corpus-established failure
mode for this option yet") and requiring its own isolated ablation before
being trusted -- treat every run of this module as an open experiment, not
a validated component.

--- Design (mirrors the roadmap's own description of Idea 5) -------------
Two halves, matching "(a) the Mamba backbone's own token-level state
updates, which have no dependency on M_{t-1} and can run through the
standard parallel scan" and "(b) a genuinely sequential, but much smaller
and cheaper, memory read/write/addressing step layered on top, consuming
the backbone's parallel output as an input stream":

  (a) self.backbone (MambaBackboneParallel, see mamba_backbone_parallel.py)
      processes the RAW input sequence X -- never concatenated with a read
      vector -- in ONE call, using mamba_ssm's own whole-sequence
      forward(). This is what makes it parallel: it has no dependency on
      any memory state, by construction, so there is nothing forcing a
      Python loop.

  (b) A per-timestep loop that: (i) combines this timestep's backbone
      output h_t with the PREVIOUS timestep's read vector read_{t-1} via a
      small Linear ("combiner") to produce the interface vector xi_t fed
      into Memory, then (ii) calls the existing dnc.memory.Memory exactly
      the same way every other controller in this project already does.
      This is the "read now, decide next hop immediately" loop the
      roadmap's own diagnosis of DEER's and chunking's failures both
      center on -- it is preserved exactly, per-step, sequential, never
      frozen for a window. The ONLY thing removed from the sequential
      critical path is the backbone's own internal computation; Memory
      itself was already the sequential part in every prior architecture
      this project has built, so this doesn't add a new bottleneck, it
      just stops making the (much more expensive) backbone share it.

Requirement from the roadmap's own methodological note on this option: "do
not layer them all at once... Concept 16 / Suspected Pattern SP-10". This
module therefore deliberately does NOT wire in Option 4's MoE (moe_enabled
raises if set) -- Option 5 must be run and ablated on its own first.

--- Built-in ablation for the roadmap's own required verification --------
"Verify [the design] preserves the exact 'read now, decide next hop
immediately' dependency, not just an approximation of it... treat as an
open risk requiring its own isolated ablation before adoption." The
combiner can be disabled entirely (combine_reads=False), which makes
xi_t = h_t for every t -- i.e. addressing becomes a pure function of the
backbone output and NEVER sees any read vector at all. Comparing
combine_reads=True vs. False on the SAME architecture is the cheapest
possible direct test of whether that dependency actually matters here,
rather than assuming it does because the roadmap says so.

--- Functional-usage check (LB-9 / Concept 6) is inherited, not special-
cased here. This module implements the SAME `pass_through_memory` kwarg
contract every other controller in this project's forward() already
honors, so the training script's existing standing verification step
(evaluate_traversal(..., ablate_memory=True), wired into
Alter_PHASE3_mamba.py's run() at every EVAL_EVERY cadence, logged to
run_{run_id}_memory_dependency.csv) applies to this controller with ZERO
additional code -- see the "elif pass_through_memory is False" branch in
forward() below.

--- API-compatibility notes (why this is written the way it is) ----------
This does NOT subclass dnc.DNC or MambaDNC -- the whole point of Option 5
is that the forward pass's control flow is fundamentally different (one
parallel backbone call, THEN a sequential loop, instead of one interleaved
step per timestep), so there's no meaningful DNC.forward() to defer to.
Instead this file reproduces, by hand, every piece of the surrounding
project's expected surface area that other files depend on:
  - self.memories: a plain python list (NOT nn.ModuleList) holding one
    dnc.memory.Memory, matching MambaDNC's exact convention, because
    dynamic_memory_resize.resize_memory() does `model.memories[layer] =
    new_memory` and link_matrix_ablation.patch_link_matrix() iterates
    `model.memories` -- both assume this exact shape regardless of
    controller class.
  - setattr(self, "rnn_layer_memory_shared", self.memories[0]): the ONLY
    reason the Memory instance is also a genuine, autograd-registered
    submodule (needed for install_stochastic_write_heads()'s
    named_modules() walk in stochastic_write_head_v2.py, and for
    resize_memory()'s "find the attribute name pointing at old_memory"
    swap in dynamic_memory_resize.py). Same convention MambaDNC uses.
  - forward(input, hx=(chx, mhx, last_read), reset_experience, 
    pass_through_memory) -> (output, (chx, mhx, last_read)), with output
    shaped (T, B, input_size) -- matching the (T, B, C) convention the
    rest of this project's training/eval code already expects from
    rnn(...)'s output (see the .transpose(0,1) calls immediately after
    every rnn(...) call in Alter_PHASE3_mamba.py), and mhx wrapped as a
    single-entry list (`mhx = [mem_state]`) so Dynamic-N's existing
    `mem_hidden[0] if isinstance(mem_hidden, list) else mem_hidden` read
    of hidden[1]["usage_vector"] keeps working unmodified.

ASSUMPTION FLAGGED FOR VERIFICATION: this file calls
`self.memories[0](xi_t, mem_state)` expecting it to return
`(read_vectors, updated_mem_state)`, inferred from this project's existing
usage of dnc.memory.Memory (`.reset()`, `write_vector_transform` swapped
inside install_stochastic_write_heads()) rather than from having the
literal pytorch-dnc source in hand. Run one batch through this module and
confirm read_vectors.shape == (B, read_heads*cell_size) before trusting a
full run -- if the installed library's Memory.forward returns the pair in
the opposite order, this is the one line to fix.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from dnc.memory import Memory

from mamba_controller.mamba_backbone_parallel import MambaBackboneParallel
from mamba_controller.mamba_controller import MambaControllerWrapper       
from mamba_controller.mamba2_controller import Mamba2ControllerWrapper  


class SplitGraphDNC(nn.Module):
    """Alternate Phase 3, Step 2, Option 5 controller+memory assembly.
    See module docstring for the full design rationale and the explicit
    "not validated yet" framing.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        nr_cells: int = 256,
        cell_size: int = 64,
        read_heads: int = 4,
        num_backbone_blocks: int = 2,
        mamba_variant: str = "mamba1",
        mamba_d_state: int = 16,
        mamba_d_conv: int = 4,
        mamba_expand: int = 2,
        mamba_headdim: int = 64,       # mamba2-only
        combine_reads: bool = True,    # built-in ablation switch, see module docstring
        # v11: combiner mechanism for the sequential addressing step. This is
        # ORTHOGONAL to the backbone (mamba_variant above, still parallel,
        # unchanged) -- it only changes how xi_t is produced from (h_t,
        # read_{t-1}) inside the per-timestep loop.
        #   "linear" (default): the original plain nn.Linear combiner --
        #       stateless, one matmul per timestep. This is the config that
        #       produced the "marvelous efficiency" mamba1-backbone result;
        #       leaving this as default means every existing call site is
        #       byte-for-byte unaffected.
        #   "controller": xi_t is instead produced by a real interleaved
        #       controller cell (MambaControllerWrapper for
        #       combiner_variant="mamba1", Mamba2ControllerWrapper for
        #       "mamba2") carrying its own (conv_state, ssm_state) across
        #       the whole episode -- i.e. the sequential step gets genuine
        #       recurrent SSM capacity instead of a stateless projection,
        #       while the backbone stays exactly as parallel as before.
        combiner_mode: str = "linear",
        combiner_variant: str = "mamba1",   # "mamba1" | "mamba2", only used when combiner_mode="controller"
        combiner_num_blocks: int = 1,       # kept small on purpose -- see module docstring's
                                             # "much smaller and cheaper" framing of this step
        combiner_d_state: int | None = None,   # None -> variant's own paper default (16 / 64)
        combiner_d_conv: int = 4,
        combiner_expand: int = 2,
        combiner_headdim: int = 64,         # mamba2 combiner_variant only
        combiner_ngroups: int = 1,          # mamba2 combiner_variant only
        independent_linears: bool = True,
        device: torch.device | None = None,
        moe_enabled: bool = False,     # deliberately NOT wired in -- see below
        moe_num_experts: int = 8,
        moe_expert_dim: int | None = None,
        moe_capacity_factor: float = 1.5,
        moe_load_balance_alpha: float = 0.01,
    ):
        super().__init__()
        if not independent_linears:
            raise ValueError(
                "SplitGraphDNC requires independent_linears=True, same "
                "requirement as install_stochastic_write_heads() (Phase 1) "
                "-- it needs Memory to expose a standalone "
                "write_vector_transform Linear to swap out."
            )

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = hidden_size
        self.nr_cells = nr_cells
        self.cell_size = cell_size
        self.read_heads = read_heads
        self.w = cell_size
        self.r = read_heads
        self.read_vectors_size = read_heads * cell_size
        self.nn_output_size = hidden_size + self.read_vectors_size
        self.device = device
        self.mamba_variant = mamba_variant
        self.combine_reads = combine_reads

        # Mamba bookkeeping, for checkpoint self-description -- same
        # rationale/convention as MambaDNC (mamba_controller.py).
        self.mamba_d_state = mamba_d_state
        self.mamba_d_conv = mamba_d_conv
        self.mamba_expand = mamba_expand
        self.mamba_headdim = mamba_headdim

        self.backbone = MambaBackboneParallel(
            in_dim=input_size,
            d_model=hidden_size,
            num_blocks=num_backbone_blocks,
            variant=mamba_variant,
            d_state=mamba_d_state,
            d_conv=mamba_d_conv,
            expand=mamba_expand,
            headdim=mamba_headdim,
            device=device,
        )

        if combiner_mode not in ("linear", "controller"):
            raise ValueError(f"SplitGraphDNC: unknown combiner_mode {combiner_mode!r}, "
                              "expected 'linear' or 'controller'.")
        self.combiner_mode = combiner_mode
        self.combiner_variant = combiner_variant
        self.combiner = None
        self.combiner_wrapper = None

        if combiner_mode == "linear":
            if combine_reads:
                self.combiner = nn.Linear(hidden_size + self.read_vectors_size, hidden_size, device=device)
                # Zero-init: at step 0 of training, xi_t == h_t exactly (the
                # split model starts by ignoring the previous read vector
                # entirely and has to learn to use it). Same "start equal to
                # the simpler/prior baseline" philosophy already used
                # elsewhere in this project (StochasticWriteHead's zero-init
                # logvar head, mu_transform copied from the original Linear --
                # stochastic_write_head_v2.py).
                nn.init.zeros_(self.combiner.weight)
                nn.init.zeros_(self.combiner.bias)
        else:  # combiner_mode == "controller"
            # v11: real interleaved cell as the combiner. combine_reads is
            # implicitly True here -- a controller combiner is read-
            # dependent by construction (its whole input is [h_t, read_{t-1}]
            # every step), so there is no meaningful "ablated" variant of
            # this mode the way the Linear combiner has combine_reads=False.
            # Use combiner_mode="linear", combine_reads=False for that check
            # instead (unchanged, still available).
            if combiner_variant == "mamba2":
                print("[SplitGraphDNC] WARNING: combiner_mode='controller' with "
                      "combiner_variant='mamba2' constructs a genuine per-timestep "
                      "interleaved Mamba-2 cell for the addressing step. This is "
                      "known to be inefficient (same reason standalone "
                      "--controller mamba2 runs aren't being pursued) and is wired "
                      "here only for completeness/future comparison -- "
                      "combiner_variant='mamba1' is the configuration expected to "
                      "actually be used.")
            _d_state = combiner_d_state if combiner_d_state is not None else (
                64 if combiner_variant == "mamba2" else 16
            )
            combiner_in_dim = hidden_size + self.read_vectors_size
            if combiner_variant == "mamba2":
                self.combiner_wrapper = Mamba2ControllerWrapper(
                    in_dim=combiner_in_dim,
                    d_model=hidden_size,
                    num_blocks=combiner_num_blocks,
                    d_state=_d_state,
                    d_conv=combiner_d_conv,
                    expand=combiner_expand,
                    headdim=combiner_headdim,
                    ngroups=combiner_ngroups,
                    device=device,
                )
            elif combiner_variant == "mamba1":
                self.combiner_wrapper = MambaControllerWrapper(
                    in_dim=combiner_in_dim,
                    d_model=hidden_size,
                    num_blocks=combiner_num_blocks,
                    d_state=_d_state,
                    d_conv=combiner_d_conv,
                    expand=combiner_expand,
                    device=device,
                )
            else:
                raise ValueError(f"SplitGraphDNC: unknown combiner_variant {combiner_variant!r}, "
                                  "expected 'mamba1' or 'mamba2'.")

        self.memories = []
        self.memories.append(
            Memory(
                input_size=self.output_size,
                nr_cells=self.nr_cells,
                cell_size=self.w,
                read_heads=self.r,
                device=self.device,
                independent_linears=independent_linears,
            )
        )
        # See module docstring's API-compatibility section for why this
        # setattr (not just the list append above) is required.
        setattr(self, "rnn_layer_memory_shared", self.memories[0])

        self.output = nn.Linear(self.nn_output_size, self.input_size, device=device)
        torch.nn.init.kaiming_uniform_(self.output.weight)

        # Option 4 (MoE) interoperability: kept OFF and deliberately NOT
        # wired into the backbone blocks here (unlike mamba_controller.py's
        # MambaControllerWrapper) -- Concept 16/SP-10 (non-additive
        # combination) is why Option 5 is being built and ablated on its
        # own first. moe_layers stays an empty list so
        # pop_total_moe_aux_loss(...) (moe_layer.py) is still safe to call
        # unconditionally from the training script regardless of which
        # controller is active this run.
        self.moe_enabled = moe_enabled
        self.moe_layers = []
        if moe_enabled:
            raise NotImplementedError(
                "SplitGraphDNC: moe_enabled=True is not wired here on "
                "purpose -- see the Concept 16/SP-10 isolation note in this "
                "file's module docstring. Run Option 5 on its own first; "
                "combining with Option 4 is a deliberate future follow-up, "
                "not a default."
            )

        if self.device is not None and getattr(self.device, "type", None) == "cuda":
            self.to(self.device)

    def forward(
        self,
        input: torch.Tensor,
        hx=(None, None, None),
        reset_experience: bool = False,
        pass_through_memory: bool = True,
    ):
        """
        input: (B, T, input_size) -- batch-first, the whole padded episode
            at once (this project always calls rnn(input_seq, ...) with the
            full sequence already assembled -- see Alter_PHASE3_mamba.py's
            training loop and evaluate_traversal()).
        hx: (chx, mhx, last_read). chx is unused by this controller (the
            combiner has no recurrent state of its own -- it's a plain
            per-timestep Linear, not a cell) and is passed through
            untouched. mhx follows dnc.DNC's own convention: None on a
            fresh call, or a single-entry list `[mem_state_dict]` on a
            resumed/continuing call.
        pass_through_memory: when False, skips Memory read AND write for
            EVERY timestep and substitutes zero read-vectors -- this is
            the functional-usage ablation check
            (evaluate_traversal(..., ablate_memory=True)) already wired
            into this project's training loop; see module docstring.

        Returns (output, (chx, mhx, last_read)) with output shaped
        (T, B, input_size) -- see module docstring for why this shape,
        not (B, T, input_size).
        """
        chx, mhx, last_read = hx
        B, T, _ = input.shape
        device = input.device

        # ---- (a) PARALLEL BACKBONE ------------------------------------
        # Single call, whole sequence, zero dependency on memory state.
        # This is the entire "removed from the sequential critical path"
        # half of Option 5.
        H = self.backbone(input)  # (B, T, hidden_size)

        # ---- memory hidden-state init (byte-identical convention to
        # dnc.DNC._init_hidden / MambaDNC._init_hidden's memory branch) ---
        if mhx is None:
            mhx = [self.memories[0].reset(B, erase=reset_experience)]
        else:
            if len(mhx) == 0 or mhx[0] is None:
                mhx = [self.memories[0].reset(B, erase=reset_experience)]
            else:
                mhx = [self.memories[0].reset(B, mhx[0], erase=reset_experience)]
        mem_state = mhx[0]

        if last_read is None:
            read_vec = torch.zeros(B, self.w * self.r, device=device, dtype=H.dtype)
        else:
            read_vec = last_read

        # ---- (b) SEQUENTIAL ADDRESSING STEP ----------------------------
        # The only sequential part left: a cheap per-timestep combine
        # (h_t, read_{t-1}) -> xi_t, followed by the existing
        # Memory.forward() call (already-sequential, already-cheap
        # relative to a full backbone block -- this project never made
        # Memory itself parallel, it only ever removed the BACKBONE from
        # the sequential critical path). Preserves the "read now, decide
        # next hop immediately" loop exactly, per-step.
        # v11: local per-episode state for the controller combiner, if
        # active -- analogous to mem_state above, NOT threaded through chx
        # (chx stays "unused, pass through untouched" per this class's
        # existing convention, since every call site in this project
        # constructs fresh episodes with reset_experience=True). If chx was
        # given (e.g. a future caller that does want continuity), reuse it
        # as the starting state instead of a fresh zero-init.
        combiner_hx = None
        if self.combiner_mode == "controller":
            combiner_hx = chx if chx is not None else self.combiner_wrapper.init_state(
                B, device=device, dtype=H.dtype
            )

        outputs = []
        for t in range(T):
            h_t = H[:, t, :]  # (B, hidden_size)

            if self.combiner_mode == "controller":
                combiner_in = torch.cat([h_t, read_vec], dim=-1).unsqueeze(1)  # (B, 1, hidden+read)
                xi_out, combiner_hx = self.combiner_wrapper(combiner_in, combiner_hx)
                xi_t = h_t + xi_out.squeeze(1)
            elif self.combine_reads:
                xi_t = h_t + self.combiner(torch.cat([h_t, read_vec], dim=-1))
            else:
                # Built-in ablation (see module docstring): addressing
                # never sees any read vector at all.
                xi_t = h_t

            if pass_through_memory:
                read_vec, mem_state = self.memories[0](xi_t, mem_state)
                read_vec = read_vec.reshape(B, -1)  # dnc.memory.Memory returns (B, r, w) unflattened
            else:
                # LB-9/Concept 6 functional-usage check: skip memory read
                # AND write for this timestep, matching every other
                # controller's ablate_memory contract in this project.
                read_vec = torch.zeros(B, self.w * self.r, device=device, dtype=H.dtype)

            step_out = self.output(torch.cat([xi_t, read_vec], dim=-1))  # (B, input_size)
            outputs.append(step_out)

        if self.combiner_mode == "controller":
            chx = combiner_hx  # v11: expose the controller combiner's final state, same
                                # convention as mem_state -> mhx just above

        mhx = [mem_state]
        output = torch.stack(outputs, dim=0)  # (T, B, input_size)

        return output, (chx, mhx, read_vec)