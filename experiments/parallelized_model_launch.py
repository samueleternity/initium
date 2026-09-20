"""
file: Alter_PHASE3_mamba_step2.py
Phase 1 -- add the complexity term, keeping everything else fixed.

This is the original graph-traversal training script (Graves 2016-style
triple encoding, curriculum over synthetic graphs, London Underground as the
held-out OOD generalization test) with exactly one addition: a local,
write-timestep-level KL complexity term on the DNC's write vector, per the
Phase 1 spec.

Nothing else changes:
  - controller: untouched
  - addressing (content-based write key/strength, allocation gate, write
    gate, read modes, link matrix, precedence): untouched
  - BATCH_SIZE, LR, curriculum table: untouched
  - AMP, gradient clipping, LR schedule: untouched

The only change is in `stochastic_write_head.py`: Memory.write_vector_transform
(an nn.Linear) is swapped for a StochasticWriteHead that emits (mu, logvar),
samples v_t = mu + sigma*eps, and exposes the closed-form
KL(q(v_t)||N(0,I)) for this training loop to add to the loss as
L = L_task + beta * L_KL.

--- Compute-budget revision (see Q21-Experiment-Log.md Section 5) ---
Two deliberate deviations from the original Phase 1 draft, both logged:

1. TOTAL_STEPS and model sizing (hidden_size/nr_cells/cell_size/read_heads)
   are now pinned to Run 0's actual executed config (128/128/64/4, 20,000
   steps, batch 16) rather than the paper's 256-config or this script's own
   prior 50,000-step non-debug default. This is required for Run 0 to serve
   as a valid comparison point: "everything else fixed" has to mean fixed
   relative to what Run 0 actually ran, not the paper's spec.

2. beta=0 is dropped from BETAS_TO_SWEEP. Run 1(beta=0) with sample=False is
   mathematically the deterministic Phase 0 write head (mu_transform is
   initialized from the original Linear's weights, and a 200-step debug run
   confirmed KL stays exactly 0 throughout). Run 0 Run B already used the
   identical model size / step budget / batch size, so re-running beta=0 at
   full budget would reproduce, not newly validate, that data point. Run 0
   is used as the beta=0 anchor instead; compute is redirected to
   beta in {0.01, 0.1, 1.0}, which still satisfies Gate 1's "sweep at least
   3-4 values" requirement. Caveat logged in Section 5: Phase 1 uses
   independent_linears=True (required so Memory exposes a standalone
   write_vector_transform for the patch to swap out) where Run 0 used the
   fused interface_weights path, so this is an anchor-by-equivalence, not a
   bit-identical substitution -- flag if Run 0 vs. Run 1(beta>=0.01) shows a
   baseline-level shift KL alone shouldn't produce.

3. Runs are launched one beta per process (see __main__ at the bottom) so
   the 3-value sweep can run in parallel across processes instead of
   sequentially in one script invocation.

--- Step-count fix (see chat log) -----------------------------------------
TOTAL_STEPS was drifting out of sync with the "matches Run 0" claim in the
comment above it (it had been left at 4500/10500 from earlier debug/diagnostic
runs). Run 0 actually executed 20,000 steps at batch 16 (320,000 examples).
Fixed back to 20000 here so Gate 1 comparisons against Run 0 are apples-to-
apples on training budget, not just on model size. See resume support below
for continuing any checkpoint saved under the old 10500-step budget out to
the corrected 20000.

--- v2: LR fix + logging additions (see chat log) --------------------------
1. FIX: `build_london_underground_eval()` no longer calls the global
   `random.seed(1234)` -- it now uses a local `random.Random(1234)`. The
   global reseed was harmless while this function only ran once at the very
   end of a completed run, but is a landmine now that OOD eval runs
   periodically during training (see #1 below): it would otherwise reset
   the training RNG stream to the same point every EVAL_EVERY steps,
   silently repeating training batches immediately after each check.
2. Periodic OOD eval every EVAL_EVERY steps, logged to its own
   `run_{run_id}_ood.csv` (id/ood accuracy + offset vs. step) -- turns the
   ID/OOD offset into a trajectory instead of a single terminal number.
3. Pre-clip gradient norm, logged per LOG_EVERY window (`grad_norm` column).
4. AMP loss-scale value, logged per LOG_EVERY window (`amp_scale` column).
5. Explicit lesson-advance event log, `run_{run_id}_lesson_advances.csv`
   (step, new lesson) -- exact checkpoint-selection reference for a future
   beta-switch experiment, no console-log grepping required.
6. Cumulative wall-clock elapsed, logged per LOG_EVERY window
   (`elapsed_sec` column) and in the final summary.

--- v3: capacity scale-up + LR warmup + two AMP bugfixes (see chat log) -------
Everything below was decided from the beta_0p0_clean full-run log (steps
0-31000): a 60,000-step run that spent its first ~14,000 steps producing 0%
accuracy while `GradScaler` walked its loss scale down from the default
65536 to 0.1 one halving per overflow (`grad_norm nan` recurring the whole
way), then plateaued at lesson 2 (`perfect_frac` oscillating ~46-73%, no
trend) for the following ~16,000 steps once the scale had stabilized.
ADVANCE_THRESHOLD is deliberately NOT touched -- that plateau ceiling (never
observed above ~73%) sits below even a relaxed 0.8 bar, so a threshold
change wouldn't have done anything here; left as-is pending a decision on
that separately.

--- v4: curriculum gate fix (see chat log) ------------------------------
Both beta_0p0_v3 and beta_0p001_v3 stalled at lesson 2/14 for the rest of
their logged budget. Root-caused to two confirmed issues in
TraversalCurriculum.maybe_advance() / evaluate_traversal(), fixed here
without touching ADVANCE_THRESHOLD, TOTAL_STEPS, model size, or anything
outside these two functions:

1. The advance-check evaluated every lesson at num_nodes=nodes_range[1] --
   always the hardest graph size in the lesson -- while training samples
   num_nodes uniformly across the whole range. Fixed: evaluate_traversal()
   now accepts nodes_range and resamples per episode the same way
   sample_episode() does, so the gate measures the lesson as trained.

2. perfect_frac requires exact match on every triple in an episode, so for
   any lesson mixing N chained hops it degrades roughly like
   triple_acc**N. Lesson 2 (50/50 split of 1-/2-hop episodes) needs
   triple_acc ~93% to ever clear a 90% perfect_frac bar -- confirmed
   against the logs (0.5p + 0.5p^2 fits the logged (triple_acc,
   perfect_frac) pairs almost exactly), and it only gets worse at longer
   lessons (lesson 14 tops out at path_length 20). Fixed: the gate now
   advances on triple_acc >= ADVANCE_THRESHOLD instead of perfect_frac.
   perfect_frac is still computed and logged (and returned, for existing
   callers/log schemas) but no longer gates advancement.

Also added: evaluate_traversal(..., hop_breakdown=True) buckets eval
episodes by actual walk length and maybe_advance() now prints a per-hop
accuracy line at every advance-check, so a chaining-specific bottleneck
(hop-2 much worse than hop-1) stays visible going forward even though it's
no longer what's gating progress.

ADDED:
1. Model capacity: hidden_size 256->512, cell_size 128->192, read_heads
   4->8, nr_cells left at 256. These four now live as top-level
   MODEL_HIDDEN_SIZE / MODEL_NR_CELLS / MODEL_CELL_SIZE / MODEL_READ_HEADS
   constants (previously a local tuple inside run(), duplicated a second
   time as a hardcoded literal dict inside save_checkpoint() -- both now
   read the same constants so a future resize can't desync the checkpoint's
   recorded model_config from the model actually being trained).
   nr_cells was deliberately left unchanged: pytorch-dnc's temporal link
   matrix is (batch, 1, nr_cells, nr_cells), so it's the one dimension that
   scales O(N^2) in activation memory rather than ~linearly, and doubling
   it risked not fitting a T4 at this task's longest episodes (~250-260
   steps, no gradient checkpointing in this script). hidden_size and
   cell_size scale ~linearly; read_heads is nearly free (measured ~2%
   memory cost for 4->8) -- all three were the cheap knobs, so those moved
   and nr_cells didn't.
2. Linear LR warmup (WARMUP_STEPS, see CONFIGURATION section) before the
   existing cosine anneal, via lr_at_step(). A wider interface
   (~1.7M->~7.8M params) is more likely to spike early than the old size,
   and the beta_0p0_clean log already showed the model fighting
   instability for its first 14k steps even at the old size -- warmup is
   cheap insurance against the bigger model making that worse, layered on
   top of the AMP fix below rather than instead of it.

REVIEWED, NOT CHANGED:
3. FREE_BITS (0.02/dim): cell_size growing 128->192 means the raw,
   summed-over-dims L_KL is ~1.5x larger for the same per-dim KL, but the
   free-bits floor is already applied per-dimension (see pop_kl() in
   stochastic_write_head.py), so it scales automatically with cell_size --
   no code change needed. Flagging here instead: verify kl_mean/dim and
   clamp_frac at the first checkpoints of this run rather than assume the
   old dynamics carry over unchanged.

FIXED (bugs, not features -- found while reading the beta_0p0_clean log):
4. AMP init_scale: GradScaler previously started at PyTorch's default
   65536 and walked itself down to a stable ~0.1 over ~14,000 steps of
   overflow/skip (grad_norm nan) before the model could learn anything --
   roughly a quarter of that 60k-step run's budget spent finding a usable
   loss scale, not training. Now started at init_scale=128.0 (near where
   it actually stabilized) to skip that walk. This is a starting guess --
   if step-0 grad_norm nan events recur, the model config has changed
   enough (new hidden_size/cell_size) that this needs re-tuning down
   further.
5. GradScaler state was never saved to or restored from checkpoints --
   `scaler = torch.amp.GradScaler(...)` was rebuilt fresh on every resume,
   throwing away whatever scale the previous leg had found and re-running
   a smaller version of the same overflow cascade right after every
   resume (visible in the log: amp_scale jumps back up to 2048 immediately
   after a resume that had ended at 4.0, with two more grad_norm nan
   events before it re-settles). save_checkpoint() now saves
   scaler.state_dict(); the resume path loads it if present, and warns
   (instead of failing) if resuming from a pre-v3 checkpoint that doesn't
   have it.

--- v5: Phase 2 -- learned, periodically-snapshotted prior + logging/audit
    additions (see Experiment-Roadmap.md, "Phase 2 -- Learned prior, and
    the data-conditioning audit (Q97)") ---------------------------------
Design (unchanged from the roadmap spec, implemented here exactly as
specified): replace the fixed p0 = N(0,I) with p0 = N(mu_g, Sigma_g), a
global, diagonal-Gaussian prior fit from the write head's own recent
output. mu_g/Sigma_g are refit on a slow, periodic cadence
(PRIOR_SNAPSHOT_EVERY steps) from a snapshot of recent writes, then frozen
until the next refit -- they are never backpropagated through at any
per-write step. The per-write KL at every other step is still computed
from only that timestep's (mu_t, sigma_t) against the current frozen
(mu_g, Sigma_g) snapshot -- same local formula/shape as Phase 1, just
against a (periodically-updated) non-zero prior instead of a fixed one.
All of the actual prior math lives in stochastic_write_head.py (v2) --
see that file's header for the closed-form KL generalization and the
snapshot-fit procedure. Nothing here changes the addressing, controller,
task loss, curriculum, LR schedule, or AMP handling; every one of those
stays byte-for-byte what v4 does.

What changed / was added in this file, net-new only (see the
per-item comments at each call site for exact mechanics):

1. New CONFIGURATION constants: PRIOR_SNAPSHOT_EVERY (K, the snapshot
   cadence), PRIOR_MIN_LOGVAR / PRIOR_MAX_LOGVAR (the numerical floor/
   ceiling passed through to update_prior_snapshot()).
2. A new periodic "prior snapshot updated" console line + its own CSV log
   file (`run_{run_id}_prior_snapshots.csv`), written every
   PRIOR_SNAPSHOT_EVERY steps: ||mu_g||, diag(Sigma_g) mean/min/max,
   trace(Sigma_g), and the sample count the fit used. This is the direct
   evidence for (a) the Q48 periodic-snapshot classification and (b) a
   collapse/drift check on the prior itself, distinct from -- and not
   visible in -- Phase 1's existing q(v_t)-collapse diagnostic. This is a
   brand new line/file; the existing per-step console line and the
   existing periodic ID/OOD console line are untouched, verbatim.
3. A `snapshot_step` column appended to the END of the existing per-step
   CSV log's header/rows (main `run_{run_id}.csv`) -- i.e. the Phase 1
   schema's columns are all still there, in the same order, so any
   existing Phase 1 log-parsing code keeps working unmodified; this is
   purely an addition. Tags every logged KL window with which frozen
   (mu_g, Sigma_g) snapshot it was computed against, per stochastic_write_
   head.py v2's pop_total_kl() addition -- the audit trail that makes the
   "per-write KL still touches only this timestep + a frozen snapshot"
   locality claim checkable rather than assumed.
4. Checkpoint: save_checkpoint()/load path now also captures/restores
   `prior_state` (mu_g, Sigma_g, and last_snapshot_step per head) via
   stochastic_write_head.py's new get_prior_state()/load_prior_state().
   Model weights alone are not enough to reproduce Phase 2 behavior after
   a resume or a standalone re-eval -- without this, resuming or
   re-evaluating a Phase 2 checkpoint would silently score against
   whatever prior snapshot happens to be sitting in a freshly-constructed
   head's (zero-init) buffers, rather than the one the model was actually
   trained against. Same class of bug the write_mode deterministic/
   sampled mismatch check already exists to catch in
   eval_from_checkpoint.py. Loading warns (doesn't fail) on a pre-v5
   checkpoint that predates this key, exactly like the existing
   scaler_state_dict backward-compatibility warning.
5. run_id: the default naming for a Phase 2 run is now
   f"beta_{beta}_seed{seed}_learnedprior" (dots -> 'p', same as before),
   rather than reusing Phase 1's f"beta_{beta}" tag. Purely so Phase 2 runs
   don't get silently pooled with Phase 1 runs by any grep-based
   aggregation matching "Training (beta_target=...)" or
   "beta_0p0_seed{N}"-style tags against the whole log directory.
6. Pre-existing (not Phase-2-introduced) confound fix: `evaluate_traversal`
   and `build_traversal_episode_from_graph` now take an optional `rng`
   argument (a `random.Random` instance). When omitted, behavior is
   byte-identical to v4 -- all sampling still goes through the global
   `random` module, exactly as before, which is what curriculum ID
   sampling continues to use. Every OOD (London Underground) call site in
   run() now passes a dedicated `ood_rng = random.Random(...)` instance
   (constructed once per run, not re-seeded per call, so it doesn't
   collapse into the "global reseed" bug already fixed for
   build_london_underground_eval() -- see the v2 header note above). This
   decouples the OOD walk sampling (path length, start node, edge choices
   over the fixed graph) from the training curriculum's RNG stream, which
   is the confound identified in Q21-Phase_1_Multi-seed_verification.md
   Section 2g. It doesn't change the Section 2g finding itself (that was
   about a different mechanism, already ruled out there) -- it's a
   independent, pre-existing entanglement worth removing now, before a
   small seed panel (0/1, 6, 2, 5) is used for Phase 2's cross-seed
   comparisons.

--- v6: console output + OOD-RNG checkpoint continuity ------------------
Two small, purely additive follow-ups on top of v5, both requested after
the first Phase 2 run's console output was reviewed:

1. The per-step console line now has a `snapshot_step` token appended at
   the very end (after `clamp_frac`). Every token before it is byte-for-
   byte the same v4 line, in the same order -- this is a pure append, not
   a reformat, so any existing scraping of this line by column position
   still works. (Note: this is a deliberate departure from v5's own stated
   goal of leaving this exact line untouched for cross-phase comparability
   -- v5 kept it verbatim and put snapshot_step only in the CSV; this
   version surfaces it in the console too, on request. The CSV column
   from v5 is unchanged.)
2. `save_checkpoint()`/the resume path now also capture/restore the
   dedicated OOD-sampling `ood_rng`'s own state (`ood_rng_state`, via
   `.getstate()`/`.setstate()`), separately from the pre-existing
   `rng_state` block (which only ever covered the global python/numpy/
   torch/cuda streams). Before this, `ood_rng` was reconstructed fresh
   from `seed` at the top of every process launch regardless of resume,
   so a resumed run's OOD walk sequence would restart from that fixed
   starting point instead of continuing where the pre-resume leg left
   off. This doesn't change the validity of the OOD metric itself (ood_rng
   was already decoupled from the training stream, which was the actual
   v5 fix) -- it only makes a resume's OOD trajectory bit-for-bit
   continuous with the pre-resume leg, the same way `rng_state` and
   `scaler_state_dict` already do for the training stream and AMP scale.
   `ood_rng` is now a required arg to save_checkpoint(); loading warns
   (doesn't fail) on a pre-v6 checkpoint that predates this key, same
   pattern as the existing scaler/prior-state backward-compat warnings.

--- v7: Alternate Phase 3, Step 1 -- optional Mamba-1 controller -----------
(see Experiment-Roadmap.md, "Alternative Phase 3 - fits better to the
programs architecture", Step 1: "Wire into the controller of the DNC code
from Phase 2 the Mamba-1 instead of LSTM (all the KL-prior implementation
must remain for Step 3) ... Library: mamba-ssm.")

This is a controller swap ONLY. Every other Phase 2 component -- memory
addressing (content lookup, allocation, temporal link matrix, read modes),
the stochastic write head + learned/snapshotted-prior KL machinery
(stochastic_write_head_v2.py, completely unmodified, imported exactly as
before), the curriculum, task loss, LR schedule, AMP handling, and the
London-Underground OOD eval -- are untouched. The actual Mamba wiring
(how a Mamba-1 block is driven one DNC timestep at a time, and why that
requires a custom autograd-safe step function rather than the mamba-ssm
library's own inference-only `.step()`) lives in the new file
`mamba_controller.py` (v1) -- see that file's module docstring for the
full design rationale. This header only documents what changed HERE, in
the training script itself:

1. New import: `from mamba_controller import MambaDNC`. `MambaDNC` is a
   `dnc.DNC` subclass that is a strict drop-in superset of `dnc.DNC` for
   `rnn_type='lstm'` (byte-identical behavior -- see mamba_controller.py's
   own smoke-test), and additionally supports `rnn_type='mamba'`. Both
   controller types now share ONE model-construction call site, through
   `MambaDNC`, in place of the plain `dnc.DNC` import used through v6 --
   for `rnn_type='lstm'` this is a no-op behavior change, since `MambaDNC`
   defers entirely to `dnc.DNC.__init__` in that case, not a fork of the
   LSTM path.
2. New CONFIGURATION constants: `CONTROLLER_TYPE` (module-level default,
   `"lstm"` -- i.e. running this script with no changes still reproduces
   Phase 2 exactly, byte-for-byte, since `MambaDNC(..., rnn_type='lstm')`
   defers entirely to stock `dnc.DNC`), and `MAMBA_D_STATE` /
   `MAMBA_D_CONV` / `MAMBA_EXPAND` (Mamba-1's own paper defaults, 16 / 4 /
   2), which only have meaning when `CONTROLLER_TYPE == "mamba"`.
3. `run()` gained a `controller: str = CONTROLLER_TYPE` parameter, threaded
   into the single `MambaDNC(...)` construction call (see #1).
4. `save_checkpoint()`'s `model_config` dict gained `controller_type` and
   (when applicable) `mamba_d_state`/`mamba_d_conv`/`mamba_expand` keys, so
   a checkpoint is self-describing about which controller produced it --
   same rationale as the existing scaler/prior-state/ood-rng backward-
   compat entries: model weights alone don't tell you how to reconstruct
   the model that produced them, and a Mamba checkpoint's `rnn_state_dict`
   keys (`mamba_layer_0....`) are structurally different from an LSTM
   checkpoint's (`lstm_layer_0....`) in a way that would otherwise only
   surface as a confusing `load_state_dict` key-mismatch error.
5. `run_id` naming: a Mamba run's default id gets a `_mambactrl` tag
   appended (e.g. `beta_0p0_seed0_learnedprior_mambactrl`), exactly the
   same "don't let a grep-based aggregation silently pool these" rationale
   already applied to the v5 `_learnedprior` tag -- an LSTM-vs-Mamba
   comparison (this step's whole point) requires the two not to collide in
   `phase1_logs/`.
6. New CLI flag `--controller {lstm,mamba}` (default `lstm`, so existing
   invocations of this script are completely unaffected unless the flag is
   passed explicitly).

Nothing about BATCH_SIZE, TOTAL_STEPS, MODEL_HIDDEN_SIZE/NR_CELLS/
CELL_SIZE/READ_HEADS, the curriculum table, ADVANCE_THRESHOLD, LR/AMP
handling, or BETAS_TO_SWEEP changes in this revision -- Step 1's own
instruction is to "make it work and reach Lesson 3 in reasonable amount of
steps, basically replicate Phase 2 run (firstly beta=0.0)" using the
existing budget as the comparison point, not a new one.

--- v8: Alternate Phase 3, Step 2 (Option 2) -- chunked-parallel
    controller/DNC -----------------------------------------------------
(see Experiment-Roadmap.md, "Attempt to modify DNC in a way that it can in
one way or another simulate SSMs parallelism"; Option 2, "chunked/blockwise
approximate parallelism", was selected -- see chunked_parallel_dnc.py's
module docstring for the full three-option analysis and exactness
argument.)

This wires the new `ChunkedParallelDNC` (chunked_parallel_dnc.py, v1) in as
THIS script's model class, in place of the v7 `MambaDNC` construction call.
`ChunkedParallelDNC` subclasses `MambaDNC` and is a byte-for-byte-equivalent
superset of it at the new class's default `chunk_size=1` (see that file's
module docstring, "Exactness at chunk_size=1") -- so this revision, run with
no flags changed, still reproduces Phase 2 / Step 1 exactly, the same
"every new knob defaults to reproducing the previous behavior" convention
v3-v7 already established for CONTROLLER_TYPE etc.

1. New import: `from chunked_parallel_dnc import ChunkedParallelDNC`. The
   `from mamba_controller import MambaDNC` import is kept (Step 1's own
   construction path is `ChunkedParallelDNC`'s base class, and the plain
   `MambaDNC` name still appears in comments/checkpoint-key documentation
   below), but the actual model-construction call site now uses
   `ChunkedParallelDNC` instead.
2. New CONFIGURATION constant: `CHUNK_SIZE` (module-level default `1` --
   i.e. running this script with no changes still reproduces Step 1 / Phase
   2 exactly). Only has an effect for `chunk_size > 1`; meaningful for
   BOTH controller types (`chunked_parallel_dnc.py`'s chunked `forward()`
   batches the stock `nn.LSTM` call across the chunk too, not just the
   Mamba path -- see that file's module docstring, step 2).
3. `run()` gained a `chunk_size: int = CHUNK_SIZE` parameter, threaded into
   the single `ChunkedParallelDNC(...)` construction call.
4. `save_checkpoint()`'s `model_config` dict gained a `chunk_size` key
   (always present, regardless of controller_type -- unlike the
   mamba_d_state/mamba_d_conv/mamba_expand keys, which are mamba-only,
   chunk_size applies to both controller types) -- same "checkpoint is
   self-describing about what produced it" rationale as every prior
   model_config addition (controller_type, ood_rng_state, prior_state,
   scaler_state_dict).
5. `run_id` naming: a run with `chunk_size > 1` gets a `_chunkN` tag
   appended (e.g. `beta_0p0_seed0_learnedprior_mambactrl_chunk8`) --
   `chunk_size=1` gets no tag, since it's behaviorally identical to a
   Step-1 run and should keep pooling with Step 1's existing `_mambactrl`
   runs the way it always did.
6. New CLI flag `--chunk-size` (default `1`, so existing invocations of
   this script -- including ones already using `--controller mamba` --
   are completely unaffected unless the flag is passed explicitly).

--- v9: Alternate Phase 3, Step 2, Option 3 -- DEER (quasi-Newton parallel
    training path), gated by USE_DEER -------------------------------------
(see Experiment-Roadmap.md, "Attempt to modify DNC in a way that it can in
one way or another simulate SSMs parallelism" -> "3. Iterative fixed-point
parallelization of the full nonlinear recurrence", plus the roadmap's later
"New important info from step 2, option 2 experiment" note: Option 2's
frozen-within-chunk approximation turned out to suppress the curriculum's
phase-transition learning dynamics rather than just trading accuracy for
speed -- a qualitatively different failure than "slightly worse for a lot
faster." That is why Option 3 is added here as an ADDITIONAL parallel-
training path, not a replacement for v8's code: DEER solves the TRUE,
un-approximated per-timestep recurrence via quasi-Newton iteration (Lim
et al. 2024; Gonzalez et al. 2025's diagonal-Jacobian "quasi-DEER"
variant), so there is no structural chunk-boundary error to worry about --
see deer_parallel_dnc.py's own module docstring for the full argument and
correctness self-check. `newton_associative_scan.py` (the generic Newton
engine) and `deer_parallel_dnc.py` (the DNC-specific wiring around it) are
both v1, carried over unmodified from the prior session that wrote them --
nothing in either file changes here; this revision only wires them into
the training script.

`USE_DEER=False` (the default) reproduces v8 exactly, byte-for-byte -- this
revision, run with no flags changed, is a pure addition. When `USE_DEER`
is True, model construction and the training loop's forward call switch to
`DEERParallelDNC` (deer_parallel_dnc.py) INSTEAD OF `ChunkedParallelDNC`;
`CHUNK_SIZE` has no effect in that mode, because `DEERParallelDNC`
subclasses `MambaDNC` directly, per the roadmap's explicit instruction not
to build DEER on top of the chunked forward pass -- linearizing an
already-approximate chunked transition would make DEER converge exactly to
the chunked model's suppressed-phase-transition behavior, defeating the
point of switching to it.

1. New import: `from deer_parallel_dnc import DEERParallelDNC`. The
   `ChunkedParallelDNC` (v8) and `MambaDNC` (v7) imports are both kept --
   DEER is a sibling parallel-training path, not a replacement for either;
   `run()` branches on `use_deer` to decide which class to construct.
2. New CONFIGURATION constants: `USE_DEER` (default `False`), and
   `DEER_MAX_NEWTON_ITERS` / `DEER_TOL` / `DEER_DAMPING` /
   `DEER_MAX_JAC_DIAG_ABS`, threaded straight through to
   `deer_quasi_newton_solve` via `DEERParallelDNC`'s constructor kwargs --
   see newton_associative_scan.py's own module docstring for what each
   controls (damping/max_jac_diag_abs are the numerical safeguards the
   roadmap flags as necessary against DNC's cosine-similarity addressing's
   flat, low-gradient regions producing an ill-conditioned Jacobian).
3. Model construction: `run()` now builds `DEERParallelDNC(...)` in place
   of `ChunkedParallelDNC(...)` when `use_deer=True`, passing the same
   input_size/hidden_size/rnn_type/nr_cells/cell_size/read_heads/
   batch_first/device/independent_linears/mamba_kwargs as before, plus the
   four `deer_*` kwargs. `chunk_size` is not passed on this branch (the
   class has no such parameter).
4. Training-loop forward call: gated on `use_deer`. When True, the call
   passes `use_deer=True` to `DEERParallelDNC.forward`, which returns its
   output tensor already batch-major, `(B, T, INPUT_DIM)` -- UNLIKE the
   inherited sequential `dnc.DNC.forward()` path (used by every other
   configuration in this script, including `DEERParallelDNC` itself
   whenever `use_deer=True` is not passed), which stacks its output
   time-major, `(T, B, INPUT_DIM)`, regardless of `batch_first`. See
   deer_parallel_dnc.py's `_forward_deer`/`_sequential_reference_forward`,
   both of which build their return value via a batch-major
   `torch.cat`/`torch.stack(dim=1)`, not the time-major `torch.stack(dim=0)`
   the inherited path uses. Getting this backwards would silently swap
   batch and time before `output_proj`/`digit_loss` without raising an
   error (both dims are present, just transposed) -- so the existing
   `.transpose(0, 1)` is SKIPPED for the `use_deer=True` branch only.
   `evaluate_traversal`, `curriculum.maybe_advance`, and the OOD eval calls
   never pass `use_deer=True`, so they keep exercising `DEERParallelDNC`'s
   inherited (exact, sequential) forward path with their existing
   `.transpose(0, 1)` unchanged -- eval always measures the same
   sequential ground truth regardless of which class trained the weights,
   which is the correct comparison (DEER's selling point is that its
   trajectory matches the sequential one at convergence, not a different
   one).
5. Startup correctness gate: when `use_deer=True` and this is not a
   resumed run, `run()` samples one training-distribution batch and calls
   deer_parallel_dnc.py's `deer_vs_sequential_max_abs_error` on it BEFORE
   the training loop starts, printing the resulting max-abs-error and
   Newton diagnostics. This is the exact gate the roadmap specifies ("run
   DEER at enough Newton rounds to fully converge and confirm it
   reproduces MambaDNC's sequential rollout... to near machine precision
   -- that's your correctness gate before trusting any speed number"). The
   check runs with `rnn.eval()` (restored to `rnn.train()` immediately
   after) so the write head's KL/prior-snapshot bookkeeping -- which is
   only active in training mode -- is not polluted by this diagnostic
   pass; it is a print-only sanity check (does not abort the run) at the
   model's random init, not a substitute for the per-step Newton
   diagnostics logged during training (#6 below), which are what actually
   tracks convergence as the trajectory being linearized changes over the
   course of training.
6. Per-step logging additions: `newton_iters`, `deer_final_max_abs_delta`,
   `deer_converged` -- appended at the very end of both the per-step
   console line and the main per-step CSV log's header/rows, after
   `snapshot_step` (v5). Pure appends, exactly like every prior per-step
   logging addition in this file -- everything before them is untouched,
   in the same order. Populated from `rnn._last_deer_diagnostics` (set by
   `DEERParallelDNC._forward_deer` after every `use_deer=True` forward
   call) when `use_deer=True`; left at 0 (not applicable) for every other
   configuration, so the CSV schema stays identical across DEER and
   non-DEER runs. This is the ongoing counterpart to the one-off startup
   gate in #5 -- per the roadmap's explicit warning that "exact at
   convergence" is doing real work in that sentence, and under-converging
   Newton reintroduces a different-shaped approximation error, so
   convergence needs to be checked continuously as training progresses,
   not assumed from a fixed round count or a single startup check.
7. `save_checkpoint()`'s `model_config` gained `use_deer` (always present)
   and, only when `use_deer` is True, `deer_max_newton_iters`/`deer_tol`/
   `deer_damping`/`deer_max_jac_diag_abs` -- same "checkpoint is
   self-describing about what produced it" rationale as every prior
   model_config addition. A DEER-trained checkpoint's `rnn_state_dict` has
   no parameters `ChunkedParallelDNC`/`MambaDNC` don't already have (DEER
   adds no new submodules, only a new forward path), so it remains loadable
   into either class -- this field only records which class/hyperparameters
   THIS run actually used, the same non-`load_state_dict`-critical role
   `chunk_size` already plays.
8. `run_id` naming: a run with `use_deer=True` gets a `_deer` tag appended
   (e.g. `beta_0p0_seed0_learnedprior_mambactrl_deer`) -- same
   don't-silently-pool rationale as every prior tag. `chunk_size` is
   ignored under DEER (see #3), so the `_chunkN` tag is never applied
   together with `_deer`.
9. New CLI flags `--use-deer` (default off), `--deer-max-newton-iters`,
   `--deer-tol`, `--deer-damping`, `--deer-max-jac-diag-abs` (all default
   to the module constants, i.e. plain undamped quasi-DEER unless
   explicitly requested otherwise). Passing `--chunk-size` together with
   `--use-deer` prints a warning (chunk_size is simply unused in that
   mode) rather than erroring, since a stale/default `--chunk-size 1` in
   an existing launch command should not break a `--use-deer` run.
"""

import csv
import math
import os
import shutil
import time
import random
import torch
import numpy as np
import torch.nn as nn
from dnc import DNC  # noqa: F401 -- kept for anyone importing DNC from this
                      # module elsewhere; model construction below now goes
                      # through MambaDNC (v7), which defers to this same
                      # dnc.DNC implementation for rnn_type='lstm'.
from model.controller.mamba_controller import MambaDNC  # noqa: F401 -- v7 (Alternate Phase 3,
                                        # Step 1): kept for the docstrings/
                                        # comments above and below that refer
                                        # to it by name; actual model
                                        # construction now goes through
                                        # ChunkedParallelDNC (v8) or
                                        # DEERParallelDNC (v9), both of which
                                        # subclass this.
from Chunked_parallel.chunked_parallel_dnc import ChunkedParallelDNC  # v8 (Alternate Phase 3,
                                        # Step 2, Option 2): see that file
                                        # for the chunked-parallel forward
                                        # pass. chunk_size=1 (the default) is
                                        # byte-for-byte equivalent to plain
                                        # MambaDNC -- see that file's module
                                        # docstring, "Exactness at chunk_size=1".
                                        # Used when USE_DEER is False (v9).
from DEER.deer_parallel_dnc import (  # v9 (Alternate Phase 3, Step 2, Option 3)
    DEERParallelDNC,
    deer_vs_sequential_max_abs_error,
)  # deer_parallel_dnc.py (v1) + newton_associative_scan.py (v1, imported by
   # it) are carried over unmodified from the prior session -- see this
   # file's v9 header note. Used in place of ChunkedParallelDNC when
   # USE_DEER is True.

from model.memory_manipulation.stochastic_write_head_v2 import (
    install_stochastic_write_heads, pop_total_kl,
    update_all_prior_snapshots, get_prior_state, load_prior_state,  # v5 (Phase 2)
)

# ==========================================
# 1. CONFIGURATION
# ==========================================
BATCH_SIZE = 16              # bucketed, so padding waste stays low even >1
TOTAL_STEPS = 120000          
LOG_EVERY = 100
EVAL_EVERY = 1000            # curriculum-advance eval cadence; matches the runs that already produced usable data (fixed from 10)
LR = 3e-4
LR_MIN = 3e-5
LR_DECAY_STEPS = TOTAL_STEPS       # cosine anneal from LR to LR_MIN over this many steps
WARMUP_STEPS = 1000                # v3: linear LR warmup 0 -> LR before the cosine anneal starts
USE_AMP = True
SEED = 0
AMP_INIT_SCALE = 128.0             # v3 fix: GradScaler previously defaulted to 65536 and spent
                                    # ~14,000 steps of the beta_0p0_clean run halving its way down
                                    # to a stable ~0.1 (see header note) before training worked at
                                    # all -- start near where it actually stabilized instead.

# v3: model capacity -- see header note for what changed and why.
# Was hidden_size, nr_cells, cell_size, read_heads = 256, 256, 128, 4.
# Promoted to module-level constants (previously a local tuple in run(),
# and separately duplicated as a hardcoded dict literal in save_checkpoint())
# so both places read the same numbers and can't desync.
MODEL_HIDDEN_SIZE = 512
MODEL_NR_CELLS = 256               # left unchanged -- see header note (O(N^2) link matrix)
MODEL_CELL_SIZE = 192
MODEL_READ_HEADS = 8

# v7 (Alternate Phase 3, Step 1): controller selection. "lstm" reproduces
# Phase 2 exactly (MambaDNC defers to stock dnc.DNC for any non-'mamba'
# rnn_type -- see mamba_controller.py). Overridable per-run via --controller.
CONTROLLER_TYPE = "lstm"
# Mamba-1 hyperparameters -- only meaningful when CONTROLLER_TYPE=="mamba".
# Defaults are Mamba-1's own paper defaults (Gu & Dao 2024, Section 3.4).
MAMBA_D_STATE = 16
MAMBA_D_CONV = 4
MAMBA_EXPAND = 2

# v8 (Alternate Phase 3, Step 2, Option 2): chunk size for
# ChunkedParallelDNC's chunked-parallel forward pass -- number of REAL
# timesteps processed per controller call. 1 (the default) reproduces the
# exact sequential Step-1/Phase-2 forward pass, byte-for-byte (see
# chunked_parallel_dnc.py's module docstring, "Exactness at chunk_size=1").
# Meaningful for both controller types; overridable per-run via
# --chunk-size. Has NO effect when USE_DEER is True (v9) -- see that
# constant's own comment.
CHUNK_SIZE = 1

# v9 (Alternate Phase 3, Step 2, Option 3): DEER, gated off by default so
# this revision, run with no flags changed, reproduces v8 exactly.
# use_deer=True switches model construction from ChunkedParallelDNC to
# DEERParallelDNC (deer_parallel_dnc.py) and switches the training loop's
# forward call to use_deer=True (see this file's v9 header note for the
# batch-major-vs-time-major output-shape consequence of that switch).
# CHUNK_SIZE is ignored in this mode -- DEER is built on plain MambaDNC,
# never on the chunked forward pass, per the roadmap's explicit warning
# against linearizing an already-approximate transition.
USE_DEER = False
# Newton-solve hyperparameters, threaded straight through to
# newton_associative_scan.py's deer_quasi_newton_solve via
# DEERParallelDNC's constructor kwargs. Defaults match that function's own
# defaults (50 max rounds; tol auto-selected per dtype, 1e-4 float32 /
# 1e-7 float64, matching Lim et al.'s reference implementation; no
# damping/clamp -- plain, undamped quasi-DEER). Only meaningful when
# USE_DEER is True.
DEER_MAX_NEWTON_ITERS = 50
DEER_TOL = None
DEER_DAMPING = 0.0
DEER_MAX_JAC_DIAG_ABS = None

LABEL_RANGE = 1000
LABEL_DIGITS = 3             # each label is a 3-digit number, 0-999
DIGIT_BASE = 10              # one-hot over digits 0-9
LABEL_DIM = LABEL_DIGITS * DIGIT_BASE     # 30
TRIPLE_DIM = 3 * LABEL_DIM                # 90  (source + edge + destination)
NUM_PHASE_CHANNELS = 2                    # [phase-transition, prediction-required]
INPUT_DIM = TRIPLE_DIM + NUM_PHASE_CHANNELS  # 92, matches Methods


TRAVERSAL_CURRICULUM = [
    ((3, 10),  (2, 4), (1, 1)),
    ((3, 10),  (2, 4), (1, 2)),
    ((5, 10),  (2, 4), (1, 3)),
    ((5, 10),  (2, 4), (1, 4)),
    ((10, 15), (2, 4), (1, 4)),
    ((10, 15), (2, 4), (1, 5)),
    ((10, 20), (2, 4), (1, 5)),
    ((10, 20), (2, 4), (1, 6)),
    ((10, 30), (2, 4), (1, 6)),
    ((10, 30), (2, 4), (1, 7)),
    ((10, 30), (2, 4), (1, 8)),
    ((10, 30), (2, 4), (1, 9)),
    ((10, 40), (2, 6), (1, 10)),
    ((10, 40), (2, 6), (1, 20)),
]

ADVANCE_THRESHOLD = 0.85       # 85% modal accuracy
OLD_LESSON_MIX_RATE = 0.10     # 10% of exemplars drawn from earlier lessons
EVAL_BATCH_SIZE = 100          # episodes per lesson-completion check

LONDON_UNDERGROUND_EDGES_RAW = [
    ("OxfordCircus", "TottenhamCtRd", "Central"),
    ("TottenhamCtRd", "OxfordCircus", "Central"),
    ("OxfordCircus", "PiccadillyCircus", "Bakerloo"),
    ("PiccadillyCircus", "OxfordCircus", "Bakerloo"),
    ("OxfordCircus", "NottingHillGate", "Central"),
    ("OxfordCircus", "Euston", "Victoria"),
    ("BakerSt", "Marylebone", "Circle"),
    ("BakerSt", "Marylebone", "Bakerloo"),
    ("BakerSt", "OxfordCircus", "Bakerloo"),
    ("LeicesterSq", "CharingCross", "Northern"),
    ("TottenhamCtRd", "LeicesterSq", "Northern"),
    ("LeicesterSq", "PiccadillyCircus", "Piccadilly"),
    ("PiccadillyCircus", "LeicesterSq", "Piccadilly"),
    ("PiccadillyCircus", "GreenPark", "Piccadilly"),
    ("GreenPark", "PiccadillyCircus", "Piccadilly"),
    ("GreenPark", "OxfordCircus", "Victoria"),
    ("GreenPark", "Victoria", "Victoria"),
    ("Victoria", "GreenPark", "Victoria"),
    ("CharingCross", "PiccadillyCircus", "Bakerloo"),
    ("PiccadillyCircus", "CharingCross", "Bakerloo"),
    ("LeicesterSq", "TottenhamCtRd", "Northern"),
    ("CharingCross", "LeicesterSq", "Northern"),
]

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# ---- Phase 1 additions ----------------------------------------------------
BETAS_TO_SWEEP = [0.0, 0.001]         # beta=0 anchored to Run 0 instead -- see header note
KL_ANNEAL_STEPS = 8000                    # ramp beta 0 -> target over this many steps
FREE_BITS = 0.02                          # per-dimension KL floor (nats); 0.0 disables
# v3 review (not a change): cell_size 128->192 makes the raw, summed-over-
# dims L_KL ~1.5x larger for the same per-dim KL, but this floor is already
# applied per-dimension in pop_kl() (stochastic_write_head.py), so it scales
# automatically with cell_size -- left as 0.02. Verify kl_mean/dim and
# clamp_frac at this run's first checkpoints rather than assume unchanged.
LOG_DIR = "./phase1_logs"
OOD_EVAL_EPISODES = 200
OOD_PATH_LENGTH_RANGE = (3, 5)

# ---- Phase 2 additions (v5) ------------------------------------------
# Learned-prior snapshot cadence and numerical floor/ceiling -- see
# stochastic_write_head.py v2 (update_prior_snapshot) for what these
# actually gate. PRIOR_SNAPSHOT_EVERY deliberately mirrors EVAL_EVERY's
# order of magnitude (both are "periodic, off the hot per-step path"
# cadences) but is its own constant since there's no requirement the two
# coincide.
PRIOR_SNAPSHOT_EVERY = 2000
PRIOR_MIN_LOGVAR = -6.0   # floor on log(Sigma_g) -- prevents a silent Sigma_g -> 0 collapse
PRIOR_MAX_LOGVAR = 6.0    # ceiling -- symmetric guard against the fit blowing up
OOD_EVAL_EPISODES_PERIODIC = 50   # log addition #1: lighter episode count for
# the every-EVAL_EVERY-steps OOD read during training, vs. the full
# OOD_EVAL_EPISODES=200 used for the one-off final eval. Keeps the added
# eval cost small (this runs every 1,000 steps) while still giving a usable
# trajectory; the terminal eval still uses the full 200-episode count.

# ---- Checkpointing ----------------------------------------------------
# Nothing about training changes here -- this only persists the trained
# weights + enough metadata to resume/re-eval, so that widening the OOD
# eval set later, or extending a run to more steps, doesn't require
# retraining from scratch.
CHECKPOINT_DIR = "./phase1_checkpoints"
CHECKPOINT_EVERY = 2000     # periodic safety checkpoint, in addition to end-of-run


def save_checkpoint(path, rnn, output_proj, stochastic_heads, optimizer,
                     curriculum, step, beta_target, run_id, scaler, ood_rng,
                     controller_type=CONTROLLER_TYPE,  # v7: see model_config note below
                     chunk_size=CHUNK_SIZE,  # v8: see model_config note below
                     use_deer=USE_DEER,  # v9: see model_config note below
                     deer_max_newton_iters=DEER_MAX_NEWTON_ITERS,
                     deer_tol=DEER_TOL,
                     deer_damping=DEER_DAMPING,
                     deer_max_jac_diag_abs=DEER_MAX_JAC_DIAG_ABS):
    """Save everything needed to resume training or re-run eval later:
      - model + output-projection + optimizer state
      - LR is NOT saved separately -- it's now a pure function of `step`
        (see lr_at_step()/set_lr() in run()), clamped at LR_DECAY_STEPS, so
        it's fully reconstructible from `step` alone on resume. This also
        sidesteps the CosineAnnealingLR periodicity bug (see chat log /
        Q21-Experiment-Log Section 5) that previously let LR drift back up
        after LR_DECAY_STEPS via the old scheduler's carried internal state.
      - the stochastic write head(s)' own params are already inside
        rnn.state_dict() (they were installed as submodules via the
        write_vector_transform swap), so no separate save needed for those
      - RNG state (python/numpy/torch/cuda) for reproducibility
      - curriculum lesson, so eval-time episode difficulty matches training
      - v3 fix: GradScaler state (scale, growth tracker, etc). Previously
        NOT saved -- `scaler = torch.amp.GradScaler(...)` was rebuilt fresh
        on every resume, throwing away whatever scale the prior leg had
        found and re-running a smaller version of the AMP overflow cascade
        right after every resume (see header note). `scaler` is now a
        required arg so this can't silently regress back to being dropped.
      - v5 (Phase 2): `prior_state` -- mu_g, Sigma_g (as prior_logvar), and
        last_snapshot_step for every installed stochastic write head (see
        get_prior_state() in stochastic_write_head.py). NOT covered by
        rnn.state_dict(): prior_mu/prior_logvar are registered buffers, so
        they normally WOULD ride along inside rnn.state_dict() automatically
        -- but they are captured here explicitly, as their own top-level
        checkpoint key, so this checkpoint's prior state is self-describing
        and independently loadable (e.g. by eval_from_checkpoint.py without
        needing to reload the entire rnn state_dict), and so a Phase-1-era
        reader that doesn't expect these buffers can still load
        rnn_state_dict unmodified. Mirrors the reasoning already applied to
        scaler_state_dict: model weights alone are not enough to reproduce
        this run's behavior on resume/re-eval.
      - v6: `ood_rng_state` -- the dedicated OOD-sampling random.Random
        instance's own state (via .getstate()), saved separately from the
        existing `rng_state` block (which only ever covered the global
        python/numpy/torch/cuda streams). Without this, a resumed run would
        silently re-seed ood_rng back to its start-of-process value
        (deterministic from `seed` alone -- see run()) instead of
        continuing the exact OOD walk sequence the pre-resume leg had
        reached, which is a smaller version of the same "silently evaluate
        against the wrong state" bug class prior_state (above) exists to
        catch -- it doesn't change the metric's validity (ood_rng was
        always decoupled from the training stream, which was the actual
        fix), but it does mean a resume's OOD trajectory isn't bit-for-bit
        continuous with the pre-resume leg unless this is restored. `ood_rng`
        is now a required arg so this can't silently regress the way the
        scaler-state omission originally did.
      - v7 (Alternate Phase 3, Step 1): `model_config` gained `controller_type`
        and (only when it's "mamba") `mamba_d_state`/`mamba_d_conv`/
        `mamba_expand`. Same "self-describing checkpoint" rationale as
        prior_state/scaler_state_dict above: an LSTM checkpoint's
        rnn_state_dict has `lstm_layer_0...` keys, a Mamba checkpoint's has
        `mamba_layer_0...` keys (see mamba_controller.py's MambaDNC), and
        without this field a reader has no way to know which model class to
        reconstruct before attempting `load_state_dict`.
      - v8 (Alternate Phase 3, Step 2, Option 2): `model_config` gained
        `chunk_size` (always present -- unlike the mamba_d_state/
        mamba_d_conv/mamba_expand keys, this applies to both controller
        types, since `chunked_parallel_dnc.py`'s chunked forward() batches
        the stock LSTM path across a chunk too). `rnn_state_dict` itself is
        unaffected by chunk_size (see chunked_parallel_dnc.py's module
        docstring: the chunk-capable wrapper only ADDS a method, never a
        parameter), so a checkpoint saved at one chunk_size loads cleanly
        into `ChunkedParallelDNC(..., chunk_size=<any other value>)` --
        this key is recorded purely so eval/resume tooling knows what
        chunk_size THIS run was trained/logged under, not because it's
        required for `load_state_dict` to succeed.
      - v9 (Alternate Phase 3, Step 2, Option 3): `model_config` gained
        `use_deer` (always present) and, only when it's True,
        `deer_max_newton_iters`/`deer_tol`/`deer_damping`/
        `deer_max_jac_diag_abs`. Same rationale as chunk_size above:
        `DEERParallelDNC` adds no new parameters over `MambaDNC` (only a
        new forward path), so `rnn_state_dict` remains loadable regardless
        -- this field records which class/hyperparameters THIS run used,
        for reconstructing an equivalent model on resume/re-eval.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    model_config = {
        "input_size": INPUT_DIM, "hidden_size": MODEL_HIDDEN_SIZE,
        "nr_cells": MODEL_NR_CELLS, "cell_size": MODEL_CELL_SIZE,
        "read_heads": MODEL_READ_HEADS,
        "controller_type": controller_type,  # v7
        "chunk_size": chunk_size,  # v8
        "use_deer": use_deer,  # v9
    }
    if controller_type == "mamba":  # v7
        model_config.update({
            "mamba_d_state": MAMBA_D_STATE,
            "mamba_d_conv": MAMBA_D_CONV,
            "mamba_expand": MAMBA_EXPAND,
        })
    if use_deer:  # v9
        model_config.update({
            "deer_max_newton_iters": deer_max_newton_iters,
            "deer_tol": deer_tol,
            "deer_damping": deer_damping,
            "deer_max_jac_diag_abs": deer_max_jac_diag_abs,
        })
    torch.save({
        "step": step,
        "run_id": run_id,
        "beta_target": beta_target,
        "model_config": model_config,
        "rnn_state_dict": rnn.state_dict(),
        "output_proj_state_dict": output_proj.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scaler_state_dict": scaler.state_dict(),  # v3 fix
        "curriculum_lesson": curriculum.lesson,
        "prior_state": get_prior_state(stochastic_heads),  # v5 (Phase 2)
        "ood_rng_state": ood_rng.getstate(),  # v6
        "rng_state": {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        },
    }, path)


def load_checkpoint_for_resume(path, device):
    """Load a checkpoint saved by save_checkpoint(), for CONTINUING training
    (as opposed to eval_from_checkpoint.py, which loads for eval only and
    doesn't need optimizer/scheduler/RNG state).

    map_location='cpu' (not `device`): torch.load's map_location moves
    EVERY tensor in the checkpoint, including the RNG-state ByteTensors.
    torch.cuda.set_rng_state_all() requires those to stay plain CPU
    ByteTensors -- if map_location drags them onto CUDA they become
    torch.cuda.ByteTensor and set_rng_state_all rejects them. Loading to
    CPU and letting rnn.load_state_dict()/optimizer.load_state_dict() do
    their own (automatic) device casting for the model/optimizer tensors
    avoids that without needing two different map_locations for one file.

    weights_only=False: this checkpoint stores non-tensor python/numpy RNG
    state alongside the tensors, and PyTorch >=2.6 defaults torch.load to
    weights_only=True. Safe here since it's a checkpoint we produced
    ourselves, not a downloaded/untrusted file.
    """
    return torch.load(path, map_location='cpu', weights_only=False)


def restore_rng_state(rng_state):
    random.setstate(rng_state["python"])
    np.random.set_state(rng_state["numpy"])
    torch.set_rng_state(rng_state["torch"].cpu() if torch.is_tensor(rng_state["torch"]) else rng_state["torch"])
    if rng_state["cuda"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(rng_state["cuda"])


# ==========================================
# 2. GRAPH GENERATOR  (unchanged)
#    Samples a fresh random graph on every call.
#    Nothing is cached -- unbounded stream of graphs.
# ==========================================
def generate_graph(num_nodes, k_range, label_range=LABEL_RANGE):
    """
    num_nodes   : N, number of nodes in this graph
    k_range     : (k_min, k_max) inclusive range for out-degree K
    label_range : label pool size (default 1000)

    Returns:
        edges       : list of (source_label, edge_label, dest_label)
        node_labels : list[int], N labels assigned to nodes
        adjacency   : dict node_idx -> list of (dest_node_idx, edge_label)
    """
    points = np.random.uniform(0.0, 1.0, size=(num_nodes, 2))
    node_labels = random.sample(range(label_range), num_nodes)

    dists = np.linalg.norm(points[:, None, :] - points[None, :, :], axis=-1)
    np.fill_diagonal(dists, np.inf)

    k_min, k_max = k_range
    adjacency = {i: [] for i in range(num_nodes)}
    edges = []

    for i in range(num_nodes):
        k = random.randint(k_min, min(k_max, num_nodes - 1))
        nearest_idx = np.argsort(dists[i])[:k]

        edge_labels = random.sample(node_labels, k)

        for dest_idx, edge_label in zip(nearest_idx, edge_labels):
            src_label = node_labels[i]
            dst_label = node_labels[int(dest_idx)]
            edges.append((src_label, edge_label, dst_label))
            adjacency[i].append((int(dest_idx), edge_label))

    return edges, node_labels, adjacency


# ==========================================
# 3. TRIPLE ENCODER  (unchanged)
# ==========================================
def encode_label(label):
    vec = torch.zeros(LABEL_DIM)
    if label is None:
        return vec
    digits = f"{label:03d}"
    for pos, ch in enumerate(digits):
        vec[pos * DIGIT_BASE + int(ch)] = 1.0
    return vec

def encode_triple(source, edge, dest, phase_transition=0.0, prediction_required=0.0):
    src_vec = encode_label(source)
    edge_vec = encode_label(edge)
    dst_vec = encode_label(dest)
    phase_vec = torch.tensor([phase_transition, prediction_required], dtype=torch.float32)
    return torch.cat([src_vec, edge_vec, dst_vec, phase_vec])

def collate_fn(batch):
    max_len = max(len(x[0]) for x in batch)
    B = len(batch)

    padded_input = torch.zeros(B, max_len, INPUT_DIM)
    padded_targets = torch.zeros(B, max_len, 9, dtype=torch.long)
    padded_mask = torch.zeros(B, max_len)

    for i, (inp, tgt, mask) in enumerate(batch):
        T = inp.size(0)
        padded_input[i, :T] = inp
        padded_targets[i, :T] = tgt
        padded_mask[i, :T] = mask

    return padded_input, padded_targets, padded_mask


# ==========================================
# 4. LOSS FUNCTION  (unchanged: this is L_task)
# ==========================================
def digit_loss(output, target_digits, answer_mask):
    B, T, _ = output.shape
    logits = output.view(B, T, 9, DIGIT_BASE)
    log_probs = torch.log_softmax(logits, dim=-1)

    gathered = torch.gather(log_probs, -1, target_digits.unsqueeze(-1)).squeeze(-1)
    per_step_loss = -gathered.sum(dim=-1)

    mask = answer_mask.float()
    total = (per_step_loss * mask).sum()
    denom = mask.sum().clamp(min=1.0)
    return total / denom


# ==========================================
# 5. EPISODE CONSTRUCTION  (unchanged)
# ==========================================
def label_to_digits(label):
    s = f"{label:03d}"
    return [int(c) for c in s]

def triple_to_digit_targets(source, edge, dest):
    return label_to_digits(source) + label_to_digits(edge) + label_to_digits(dest)

def build_traversal_episode_from_graph(edges, node_labels, adjacency, num_nodes, path_length_range, rng=None):
    """
    rng: v5 (Phase 2) addition. Optional random.Random instance to draw
    edge-shuffle/path-sampling randomness from. Defaults to None, in which
    case this uses the global `random` module exactly as before (byte-
    identical to v4) -- this is what curriculum ID sampling continues to
    use. Passing a dedicated `random.Random(...)` instance (as run()'s OOD
    eval calls now do) decouples that call's randomness from whichever
    other stream is calling this function, without changing anything about
    what's sampled or how -- see the v5 header note for the confound this
    fixes.
    """
    rng = rng if rng is not None else random
    inputs, target_digits, answer_mask = [], [], []

    def add_step(src, edge, dst, phase_transition, prediction_required, target_triple=None):
        inputs.append(encode_triple(src, edge, dst, phase_transition, prediction_required))
        if target_triple is not None:
            target_digits.append(triple_to_digit_targets(*target_triple))
            answer_mask.append(1)
        else:
            target_digits.append([0] * 9)
            answer_mask.append(0)

    shuffled_edges = edges[:]
    rng.shuffle(shuffled_edges)
    for i, (s, e, d) in enumerate(shuffled_edges):
        add_step(s, e, d, 1.0 if i == 0 else 0.0, 0.0)

    path_length = rng.randint(*path_length_range)
    start_idx = rng.randrange(num_nodes)
    cur = start_idx
    walk = []
    for _ in range(path_length):
        if not adjacency[cur]:
            break
        dst_idx, edge_label = rng.choice(adjacency[cur])
        walk.append((cur, edge_label, dst_idx))
        cur = dst_idx
    if not walk:
        return None

    for i, (src_idx, edge_label, _dst_idx) in enumerate(walk):
        src = node_labels[src_idx] if i == 0 else None
        add_step(src, edge_label, None, 1.0 if i == 0 else 0.0, 0.0)

    for i, (src_idx, edge_label, dst_idx) in enumerate(walk):
        target = (node_labels[src_idx], edge_label, node_labels[dst_idx])
        add_step(None, None, None, 1.0 if i == 0 else 0.0, 1.0, target_triple=target)

    input_seq = torch.stack(inputs)
    target_digits_t = torch.tensor(target_digits, dtype=torch.long)
    answer_mask_t = torch.tensor(answer_mask, dtype=torch.float32)
    return input_seq, target_digits_t, answer_mask_t

def build_traversal_episode(num_nodes, k_range, path_length_range, rng=None):
    # v5: rng passthrough -- see build_traversal_episode_from_graph(). Note
    # generate_graph() itself still always uses the global `random`/`numpy`
    # RNG streams (unchanged) -- it's only ever called for ID episodes
    # (fresh random graphs), never for the fixed-graph OOD path, so there is
    # nothing to decouple there.
    edges, node_labels, adjacency = generate_graph(num_nodes, k_range)
    return build_traversal_episode_from_graph(edges, node_labels, adjacency, num_nodes, path_length_range, rng=rng)


# ==========================================
# 6. CURRICULUM  (unchanged, but instantiated fresh per run -- see run())
# ==========================================
class TraversalCurriculum:
    def __init__(self, table=TRAVERSAL_CURRICULUM):
        self.table = table
        self.lesson = 0

    def _sample_lesson_params(self):
        if self.lesson > 0 and random.random() < OLD_LESSON_MIX_RATE:
            idx = random.randint(0, self.lesson - 1)
        else:
            idx = self.lesson
        return self.table[idx]

    def sample_episode(self):
        nodes_range, out_degree_range, path_len_range = self._sample_lesson_params()
        ep = None
        while ep is None:
            num_nodes = random.randint(*nodes_range)
            ep = build_traversal_episode(num_nodes, out_degree_range, path_len_range)
        return ep

    def maybe_advance(self, model, device, step=None):
        """Returns (lesson, id_triple_acc, id_perfect_frac). The two accuracy
        values are returned (not just used internally for the advance
        decision) so callers -- specifically the log addition #1 periodic
        OOD eval below -- can log ID and OOD numbers from the same step
        without a second, redundant ID eval pass.

        --- v4: gate-design fix (see chat log, confirmed against
        beta_0p0_v3/beta_0p001_v3 logs) ---------------------------------
        Two changes here, both diagnosed from data, neither touching
        ADVANCE_THRESHOLD itself:

        1. Eval now samples num_nodes ~ nodes_range (nodes_range=...) the
           same way sample_episode() does during training, instead of
           pinning every advance-check to num_nodes=nodes_range[1] (the
           hardest graph size in the lesson, every time). The gate now
           measures "the lesson as trained," not its worst-case slice.

        2. The gate now advances on triple_acc (per-triple accuracy)
           instead of perfect_frac. perfect_frac requires every triple in
           an episode correct, so for a lesson whose episodes mix N
           chained hops, it's approximately triple_acc**N (worse than
           that once errors correlate) -- e.g. lesson 2's 50/50 split of
           1- and 2-hop episodes needs triple_acc ~93% to clear a 90%
           perfect_frac bar, and lesson 14 (path_length up to 20) would
           need something close to unreachable. That's a property of
           episode-exact-match compounding over hops, not of what the
           model has actually learned, and it's what produced the
           logged 55k-step stall at lesson 2 in both beta runs (confirmed
           against the 0.5*p + 0.5*p^2 fit to the logged (triple_acc,
           perfect_frac) pairs). triple_acc doesn't have this compounding
           artifact -- it's a flat per-decision accuracy regardless of
           how many hops a lesson's episodes chain, so ADVANCE_THRESHOLD
           means the same thing at every lesson. perfect_frac is still
           computed, logged, and returned unchanged (callers/log schemas
           depend on it), it's just no longer what gates advancement.
        """
        nodes_range, out_degree_range, path_len_range = self.table[self.lesson]
        triple_acc, perfect_frac, hop_breakdown = evaluate_traversal(
            model, device, num_episodes=EVAL_BATCH_SIZE, verbose_n=0,
            nodes_range=nodes_range, k_range=out_degree_range, path_length_range=path_len_range,
            hop_breakdown=True,
        )
        if hop_breakdown:
            breakdown_str = ", ".join(
                f"{hops}-hop: acc {acc:.1f}% perfect {pf:.1f}% (n={n})"
                for hops, (acc, pf, n) in hop_breakdown.items()
            )
            print(f"    [lesson {self.lesson + 1} eval by hop count] {breakdown_str}")

        if triple_acc / 100.0 >= ADVANCE_THRESHOLD and self.lesson < len(self.table) - 1:
            self.lesson += 1
            print(f">>> Curriculum advanced to lesson {self.lesson + 1}/{len(self.table)}")
            # Log addition #4: record the exact step of every lesson advance,
            # so future switch-in points (e.g. "post lesson-2 breakthrough")
            # can be read directly from this file instead of grepped from
            # console output after the fact.
            writer = getattr(self, "advance_log_writer", None)
            if writer is not None:
                writer.writerow([step, self.lesson + 1, len(self.table)])
                self.advance_log_file.flush()
        return self.lesson, triple_acc, perfect_frac


def sample_batch(curriculum, batch_size):
    episodes = [curriculum.sample_episode() for _ in range(batch_size)]
    return collate_fn(episodes)


# ==========================================
# 8. DIAGNOSTICS  (unchanged)
# ==========================================
def prediction_diversity(output, target_digits, answer_mask):
    mask = answer_mask.bool()
    if mask.sum() == 0:
        return 0.0
    logits = output.view(*output.shape[:2], 9, DIGIT_BASE)
    preds = logits.argmax(dim=-1)          # (B,T,9)
    preds_masked = preds[mask]             # (num_answer_steps, 9)
    if preds_masked.numel() == 0:
        return 0.0
    diversities = [preds_masked[:, d].unique().numel() for d in range(9)]
    return sum(diversities) / 9.0


# ==========================================
# 9. EVALUATION  (unchanged)
# ==========================================
def decode_prediction(output_step):
    logits = output_step.view(9, DIGIT_BASE)
    digit_preds = logits.argmax(dim=-1).tolist()
    src = int("".join(str(d) for d in digit_preds[0:3]))
    edge = int("".join(str(d) for d in digit_preds[3:6]))
    dst = int("".join(str(d) for d in digit_preds[6:9]))
    return src, edge, dst

def evaluate_traversal(model, device, num_episodes=100, verbose_n=3,
                       num_nodes=None, nodes_range=None, k_range=None, path_length_range=None,
                       fixed_graph=None, hop_breakdown=False, rng=None):
    """
    num_nodes: fixed graph size for every eval episode (old behavior).
    nodes_range: (lo, hi) tuple -- if given (and num_nodes is None), num_nodes
        is re-sampled per episode via random.randint(*nodes_range), i.e. the
        SAME distribution sample_episode() trains on, rather than pinning to
        one size. Passing both is an error; passing neither is only valid
        with fixed_graph.
    hop_breakdown: if True, also return a dict {path_length: (triple_acc,
        perfect_frac, n_episodes)} bucketing this eval's episodes by their
        actual walk length, so a chaining-specific bottleneck (hop-2 much
        worse than hop-1) can be distinguished from general accuracy noise.
    rng: v5 (Phase 2) addition. Optional random.Random instance threaded
        into build_traversal_episode[_from_graph] for this call's sampling.
        Defaults to None (global `random` stream, byte-identical to v4).
        run()'s OOD (fixed_graph) call sites now pass a dedicated
        `ood_rng`, decoupling OOD walk sampling from the ID curriculum's
        RNG stream -- see the v5 header note. ID calls (curriculum.
        maybe_advance) pass nothing, so their behavior is unchanged.

        Note (v9): this function never passes use_deer=True to `model(...)`
        below -- every caller (curriculum.maybe_advance, the periodic and
        terminal OOD checks) always exercises the model's inherited
        sequential forward path, regardless of whether USE_DEER was used
        for training. That is deliberate: eval should always measure the
        same exact ground-truth trajectory a deployed sequential DNC would
        produce, not the (at-convergence-equivalent, but round-budget-
        limited) DEER trajectory -- see this file's v9 header note.
    """
    if num_nodes is not None and nodes_range is not None:
        raise ValueError("evaluate_traversal: pass num_nodes or nodes_range, not both")

    model.eval()
    total_triples, correct_triples = 0, 0
    perfect_episodes = 0
    tested = 0
    by_hops = {}  # path_length -> [triples_total, triples_correct, episodes, episodes_perfect]

    with torch.no_grad():
        while tested < num_episodes:
            if fixed_graph is not None:
                edges, node_labels, adjacency, n = fixed_graph
                ep = build_traversal_episode_from_graph(edges, node_labels, adjacency, n, path_length_range, rng=rng)
            else:
                n = (rng or random).randint(*nodes_range) if nodes_range is not None else num_nodes
                ep = build_traversal_episode(n, k_range, path_length_range, rng=rng)
            if ep is None:
                continue
            input_seq, target_digits, answer_mask = ep
            input_seq = input_seq.unsqueeze(0).to(device)

            hidden = (None, None, None)
            output, _ = model(input_seq, hidden, reset_experience=True)
            output = output.transpose(0, 1).contiguous().squeeze(0)  # (T, 92)
            output = output_proj_current(output)  # (T, 90) -- see run(); module set per-run

            answer_idx = (answer_mask == 1).nonzero(as_tuple=True)[0]
            episode_perfect = True
            ep_total = ep_correct = 0
            for idx in answer_idx:
                pred = decode_prediction(output[idx])
                tgt_digits = target_digits[idx].tolist()
                tgt = (int("".join(map(str, tgt_digits[0:3]))),
                       int("".join(map(str, tgt_digits[3:6]))),
                       int("".join(map(str, tgt_digits[6:9]))))
                is_correct = pred == tgt
                correct_triples += int(is_correct)
                total_triples += 1
                ep_correct += int(is_correct)
                ep_total += 1
                if not is_correct:
                    episode_perfect = False
                if tested < verbose_n:
                    print(f"  Pred: {pred} | Target: {tgt} | Correct: {is_correct}")

            perfect_episodes += int(episode_perfect)
            tested += 1

            if hop_breakdown:
                # ep_total == the episode's walk length (path_length), since
                # there's exactly one answer-required step per hop.
                hops = ep_total
                acc = by_hops.setdefault(hops, [0, 0, 0, 0])
                acc[0] += ep_total
                acc[1] += ep_correct
                acc[2] += 1
                acc[3] += int(episode_perfect)

    triple_acc = correct_triples / max(total_triples, 1) * 100
    perfect_frac = perfect_episodes / tested * 100
    print(f"Eval: triple-level acc {triple_acc:.2f}% | "
          f"perfect-traversal fraction {perfect_frac:.2f}% ({tested} episodes)")
    model.train()

    if hop_breakdown:
        breakdown = {
            hops: (
                correct / max(total, 1) * 100,          # triple_acc for this hop count
                n_perfect / max(n_eps, 1) * 100,         # perfect_frac for this hop count
                n_eps,
            )
            for hops, (total, correct, n_eps, n_perfect) in sorted(by_hops.items())
        }
        return triple_acc, perfect_frac, breakdown
    return triple_acc, perfect_frac

def build_london_underground_eval():
    """FIX (see chat log): previously called `random.seed(1234)`, which
    mutates the GLOBAL `random` module state. That was harmless while this
    function was only ever called once, at the very end of a completed run
    -- but it's a landmine for periodic mid-training OOD eval (added below,
    log addition #1): every call would reset the global RNG to the exact
    same point, making the training batches immediately following each
    periodic OOD eval identical/repeated across the whole run, silently
    degrading training-data diversity in a way invisible in the loss curve.
    Fixed by using a local `random.Random(1234)` instance so the fixed,
    reproducible OOD graph/label mapping no longer touches -- or depends on
    the call-time state of -- the global RNG stream used for curriculum
    sampling.
    """
    stations = sorted({s for s, d, _ in LONDON_UNDERGROUND_EDGES_RAW} |
                       {d for s, d, _ in LONDON_UNDERGROUND_EDGES_RAW})
    lines = sorted({l for _, _, l in LONDON_UNDERGROUND_EDGES_RAW})

    rng = random.Random(1234)  # FIX: local instance, does not touch global `random` state
    all_labels = rng.sample(range(1000), len(stations) + len(lines))
    station_to_label = dict(zip(stations, all_labels[:len(stations)]))
    line_to_label = dict(zip(lines, all_labels[len(stations):]))

    edges = [(station_to_label[s], line_to_label[l], station_to_label[d])
             for s, d, l in LONDON_UNDERGROUND_EDGES_RAW]

    adjacency = {i: [] for i in range(len(stations))}
    station_idx = {s: i for i, s in enumerate(stations)}
    for s, d, l in LONDON_UNDERGROUND_EDGES_RAW:
        adjacency[station_idx[s]].append((station_idx[d], line_to_label[l]))

    node_labels = [station_to_label[s] for s in stations]
    return edges, node_labels, adjacency


# `output_proj_current` is a module-level indirection so evaluate_traversal
# (unchanged from Phase 0) can be reused across runs without threading
# output_proj through every call. Set at the top of each run().
output_proj_current = None


# ==========================================
# 10. TRAINING LOOP -- per-beta run
#     Only new lines vs. Phase 0 are marked "# Phase 1".
# ==========================================
def run(beta_target: float, run_id: str, seed: int = SEED, resume_from: str = None,
        controller: str = CONTROLLER_TYPE,  # v7 (Alternate Phase 3, Step 1)
        chunk_size: int = CHUNK_SIZE,  # v8 (Alternate Phase 3, Step 2, Option 2)
        total_steps: int = None,  # v8: how many steps THIS run should stop at.
        use_deer: bool = USE_DEER,  # v9 (Alternate Phase 3, Step 2, Option 3)
        deer_max_newton_iters: int = DEER_MAX_NEWTON_ITERS,
        deer_tol: float = DEER_TOL,
        deer_damping: float = DEER_DAMPING,
        deer_max_jac_diag_abs: float = DEER_MAX_JAC_DIAG_ABS):
    # Deliberately independent of the LR schedule: lr_at_step()/set_lr()
    # below key off LR_DECAY_STEPS, which is derived from the module-level
    # TOTAL_STEPS constant (120000) at import time -- not from this
    # argument -- so a shorter/longer pilot run still anneals on the full
    # 120k-step cosine schedule instead of getting its own compressed one.
    # `total_steps=None` (the default, e.g. when run() is called directly
    # without the CLI) falls back to the module constant so behavior is
    # unchanged for any existing caller.
    effective_total_steps = TOTAL_STEPS if total_steps is None else total_steps
    global output_proj_current

    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = True

    os.makedirs(LOG_DIR, exist_ok=True)
    # Resume: append to the existing per-run log instead of truncating it,
    # so the loss/KL curve in the CSV stays one continuous series across
    # the original run and the resumed continuation.
    log_path = os.path.join(LOG_DIR, f"run_{run_id}.csv")
    resuming = resume_from is not None
    log_file = open(log_path, "a" if resuming else "w", newline="")
    log_writer = csv.writer(log_file)
    if not resuming:
        log_writer.writerow([
            "step", "lesson", "beta_effective",
            "task_loss", "kl_loss", "total_loss", "digit_diversity",
            "kl_mean", "kl_max", "kl_min", "kl_std", "lr",
            "grad_norm", "amp_scale", "elapsed_sec",  # log addition #2, #3, #5
            "snapshot_step",  # v5 (Phase 2): net-new, appended at the end --
            # everything above is the untouched Phase 1 schema, in the same
            # order. Which frozen (mu_g, Sigma_g) snapshot this window's
            # kl_* columns were computed against; 0 if the prior has never
            # been snapshotted yet (still N(0,I), i.e. Phase-1-equivalent).
            "newton_iters", "deer_final_max_abs_delta", "deer_converged",
            # v9 (Alternate Phase 3, Step 2, Option 3): net-new, appended at
            # the end -- everything above is the untouched v5/v6 schema, in
            # the same order. Only meaningful when use_deer=True (see #6 in
            # this file's v9 header note); left at 0 for every other
            # configuration so the schema stays identical across runs.
        ])

    # Log addition #1: periodic OOD (London Underground) eval, logged as its
    # own trajectory instead of only a single terminal number. This is what
    # lets the ID/OOD offset be read as a curve over training instead of
    # reconstructed after the fact from separate checkpoints (as it was for
    # the extended β=0 run -- see Q21-Experiment-Log.md).
    ood_log_path = os.path.join(LOG_DIR, f"run_{run_id}_ood.csv")
    ood_log_file = open(ood_log_path, "a" if resuming else "w", newline="")
    ood_log_writer = csv.writer(ood_log_file)
    if not resuming:
        ood_log_writer.writerow([
            "step", "lesson", "id_triple_acc", "id_perfect_frac",
            "ood_triple_acc", "ood_perfect_frac", "ood_offset_triple", "ood_offset_perfect",
        ])

    # Log addition #4: explicit lesson-advance event log (step at which each
    # advance happened), separate from the per-step "lesson" column, so a
    # future switch-in point can be picked from this file directly instead
    # of grepping console output for ">>> Curriculum advanced...".
    lesson_log_path = os.path.join(LOG_DIR, f"run_{run_id}_lesson_advances.csv")
    lesson_log_file = open(lesson_log_path, "a" if resuming else "w", newline="")
    lesson_log_writer = csv.writer(lesson_log_file)
    if not resuming:
        lesson_log_writer.writerow(["step", "new_lesson", "of_total_lessons"])

    # v5 (Phase 2): periodic "prior snapshot updated" log -- net-new file,
    # written every PRIOR_SNAPSHOT_EVERY steps by update_all_prior_snapshots()
    # below. This is the direct evidence for the Q48 periodic-snapshot
    # classification and for a collapse/drift check on the prior itself
    # (Sigma_g -> 0 would not show up in the existing kl_mean diagnostic,
    # which only ever describes q(v_t), never the prior it's compared
    # against). `snapshot_step` here is always == `step` (the update just
    # happened at this step); it's included anyway so this file's schema is
    # self-describing without cross-referencing the main log.
    prior_log_path = os.path.join(LOG_DIR, f"run_{run_id}_prior_snapshots.csv")
    prior_log_file = open(prior_log_path, "a" if resuming else "w", newline="")
    prior_log_writer = csv.writer(prior_log_file)
    if not resuming:
        prior_log_writer.writerow([
            "step", "snapshot_step", "mu_g_norm",
            "sigma_g_mean", "sigma_g_min", "sigma_g_max", "trace_sigma_g", "n_samples",
        ])

    curriculum = TraversalCurriculum()
    curriculum.advance_log_writer = lesson_log_writer  # log addition #4, read by maybe_advance
    curriculum.advance_log_file = lesson_log_file       # flushed after each write

    # v5 (Phase 2): dedicated RNG stream for OOD (London Underground) walk
    # sampling, decoupled from the global `random` stream that curriculum ID
    # sampling uses. Constructed once per run (not re-seeded per call --
    # that was exactly the bug already fixed for
    # build_london_underground_eval()'s station/label mapping, see the v2
    # header note above) and passed as `rng=` to every OOD evaluate_traversal()
    # call below. Seed is derived from this run's own seed so different
    # seeds in a panel still get different (but each internally
    # reproducible) OOD walk sequences, rather than all sharing one fixed
    # stream. Purely a cleanliness fix for the cross-seed comparisons Phase
    # 2 relies on -- see Section 2g in
    # Q21-Phase_1_Multi-seed_verification.md for the confound this removes.
    ood_rng = random.Random(seed * 1_000_003 + 17)

    # ---- 7. MODEL ----
    # v3: capacity scale-up -- see header note for what changed and why.
    # No longer pinned to Run 0 Run B's config; now reads the top-level
    # MODEL_* constants (CONFIGURATION section) so save_checkpoint()'s
    # recorded model_config can't desync from what's actually built here.
    hidden_size, nr_cells, cell_size, read_heads = (
        MODEL_HIDDEN_SIZE, MODEL_NR_CELLS, MODEL_CELL_SIZE, MODEL_READ_HEADS
    )

    print(f"[{run_id}] Model config: hidden={hidden_size} nr_cells={nr_cells} "
          f"cell_size={cell_size} read_heads={read_heads} | beta={beta_target} "
          f"| controller={controller} | chunk_size={chunk_size} "
          f"| use_deer={use_deer}")  # v7/v8/v9: controller + chunk_size + use_deer tags

    # v7 (Alternate Phase 3, Step 1): model construction goes through
    # MambaDNC (via ChunkedParallelDNC/DEERParallelDNC below) instead of the
    # plain DNC import. For controller='lstm' (the default, and everything
    # Phase 2 ever ran) MambaDNC.__init__ defers entirely to
    # dnc.DNC.__init__ -- see mamba_controller.py's module docstring -- so
    # every kwarg below and its meaning is byte-for-byte what Phase 2's
    # `rnn = DNC(...)` call already was. For controller='mamba',
    # MAMBA_D_STATE/MAMBA_D_CONV/MAMBA_EXPAND are also passed; MambaDNC
    # ignores them silently for the lstm path (they're simply unused kwargs
    # there), so this one call site's mamba_kwargs dict covers both classes
    # below.
    mamba_kwargs = {}
    if controller == "mamba":
        mamba_kwargs = dict(
            mamba_d_state=MAMBA_D_STATE,
            mamba_d_conv=MAMBA_D_CONV,
            mamba_expand=MAMBA_EXPAND,
        )

    # v9 (Alternate Phase 3, Step 2, Option 3): DEERParallelDNC in place of
    # ChunkedParallelDNC when use_deer=True. DEERParallelDNC subclasses
    # MambaDNC DIRECTLY (not ChunkedParallelDNC) -- per the roadmap's
    # explicit instruction not to build DEER on top of the chunked forward
    # pass, since linearizing an already-approximate chunked transition
    # would make DEER converge exactly to the chunked model's (suppressed-
    # phase-transition) behavior, defeating the point of switching to it.
    # chunk_size has no meaning here and is not passed. Every other kwarg
    # below is identical to the v8 ChunkedParallelDNC call (see that
    # branch), so a DEER run and a chunk_size=1 run construct the same
    # underlying MambaDNC configuration modulo the deer_* kwargs.
    if use_deer:
        rnn = DEERParallelDNC(
            input_size=INPUT_DIM,
            hidden_size=hidden_size,
            rnn_type=controller,  # v7: 'lstm' (default) or 'mamba'
            num_layers=1,
            nr_cells=nr_cells,
            cell_size=cell_size,
            read_heads=read_heads,
            batch_first=True,
            device=device,
            independent_linears=True,  # Phase 1: required so Memory exposes a
            # standalone write_vector_transform Linear for
            # install_stochastic_write_heads to swap out. Doesn't change the
            # addressing math -- just which code path builds the (functionally
            # equivalent) per-head transforms. See header note + Section 5 of
            # Q21-Experiment-Log.md for why this makes Run 0 an anchor-by-
            # equivalence rather than a bit-identical beta=0 substitute.
            **mamba_kwargs,
            deer_max_newton_iters=deer_max_newton_iters,  # v9
            deer_tol=deer_tol,  # v9
            deer_damping=deer_damping,  # v9
            deer_max_jac_diag_abs=deer_max_jac_diag_abs, # v9
            deer_use_analytic_diag=True,     # Option 3 -- replaces the O(D^2) jvp-loop diagonal with the closed-form one
            deer_jac_chunk_size=64,          # unused while deer_use_analytic_diag=True; kept as the fallback value
            deer_jac_sample_batch_size=2048,
            deer_step_sample_batch_size=2048 
        ).to(device)
    else:
        # v8 (Alternate Phase 3, Step 2, Option 2): ChunkedParallelDNC in
        # place of the v7 MambaDNC construction call. ChunkedParallelDNC
        # subclasses MambaDNC and passes every kwarg below straight through
        # to it unchanged (see chunked_parallel_dnc.py's __init__) -- the
        # only new argument is chunk_size itself, consumed by
        # ChunkedParallelDNC and never forwarded to MambaDNC/DNC. At
        # chunk_size=1 (the default) this is byte-for-byte equivalent to
        # the v7 `MambaDNC(...)` call it replaces (see that file's module
        # docstring, "Exactness at chunk_size=1") -- for either controller
        # type, not just 'mamba'.
        rnn = ChunkedParallelDNC(
            input_size=INPUT_DIM,
            hidden_size=hidden_size,
            rnn_type=controller,  # v7: 'lstm' (default) or 'mamba'
            num_layers=1,
            nr_cells=nr_cells,
            cell_size=cell_size,
            read_heads=read_heads,
            batch_first=True,
            device=device,
            independent_linears=True,  # Phase 1: required so Memory exposes a
            # standalone write_vector_transform Linear for
            # install_stochastic_write_heads to swap out. Doesn't change the
            # addressing math -- just which code path builds the (functionally
            # equivalent) per-head transforms. See header note + Section 5 of
            # Q21-Experiment-Log.md for why this makes Run 0 an anchor-by-
            # equivalence rather than a bit-identical beta=0 substitute.
            # v7: unaffected by the controller swap -- MambaDNC builds
            # dnc.memory.Memory identically regardless of rnn_type.
            **mamba_kwargs,
            chunk_size=chunk_size,  # v8: consumed by ChunkedParallelDNC only
        ).to(device)

    output_proj = nn.Linear(INPUT_DIM, TRIPLE_DIM).to(device)
    output_proj_current = output_proj

    # Phase 1: install the stochastic write head(s) BEFORE building the
    # optimizer, so their parameters (mu_transform, logvar_transform) are
    # included in optimizer.parameters(). mu_transform is initialized from
    # the original deterministic write_vector_transform's weights, so at
    # step 0 the sampled mean exactly matches Phase 0's write vector.
    # sample=True for all beta in BETAS_TO_SWEEP now that beta=0 is dropped
    # from the sweep (see header note) -- beta_target is always > 0.0 here.
    stochastic_heads = install_stochastic_write_heads(
        rnn, device=device, sample=(beta_target > 0.0)
    )  # Phase 1

    # Defer Adam + GradScaler until after the one-off DEER correctness gate
    # on fresh runs: optimizer state (~2x params) otherwise leaves ~16 GiB
    # GPUs with no headroom for JVP-based diag_jac during that check.
    defer_optimizer = use_deer and not resuming
    optimizer = None
    scaler = None
    amp_enabled = USE_AMP and device.type == 'cuda'

    # --- LR schedule -----------------------------------------------------
    # FIX (see chat log / Q21-Experiment-Log Section 5): CosineAnnealingLR's
    # formula -- eta_min + 0.5*(base_lr-eta_min)*(1+cos(pi*last_epoch/T_max))
    # -- is PERIODIC in last_epoch with period 2*T_max, not a one-way ramp
    # that flatlines at eta_min once last_epoch exceeds T_max. Every run
    # that trains past LR_DECAY_STEPS=20000 (i.e. every extended/switch-late
    # run in this project, since TOTAL_STEPS=80000) was therefore climbing
    # back toward LR=3e-4 after each trough at step 20000, 60000, ... and
    # peaking again at step 40000, 80000, ... -- confirmed empirically via
    # the added LR log column (observed ~2.94e-4 at step 78000, matching
    # the closed-form prediction almost exactly). The original 20,000-step
    # beta-sweep (Run 1 beta in {0.01,0.1,1.0}) is NOT affected -- it ends
    # exactly at the first trough -- but the extended beta=0 run and both
    # switch-late runs (beta=0.001, beta=0.01) trained/evaluated well past
    # that point under a partially-cycling, not monotonically-decaying, LR.
    #
    # Fix: don't use torch's stateful scheduler at all (its internal
    # last_epoch counter is also what silently carried the drift across
    # checkpoint resumes). Instead recompute LR directly from the absolute
    # step every iteration, with the step clamped at LR_DECAY_STEPS so the
    # schedule flatlines at LR_MIN once reached, exactly matching the
    # "cosine anneal from LR to LR_MIN over this many steps" comment this
    # constant always had -- it just was never enforced past that point.
    #
    # v3 addition: linear warmup 0 -> LR over the first WARMUP_STEPS,
    # before the cosine anneal (which now spans WARMUP_STEPS..LR_DECAY_STEPS
    # instead of 0..LR_DECAY_STEPS, so it still reaches LR_MIN exactly at
    # LR_DECAY_STEPS as before). Since LR is still a pure function of the
    # absolute step, this stays resume-safe for the same reason the original
    # fix was -- a resume past WARMUP_STEPS just lands in the cosine part,
    # no separate warmup state to save/restore.
    def lr_at_step(s):
        if s < WARMUP_STEPS:
            return LR * (s + 1) / WARMUP_STEPS
        s_clamped = min(s, LR_DECAY_STEPS)
        decay_span = max(1, LR_DECAY_STEPS - WARMUP_STEPS)
        progress = (s_clamped - WARMUP_STEPS) / decay_span
        return LR_MIN + 0.5 * (LR - LR_MIN) * (1 + math.cos(math.pi * progress))

    def set_lr(step_now):
        lr_now = lr_at_step(step_now)
        if optimizer is not None:
            for g in optimizer.param_groups:
                g['lr'] = lr_now
        return lr_now

    step = 0
    anneal_start_step = 0  # Phase 1: absolute step the KL anneal ramp is measured from.
    # Stays 0 for a normal fresh run or a same-beta resume (anneal counted
    # from the true start, as before). Reset to the resume step below when
    # the checkpoint's beta_target differs from this run's -- i.e. a
    # deliberate deterministic(or other-beta)->this-beta switch -- so the
    # ramp restarts at 0 from the switch point instead of reading the
    # already-large absolute step and jumping straight to full beta_target
    # on the first post-switch update.
    if resuming:
        ckpt = load_checkpoint_for_resume(resume_from, device)
        if ckpt["run_id"] != run_id or ckpt["beta_target"] != beta_target:
            print(f"WARNING: resuming {ckpt['run_id']} (beta={ckpt['beta_target']}) "
                  f"into a run configured as {run_id} (beta={beta_target}). "
                  f"Proceeding, but double-check this is the checkpoint you meant.")
        rnn.load_state_dict(ckpt["rnn_state_dict"])
        output_proj.load_state_dict(ckpt["output_proj_state_dict"])
        resume_optimizer_state = ckpt["optimizer_state_dict"]
        resume_scaler_state = ckpt.get("scaler_state_dict")
        # NOT loading scheduler_state_dict: LR is now recomputed from the
        # absolute step every iteration via set_lr() (see above), clamped
        # at LR_DECAY_STEPS -- this sidesteps the periodicity bug entirely,
        # and also means old checkpoints' scheduler_state_dict (which may
        # already reflect the buggy drifted state) is simply ignored rather
        # than needing migration.
        curriculum.lesson = ckpt["curriculum_lesson"]
        # v5 (Phase 2): restore (mu_g, Sigma_g, last_snapshot_step) for every
        # installed stochastic write head. Without this, resuming would
        # silently continue training/evaluating against whatever prior
        # snapshot happens to sit in each freshly-constructed head's
        # zero-init buffers (i.e. silently reset to N(0,I)) instead of the
        # snapshot this checkpoint was actually trained against -- the same
        # class of bug the write_mode deterministic/sampled mismatch check
        # already exists to catch in eval_from_checkpoint.py. `.get(...)`
        # instead of ["..."] so a pre-v5 (Phase 1) checkpoint, which has no
        # prior state to restore at all, still loads -- just starting the
        # learned prior fresh at N(0,I), same as a brand-new run would.
        if "prior_state" in ckpt:
            load_prior_state(stochastic_heads, ckpt["prior_state"])
        else:
            print(f"[{run_id}] WARNING: checkpoint predates the learned-prior "
                  f"snapshot (v5/Phase 2) -- prior starting fresh at N(0,I) "
                  f"instead of resuming a prior snapshot.")
        # v6: restore the dedicated OOD-sampling RNG's state, so a resumed
        # run's OOD walk sequence continues from exactly where the
        # pre-resume leg left off, rather than silently restarting from
        # ood_rng's fresh, seed-derived starting state (see save_checkpoint()
        # docstring). `.get(...)` so a pre-v6 checkpoint still loads --
        # ood_rng just starts fresh in that case, identical to today's
        # behavior before this fix.
        if "ood_rng_state" in ckpt:
            ood_rng.setstate(ckpt["ood_rng_state"])
        else:
            print(f"[{run_id}] WARNING: checkpoint predates ood_rng-state "
                  f"saving (v6 fix) -- OOD sampling restarting from its "
                  f"fresh, seed-derived state instead of continuing the "
                  f"pre-resume leg's exact walk sequence.")
        restore_rng_state(ckpt["rng_state"])
        step = ckpt["step"]
        if ckpt["beta_target"] != beta_target:
            anneal_start_step = step  # Phase 1: restart the ramp here
            print(f"[{run_id}] beta_target changed on resume "
                  f"({ckpt['beta_target']} -> {beta_target}); KL anneal "
                  f"restarted from step {anneal_start_step}, ramping over "
                  f"the next {KL_ANNEAL_STEPS} steps.")
        print(f"[{run_id}] Resumed from {resume_from} at step {step} "
              f"(lesson {curriculum.lesson + 1}/{len(curriculum.table)}); "
              f"continuing to total_steps={effective_total_steps}")
        if step >= effective_total_steps:
            print(f"[{run_id}] Checkpoint step {step} already >= total_steps "
                  f"{effective_total_steps} -- nothing to do. Pass a larger "
                  f"--total-steps if you want to extend further.")

    # v9 (Alternate Phase 3, Step 2, Option 3): one-off startup correctness
    # gate, per the roadmap's explicit instruction ("run DEER at enough
    # Newton rounds to fully converge and confirm it reproduces MambaDNC's
    # sequential rollout on a fixed batch/seed to near machine precision --
    # that's your correctness gate before trusting any speed number").
    # Skipped on resume (the check already passed on this run_id's first
    # leg, and re-running it would just cost a Newton solve for no new
    # information). Runs in eval mode so the write head's KL/prior-snapshot
    # bookkeeping (only active in training mode) is not polluted by this
    # diagnostic-only forward pass, and restores train mode immediately
    # after. Print-only: it does not abort the run even if the residual
    # looks large, since a poor result at random init (before any addressing
    # softmax has sharpened) is a real but not necessarily run-ending
    # possibility the roadmap itself flags ("a poorly-conditioned Jacobian
    # there could make Newton's method converge slowly or unstably") -- the
    # per-step diagnostics logged during training (see the training loop
    # below) are what actually tracks this as the trajectory evolves.
    if use_deer and not resuming:
        rnn.eval()
        check_model = DEERParallelDNC(
            input_size=INPUT_DIM, hidden_size=32, rnn_type=controller,
            num_layers=1, nr_cells=8, cell_size=16, read_heads=2,
            batch_first=True, device=device, independent_linears=True,
            **mamba_kwargs,
            deer_max_newton_iters=deer_max_newton_iters, deer_tol=deer_tol,
            deer_damping=deer_damping, deer_max_jac_diag_abs=deer_max_jac_diag_abs,
            deer_jac_chunk_size=256, deer_jac_sample_batch_size=1,
        ).to(device).eval()
        install_stochastic_write_heads(check_model, device=device, sample=(beta_target > 0.0))
        check_input, _, _ = sample_batch(curriculum, BATCH_SIZE)
        check_input = check_input.to(device)
        check_input = check_input[:, :5, :]
        check_result = deer_vs_sequential_max_abs_error(
            check_model, check_input, hx=(None, None, None), reset_experience=True, seed=seed,
            max_batch_size=1,
        )
        del check_model
        rnn.train()
        if check_input.is_cuda:
            torch.cuda.empty_cache()
        print(f"[{run_id}] DEER correctness check (pre-training, random init, "
              f"B=1): max_abs_error={check_result['max_abs_error']:.3e} | "
              f"newton_diagnostics={check_result['newton_diagnostics']}")

    if optimizer is None:
        optimizer = torch.optim.Adam(
            list(rnn.parameters()) + list(output_proj.parameters()),
            lr=LR,
        )
        scaler = torch.amp.GradScaler(
            'cuda', enabled=amp_enabled, init_scale=AMP_INIT_SCALE
        )
        if resuming:
            optimizer.load_state_dict(resume_optimizer_state)
            if resume_scaler_state is not None:
                scaler.load_state_dict(resume_scaler_state)
            else:
                print(f"[{run_id}] WARNING: checkpoint predates GradScaler-state "
                      f"saving (v3 fix) -- scaler starting fresh at "
                      f"init_scale={scaler.get_scale()} instead of resuming the "
                      f"prior leg's scale.")

    print(f"\n=== [{run_id}] Training (beta_target={beta_target}) "
          f"{'[resumed]' if resuming else ''} ===")
    rnn.train()
    set_lr(step)  # Phase 1 FIX: ensure correct clamped LR from the very first
    # iteration -- otherwise a resumed run's first step would briefly use
    # whatever LR was saved inside optimizer_state_dict at checkpoint time
    # (potentially still reflecting the old buggy drifted value).
    running_task_loss, running_kl_loss, running_div, running_grad_norm = 0.0, 0.0, 0, 0.0
    t0 = time.time()
    # Log addition #5: cumulative wall-clock for this run(). Starts fresh on
    # a resumed leg too (i.e. measures THIS process's elapsed time, not
    # elapsed time since the original run began across all resumed legs --
    # legs already have their own timestamped log rows if that's needed).
    t_run_start = time.time()

    while step < effective_total_steps:
        input_seq, target_digits, answer_mask = sample_batch(curriculum, BATCH_SIZE)

        input_seq = input_seq.to(device, non_blocking=True)
        target_digits = target_digits.to(device, non_blocking=True)
        answer_mask = answer_mask.to(device, non_blocking=True)
        hidden = (None, None, None)
        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast('cuda', enabled=amp_enabled):
            if use_deer:
                # v9: DEERParallelDNC's use_deer=True forward path returns
                # its output already batch-major, (B, T, INPUT_DIM) -- see
                # this file's v9 header note (#4) for why the .transpose(0,1)
                # every other branch needs here would silently swap batch
                # and time if applied to this path's output.
                output, hidden = rnn(input_seq, hidden, reset_experience=True, use_deer=True)
            else:
                output, hidden = rnn(input_seq, hidden, reset_experience=True)
                output = output.transpose(0, 1).contiguous()  # (B, T, 92)
            output = output_proj(output)  # (B, T, 90)
            task_loss = digit_loss(output, target_digits, answer_mask)

        # Phase 1: pull the KL accumulated by the stochastic write head(s)
        # during this forward pass (one term per write timestep, computed
        # from that timestep's write-head output only), apply free bits,
        # and combine with the task loss under the current annealed beta.
        # v5 (Phase 2): unchanged in meaning and unchanged here -- this is
        # still a per-write-timestep scalar and the collapse check (does it
        # flatline near the free-bits floor) is identical to Phase 1. What
        # changed is only what it's computed against: KL(q(v_t)||N(0,I)) in
        # Phase 1, KL(q(v_t)||N(mu_g,Sigma_g)) here, against whichever
        # snapshot is currently frozen in the stochastic_heads' buffers
        # (see stochastic_write_head.py v2). kl_diag now also carries
        # `snapshot_step`, logged below. v9: under use_deer=True, this pops
        # exactly one (B*T, cell_size)-shaped accumulated entry -- see
        # deer_parallel_dnc.py's _forward_deer, point 4b -- so this call and
        # its downstream free-bits/mean-over-(T*B) math are unaffected by
        # which forward path just ran.
        kl_loss, kl_diag = pop_total_kl(stochastic_heads, free_bits=FREE_BITS)  # Phase 1
        beta_eff = beta_target * min(1.0, (step - anneal_start_step) / max(1, KL_ANNEAL_STEPS))  # Phase 1: linear KL annealing 0 -> beta_target, measured from anneal_start_step
        loss = task_loss + beta_eff * kl_loss  # Phase 1: L = L_task + beta * L_KL

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        # Log addition #2: capture the pre-clip gradient norm instead of
        # discarding clip_grad_norm_'s return value. Gives a β=0 baseline
        # gradient-scale trajectory to compare a future β>0 switch-in
        # against, useful for telling a genuine instability apart from the
        # expected one-off "shock" when sampling noise is switched on.
        grad_norm = torch.nn.utils.clip_grad_norm_(rnn.parameters(), max_norm=10.0)
        scaler.step(optimizer)
        scaler.update()
        # Log addition #3: AMP loss-scale value. A collapsing/repeatedly
        # halved scale is the standard AMP symptom of inf/NaN gradients --
        # the exp()/logvar path in the stochastic write head is the most
        # numerically fragile part of this pipeline, so a clean β=0 baseline
        # of scaler behavior is useful context once β>0 introduces that path.
        amp_scale = scaler.get_scale()
        running_task_loss += task_loss.item()
        running_kl_loss += kl_loss.item()
        running_div += prediction_diversity(output.detach(), target_digits, answer_mask)
        running_grad_norm += float(grad_norm)
        step += 1
        current_lr = set_lr(step)  # Phase 1 FIX: clamped manual LR, replaces scheduler.step()

        if step % LOG_EVERY == 0:
            elapsed = time.time() - t0
            t0 = time.time()
            total_elapsed = time.time() - t_run_start  # log addition #5
            avg_task = running_task_loss / LOG_EVERY
            avg_kl = running_kl_loss / LOG_EVERY
            avg_div = running_div / LOG_EVERY
            avg_grad_norm = running_grad_norm / LOG_EVERY
            # v9: latest Newton-convergence diagnostics from this window's
            # last use_deer=True forward call (rnn._last_deer_diagnostics is
            # overwritten every such call, not accumulated -- same "current
            # value at log time" convention amp_scale already uses above,
            # rather than an average over the LOG_EVERY window). 0/False for
            # every non-DEER configuration, so the schema is identical
            # across runs.
            if use_deer and rnn._last_deer_diagnostics is not None:
                newton_iters_log = rnn._last_deer_diagnostics["newton_iters"]
                deer_max_abs_delta_log = rnn._last_deer_diagnostics["final_max_abs_delta"]
                deer_converged_log = int(rnn._last_deer_diagnostics["converged"])
            else:
                newton_iters_log = 0
                deer_max_abs_delta_log = 0.0
                deer_converged_log = 0
            print(f"[{run_id}] Step {step}/{effective_total_steps} | Lesson {curriculum.lesson + 1}/{len(curriculum.table)} "
                  f"| L_task {avg_task:.4f} | L_KL {avg_kl:.4f} | beta {beta_eff:.4f} "
                  f"| diversity {avg_div:.2f} | {LOG_EVERY / elapsed:.2f} steps/s "
                  f"| KL[mean {kl_diag['kl_mean']:.4f} max {kl_diag['kl_max']:.4f}] | LR {current_lr:.6f} "
                  f"| grad_norm {avg_grad_norm:.4f} | amp_scale {amp_scale:.1f}"
                  f"| clamp_frac {kl_diag['clamp_frac']:.4f}"
                  f"| snapshot_step {kl_diag['snapshot_step']}"  # v6: net-new, appended at the end --
                  # everything before this token is the untouched Phase 1
                  # console line, in the same order, so any existing
                  # parsing/scraping of this line by column position still
                  # works; this is purely an appended trailing field.
                  f"| newton_iters {newton_iters_log} | deer_max_abs_delta "
                  f"{deer_max_abs_delta_log:.3e} | deer_converged {deer_converged_log}")
                  # v9: net-new, appended at the very end -- same pure-append
                  # convention as every prior console-line addition.
            log_writer.writerow([
                step, curriculum.lesson + 1, beta_eff,
                avg_task, avg_kl, avg_task + beta_eff * avg_kl, avg_div,
                kl_diag["kl_mean"], kl_diag["kl_max"], kl_diag["kl_min"], kl_diag["kl_std"], 
                kl_diag["clamp_frac"], current_lr, avg_grad_norm, amp_scale, total_elapsed,
                kl_diag["snapshot_step"],  # v5 (Phase 2): net-new, appended at the end -- see header schema note
                newton_iters_log, deer_max_abs_delta_log, deer_converged_log,  # v9: net-new, appended at the end
            ])
            log_file.flush()
            running_task_loss, running_kl_loss, running_div, running_grad_norm = 0.0, 0.0, 0, 0.0

        if step % PRIOR_SNAPSHOT_EVERY == 0:
            # v5 (Phase 2): refit (mu_g, Sigma_g) from the writes
            # accumulated since the last snapshot, freeze the result into
            # every stochastic write head's buffers, and log a summary.
            # This call -- and only this call -- is where the prior's
            # non-locality lives (see stochastic_write_head.py v2,
            # update_prior_snapshot); it happens here, outside the
            # per-step backward pass above, never inside it. This is a
            # brand-new console line and a brand-new log file -- it does
            # not touch the per-step console line above, or any other
            # existing print/log call in this loop.
            snap_diag = update_all_prior_snapshots(
                stochastic_heads, step,
                min_logvar=PRIOR_MIN_LOGVAR, max_logvar=PRIOR_MAX_LOGVAR,
            )
            print(f"[{run_id}] Step {step} prior snapshot updated | "
                  f"||mu_g|| {snap_diag['mu_g_norm']:.4f} | "
                  f"diag(Sigma_g) mean/min/max {snap_diag['sigma_g_mean']:.4f}/"
                  f"{snap_diag['sigma_g_min']:.4f}/{snap_diag['sigma_g_max']:.4f} | "
                  f"trace(Sigma_g) {snap_diag['trace_sigma_g']:.4f} | "
                  f"n_samples {snap_diag['n_samples']}")
            prior_log_writer.writerow([
                step, snap_diag["snapshot_step"], snap_diag["mu_g_norm"],
                snap_diag["sigma_g_mean"], snap_diag["sigma_g_min"], snap_diag["sigma_g_max"],
                snap_diag["trace_sigma_g"], snap_diag["n_samples"],
            ])
            prior_log_file.flush()

        if step % EVAL_EVERY == 0:
            _, id_triple_acc, id_perfect_frac = curriculum.maybe_advance(rnn, device, step=step)

            # Log addition #1: periodic OOD (London Underground) eval, using
            # the RNG-safe build_london_underground_eval() (see fix above --
            # local random.Random(1234), doesn't reset the global training
            # RNG stream). This is what turns the ID/OOD offset into a
            # trajectory over the whole run instead of one terminal number.
            # v5 (Phase 2): rng=ood_rng -- decouples this call's walk
            # sampling (path length / start node / edge choices over the
            # fixed graph) from the global `random` stream curriculum ID
            # sampling uses. See v5 header note.
            edges, node_labels, adjacency = build_london_underground_eval()
            ood_triple_acc, ood_perfect_frac = evaluate_traversal(
                rnn, device, num_episodes=OOD_EVAL_EPISODES_PERIODIC, verbose_n=0,
                fixed_graph=(edges, node_labels, adjacency, len(node_labels)),
                path_length_range=OOD_PATH_LENGTH_RANGE,
                rng=ood_rng,
            )
            ood_log_writer.writerow([
                step, curriculum.lesson + 1,
                id_triple_acc, id_perfect_frac,
                ood_triple_acc, ood_perfect_frac,
                id_triple_acc - ood_triple_acc, id_perfect_frac - ood_perfect_frac,
            ])
            ood_log_file.flush()
            print(f"[{run_id}] Step {step} periodic OOD check: "
                  f"ID {id_triple_acc:.2f}% | OOD {ood_triple_acc:.2f}% | "
                  f"offset {id_triple_acc - ood_triple_acc:.2f}")

        if step % CHECKPOINT_EVERY == 0 or step == effective_total_steps:
            ckpt_path = os.path.join(CHECKPOINT_DIR, f"{run_id}_step{step}.pt")
            save_checkpoint(ckpt_path, rnn, output_proj, stochastic_heads,
                             optimizer, curriculum, step,
                             beta_target, run_id, scaler, ood_rng,  # v6: + ood_rng
                             controller_type=controller,  # v7
                             chunk_size=chunk_size,  # v8
                             use_deer=use_deer,  # v9
                             deer_max_newton_iters=deer_max_newton_iters,
                             deer_tol=deer_tol,
                             deer_damping=deer_damping,
                             deer_max_jac_diag_abs=deer_max_jac_diag_abs)
            # Also keep a stable "latest" pointer so eval/resume don't need
            # to know the exact final step number in advance. Plain file
            # copy -- NOT a torch.load()+torch.save() round-trip. The
            # checkpoint contains non-tensor objects (python/numpy RNG
            # state), and PyTorch >=2.6 defaults torch.load to
            # weights_only=True, which refuses to unpickle those unless you
            # explicitly opt in (see load_checkpoint_for_resume() above).
            latest_path = os.path.join(CHECKPOINT_DIR, f"{run_id}_latest.pt")
            shutil.copyfile(ckpt_path, latest_path)

    print(f"\n[{run_id}] Training complete. Final evaluation on training-distribution lesson:")
    _, id_triple_acc, id_perfect_frac = curriculum.maybe_advance(rnn, device, step=step)

    print(f"\n[{run_id}] Generalization test: London Underground (held out, never trained on):")
    # v5 (Phase 2): rng=ood_rng -- same dedicated-stream fix as the periodic
    # OOD check above, applied to the terminal eval too, so the whole run's
    # OOD sampling (periodic and final) is decoupled from the ID/curriculum
    # stream consistently.
    edges, node_labels, adjacency = build_london_underground_eval()
    ood_triple_acc, ood_perfect_frac = evaluate_traversal(
        rnn, device, num_episodes=OOD_EVAL_EPISODES, verbose_n=10,
        fixed_graph=(edges, node_labels, adjacency, len(node_labels)),
        path_length_range=OOD_PATH_LENGTH_RANGE,
        rng=ood_rng,
    )

    ood_offset_triple = id_triple_acc - ood_triple_acc
    ood_offset_perfect = id_perfect_frac - ood_perfect_frac

    # Log addition #1: also record this full-episode-count terminal read in
    # the OOD trajectory file (tagged step=step, same schema as the periodic
    # rows) so the trajectory file is self-contained -- no need to cross-
    # reference the summary block in the main log for the final point.
    ood_log_writer.writerow([
        step, curriculum.lesson + 1,
        id_triple_acc, id_perfect_frac,
        ood_triple_acc, ood_perfect_frac,
        ood_offset_triple, ood_offset_perfect,
    ])
    ood_log_file.flush()

    summary = {
        "run_id": run_id,
        "beta_target": beta_target,
        "id_triple_acc": id_triple_acc,
        "id_perfect_frac": id_perfect_frac,
        "ood_triple_acc": ood_triple_acc,
        "ood_perfect_frac": ood_perfect_frac,
        "ood_offset_triple": ood_offset_triple,
        "ood_offset_perfect": ood_offset_perfect,
        "total_elapsed_sec": time.time() - t_run_start,  # log addition #5
    }
    print(f"\n[{run_id}] SUMMARY: {summary}")

    log_writer.writerow([])
    log_writer.writerow(["SUMMARY"] + list(summary.keys()))
    log_writer.writerow([""] + list(summary.values()))
    log_file.close()
    ood_log_file.close()
    lesson_log_file.close()
    prior_log_file.close()  # v5 (Phase 2)

    return summary


# ==========================================
# 11. GATE 1 SWEEP
# ==========================================
if __name__ == "__main__":
    import sys
    import argparse

    parser = argparse.ArgumentParser(
        description="Phase 2 KL sweep (learned, periodically-snapshotted "
                    "prior). Positional beta for single-beta mode; "
                    "no beta runs the full BETAS_TO_SWEEP list sequentially."
    )
    parser.add_argument("beta", nargs="?", type=float, default=None,
                         help=f"beta to run, one of {BETAS_TO_SWEEP}. Omit to run the full sweep sequentially.")
    parser.add_argument("--resume", type=str, default=None,
                         help="path to a checkpoint .pt to resume from (requires --beta / positional beta "
                              "to also be given, so run_id/beta_target can be matched against the checkpoint). "
                              "Training continues from the checkpoint's step up to TOTAL_STEPS.")
    parser.add_argument("--run-id-suffix", type=str, default=None,
                         help="appended to the derived run_id (e.g. 'switch') so this run's checkpoints "
                              "(phase1_checkpoints/beta_XXX<suffix>_stepN.pt) and log "
                              "(phase1_logs/run_beta_XXX<suffix>.csv) don't collide with an existing run "
                              "that used the same beta value -- e.g. resuming a beta=0 checkpoint into a "
                              "beta=0.01 run would otherwise reuse the same run_id/files as a prior plain "
                              "beta=0.01 sweep run.")
    parser.add_argument("--seed", type=int, default=SEED,
                     help="random seed for torch/random/numpy (default: SEED module constant)")
    parser.add_argument("--controller", type=str, default=CONTROLLER_TYPE,
                         choices=["lstm", "mamba"],
                         help="v7 (Alternate Phase 3, Step 1): DNC controller type. "
                              "'lstm' (default) reproduces Phase 2 exactly. "
                              "'mamba' swaps in the Mamba-1 controller from "
                              "mamba_controller.py; requires the mamba-ssm package.")
    parser.add_argument("--chunk-size", type=int, default=CHUNK_SIZE,
                         help="v8 (Alternate Phase 3, Step 2, Option 2): number of real "
                              "timesteps processed per controller call via "
                              "ChunkedParallelDNC (chunked_parallel_dnc.py). "
                              "1 (default) reproduces the exact sequential "
                              "Step-1/Phase-2 forward pass byte-for-byte; "
                              ">1 trades a tunable within-chunk stale-read "
                              "approximation for real parallelism. See that "
                              "file's module docstring for the full argument. "
                              "Ignored when --use-deer is passed.")
    parser.add_argument("--total-steps", type=int, default=None,
                         help="v7 (Phase 3): override the training loop's stopping point for a "
                              "cheap pilot run (e.g. the seed-0/seed-2, short-budget staging pass "
                              "suggested before committing to the full 4-seed x 3-config x "
                              f"{TOTAL_STEPS}-step sweep). Defaults to the module constant "
                              f"TOTAL_STEPS ({TOTAL_STEPS}) if omitted. Does NOT change "
                              "LR_DECAY_STEPS -- a pilot run still anneals on the full schedule.")
    parser.add_argument("--use-deer", action="store_true",
                         help="v9 (Alternate Phase 3, Step 2, Option 3): train through "
                              "DEERParallelDNC (deer_parallel_dnc.py) instead of "
                              "ChunkedParallelDNC -- quasi-Newton parallel evaluation of the "
                              "TRUE, un-approximated per-timestep recurrence, in place of "
                              "Option 2's chunked/frozen-read approximation. Off by default. "
                              "--chunk-size is ignored when this is set.")
    parser.add_argument("--deer-max-newton-iters", type=int, default=DEER_MAX_NEWTON_ITERS,
                         help="v9: hard cap on Newton rounds per forward call, threaded "
                              "through to deer_quasi_newton_solve. Only meaningful with "
                              "--use-deer.")
    parser.add_argument("--deer-tol", type=float, default=DEER_TOL,
                         help="v9: convergence threshold on max(abs(y_new - y_old)). "
                              "Default (None) auto-selects 1e-4 (float32) / 1e-7 (float64) "
                              "inside deer_quasi_newton_solve, matching Lim et al.'s reference "
                              "implementation. Only meaningful with --use-deer.")
    parser.add_argument("--deer-damping", type=float, default=DEER_DAMPING,
                         help="v9: scale-ELK style shrink factor in [0, 1) applied to the "
                              "diagonal Jacobian used in the Newton linear solve (numerical "
                              "safeguard against DNC's addressing-softmax ill-conditioning near "
                              "flat regions -- see newton_associative_scan.py's module "
                              "docstring). 0.0 (default) = plain, undamped quasi-DEER. Only "
                              "meaningful with --use-deer.")
    parser.add_argument("--deer-max-jac-diag-abs", type=float, default=DEER_MAX_JAC_DIAG_ABS,
                         help="v9: optional hard clamp on |diagonal Jacobian entries| before "
                              "they enter the Newton linear solve -- an additional numerical "
                              "safety valve beyond --deer-damping. Default (None) = no clamp. "
                              "Only meaningful with --use-deer.")
    args = parser.parse_args()

    if args.chunk_size < 1:
        raise SystemExit(f"--chunk-size must be a positive int, got {args.chunk_size}")

    if args.use_deer and args.chunk_size != 1:
        print(f"WARNING: --use-deer is set; --chunk-size={args.chunk_size} has no effect "
              f"(DEERParallelDNC subclasses MambaDNC directly, not ChunkedParallelDNC -- "
              f"see this file's v9 header note).")

    if args.resume is not None and args.beta is None:
        raise SystemExit("--resume requires the beta positional arg too, e.g.:\n"
                          f"  python3 {sys.argv[0]} 0.02 --resume phase1_checkpoints/beta_0p02_latest.pt")

    if args.beta is not None:
        betas_to_run = [args.beta]
        if args.beta not in BETAS_TO_SWEEP:
            print(f"WARNING: {args.beta} is not in the planned sweep "
                  f"{BETAS_TO_SWEEP} -- running it anyway, but it won't be "
                  f"picked up by the combined summary step below.")
    else:
        betas_to_run = BETAS_TO_SWEEP

    all_summaries = []
    for beta in betas_to_run:
        # v5 (Phase 2): run_id now includes seed + a "_learnedprior" tag by
        # default, rather than reusing Phase 1's bare f"beta_{beta}" naming
        # -- e.g. "beta_0p001_seed0_learnedprior". This is deliberate, not
        # cosmetic: Phase 1's analysis scripts match on
        # "Training (beta_target=...)" / "beta_0p0_seed{N}"-style tags, and
        # without a distinct tag here, Phase 2 runs would get silently
        # pooled with Phase 1 runs by any grep-based aggregation over the
        # log directory. --run-id-suffix still applies on top of this, for
        # ad-hoc disambiguation beyond the seed/learned-prior tag.
        run_id = f"beta_{beta}_seed{args.seed}_learnedprior".replace(".", "p")
        if args.controller == "mamba":  # v7: keep lstm/mamba runs from colliding in phase1_logs/
            run_id = f"{run_id}_mambactrl"
        if args.use_deer:  # v9: DEER runs never pool with chunked/sequential runs
            run_id = f"{run_id}_deer"
        elif args.chunk_size > 1:  # v8: chunk_size=1 is behaviorally identical to
            # Step 1, so it deliberately gets no tag and keeps pooling with
            # existing Step-1 runs -- see header note. Mutually exclusive
            # with the _deer tag above, since --chunk-size is ignored under
            # --use-deer (see the warning printed above).
            run_id = f"{run_id}_chunk{args.chunk_size}"
        if args.run_id_suffix:
            run_id = f"{run_id}_{args.run_id_suffix}"
        summary = run(beta_target=beta, run_id=run_id, seed=args.seed,
                       resume_from=args.resume, controller=args.controller,  # v7
                       chunk_size=args.chunk_size, total_steps=args.total_steps,  # v8
                       use_deer=args.use_deer,  # v9
                       deer_max_newton_iters=args.deer_max_newton_iters,
                       deer_tol=args.deer_tol,
                       deer_damping=args.deer_damping,
                       deer_max_jac_diag_abs=args.deer_max_jac_diag_abs)
        all_summaries.append(summary)

    print("\n===== Sweep summary (this process) =====")
    for s in all_summaries:
        print(s)

    # Per-process partial summary -- when running one beta per process, each
    # writes its own file; merge them by hand (or a short follow-up script)
    # once all processes finish. Only the no-arg (full sequential) path
    # produces a complete combined summary in one file.
    summary_suffix = "_".join(f"b{b}".replace(".", "p") for b in betas_to_run)
    summary_path = os.path.join(LOG_DIR, f"gate1_sweep_summary_{summary_suffix}.csv")
    with open(summary_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(all_summaries[0].keys()))
        w.writeheader()
        for s in all_summaries:
            w.writerow(s)
    print(f"\nWrote summary to {summary_path}")
