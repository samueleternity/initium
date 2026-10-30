# Workspace Capability Task Family

This package provides a synthetic, oracle-backed benchmark for testing what
the split-graph workspace can retain, update, retrieve, and combine. It is
designed to separate memory capability from natural-language comprehension:
episodes are typed event streams, answers are categorical tokens, and the
Python generator maintains the exact symbolic state used to produce labels.

## What the benchmark compares

The intended model is a split-graph Mamba-1 controller with MoE. Run the same
task configuration with four resource arms:

| Arm | Sequence memory | Workspace | What it isolates |
|---|---|---|---|
| Backbone/MoE control | none | off | Performance without either persistent memory mechanism |
| Workspace only | none | on | What the learned slots add to the same backbone |
| KV only | causal KV | off | What the token-addressable sequence memory adds |
| KV + workspace | causal KV | on | Whether the two memories complement one another |

For the interaction, compare `(KV + workspace - KV only) - (workspace only -
backbone control)`. Keep backbone depth, MoE settings, training steps, optimizer,
and seed fixed across a matched panel. Workspace-enabled runs need MoE and use
the project's existing workspace gate, reconsideration, prediction-error
salience, and slot-count options. Treat workspace variants as separate
experiments; do not aggregate them into one “workspace” score.

## Task tracks

Each track has its own generator and symbolic oracle in `tracks/`. Select one
track for diagnosis or use `all` for a balanced multi-track curriculum.

### `permutation` — sequential state composition

The stream initializes objects in positions and then applies pairwise swaps.
The query asks either where an object ended up or which object occupies a
position. The oracle applies every swap in order; distractor swaps avoid the
queried object/occupant. This tests maintaining and composing state across a
long sequence, and separates relevant swaps from unrelated operations. The
curriculum scales object count and swap depth. Entity IDs `0–47` are reserved
for ID episodes and `48–63` for held-out label evaluation, so those sets do not
overlap. The evaluator reports four matched conditions: ID baseline,
label-only (same lesson difficulty and sequence structure, held-out IDs),
length-only (same entity count and ID labels, more swaps/distractors), and
combined. This keeps label generalization separate from sequence-length
difficulty. The original max-difficulty OOD result remains a combined stress
test for continuity with other tracks.

For a binding diagnostic, combine `--workspace-entity-keyed-writes` with
`--workspace-semantic-entity-addressing`. The latter uses a fixed one-hot key
for each entity token and writes initialization only after its location token.
Swap operands first appear in a read-only event prefix. An `EV_COMMIT` marker
then starts a second, write-enabled pair of entity tokens, so both operands are
known before either entity slot is updated. The two addressed updates remain
sequential within the recurrent workspace step; this staging does not claim a
hardware-atomic multi-slot write. Query entity tokens can read their slot but
never write. Stored values still come from selected learned specialist
payloads, not from injected oracle state. On slot exhaustion, a new semantic
write is skipped and counted as address overflow rather than overwriting
another entity's binding. Keep at least as many slots as the lesson's entity
count. This mode supports `--workspace-compile-recurrence` and requires a
model width of at least 64 for the fixed entity-ID key basis.

Semantic entity keys identify addresses; the first occurrence of each ID is
assigned to the next free slot. Slot numbers are therefore episode-local and
should not be compared across episodes. The binding report checks every active
MoE write route, reports within-episode entity fragmentation and
different-entity slot collisions, and flags when routes for one token disagree.
For end-to-end state verification, add
`--workspace-semantic-state-diagnostics`. At periodic and final evaluations it
replays one episode prefix after selected initialization/swap events, queries
each currently initialized entity, and logs the accepted slot, its value RMS,
the oracle position, and the model's readback prediction alongside a matched
workspace-readback-bypassed prediction. Per-event accuracy is written to the
workspace task metrics CSV; entity-level slot and position details go to the
console. The event limit is
controlled by `--workspace-state-diagnostic-max-events` (default 16; use 0 to
include every transition). This is evaluation-only and can add noticeable
runtime, especially on later lessons with many swaps.

To directly train the stored payloads to represent the permutation state, add
`--workspace-state-loss-weight WEIGHT` (for example `0.1`). At every
initialization and swap boundary, the auxiliary head reads each initialized
entity through its semantic address and predicts its oracle location. Only the
loss target uses the episode's oracle snapshot; the model input and workspace
payload remain the normal causal token stream and learned specialist values.
The console reports `L_ws_state`, accuracy over supervised entity states, and
the task loss separately. This option requires semantic entity addressing and
the `permutation` or `all` track. A fresh run is recommended because the
auxiliary decoder is a trainable model component.

Semantic-address mode also bounds each stored content vector to RMS 1.0 and
clips each readback vector to at most the corresponding routed writer's RMS.
This prevents an unconstrained learned value projection from making workspace
readback grow larger than the specialist signal. The console identifies this
as `semantic_state_bounds`; `slot_rms` and `broadcast/specialist` telemetry
show whether the bounds are active. This affects only semantic entity-address
mode, leaving ordinary shared and learned-address workspace runs unchanged.

For an explicitly swap-aware ablation, add
`--workspace-permutation-transactional`. This is a separate update rule layered
on semantic entity addressing: the two swap commit tokens are read-only, then
at the event boundary both entity slots are resolved from the same pre-commit
address state and their content payloads are exchanged together. It requires
`--workspace`, `--workspace-entity-keyed-writes`,
`--workspace-semantic-entity-addressing`, at least 16 slots, and the
`permutation` or `all` track. The state decoder is also used as the task
readout: object-to-position queries score the requested entity's slot value;
position-to-object queries compare each occupied slot's decoded position and
score the corresponding entity address. Its class logits replace the normal
task logits only at the answer token, so the task loss trains the same decoder
that is evaluated. It supports `--workspace-compile-recurrence`; pair commits
are applied at event boundaries around the compiled per-token recurrence.

Transactional evaluations write separate `permutation_query/` rows for the
two query directions and `permutation_state/swap_depth_N` rows in the workspace
task-metrics CSV. State accuracy is measured on held-out episodes after each
initialization/swap snapshot, grouped by the number of swaps committed so far.
The `permutation_pair_commit` row records how many requested exchanges found
both distinct entity slots and were actually applied.
Use a fresh run because this mode changes state dynamics and makes the task
decoder part of the answer path.

Example permutation run:

```bash
python -m src.initium.core_training 0.0 --dataset-type workspace-capability --workspace-task-track permutation --split-graph --split-graph-variant mamba1 --split-graph-num-blocks 4 --split-graph-backbone-only --moe --moe-placement all --moe-backbone-residual-scale 0.1 --workspace --workspace-num-slots 16 --workspace-entity-keyed-writes --workspace-semantic-entity-addressing --workspace-state-loss-weight 0.1 --workspace-semantic-state-diagnostics --workspace-state-diagnostic-max-events 16 --workspace-compile-recurrence --seed 0 --total-steps 50000 --run-id-suffix wcap-permutation-semantic-state-loss-s0 --checkpoint-every 5000
```

### `entity_state` — source priority, corrections, and retractions

Facts assign a relation value to an entity under a source. Sources have a
documented priority (`source_id % 3`); higher priority wins, then the most
recent active fact wins, then lower source ID breaks a remaining tie. The event
stream includes assertions, moves, retractions, removals, and supersessions.
The query asks for the winning current value or `NONE`. High-surprise facts on
other relations are irrelevant distractors. This tests state replacement,
source conflict resolution, correction handling, and whether update signals
favor useful corrections over novelty. Per-episode metadata flags relevant
corrections, novelty distractors, and equal-priority/equal-time conflicts.

### `relational_join` — multi-hop fact composition

Source-tagged facts form a one-, two-, or three-hop chain between entities.
The query supplies the start entity and relation sequence, and asks for the
terminal entity. Interleaved near-miss facts use unrelated relations and
cannot create an alternate valid path. This tests retaining separate facts
and composing them at query time, rather than retrieving one matching token.
OOD examples use held-out entity/relation bands and more distractors.

### `nth_farthest` — relational comparison across remembered items

The stream presents quantized vectors for candidate entities and a query
vector. The model must return the identity of the Nth-farthest candidate.
Nearest and farthest controls are mixed into the track and reported as separate
query types. This is a positive control for slot interaction and comparison:
the answer depends on comparing multiple remembered items to one another and
to the query vector.

### `retrieval_aggregation` — delayed recall and aggregation

The stream writes key/value pairs, updates selected keys, adds unrelated
key/value distractors, and asks for one or more current values or an
aggregation. Supported operations are exact recall, sum modulo 16, count of
even values, and parity. Multi-query episodes place all requested keys at the
end. This tests associative retrieval, update precedence, multi-item retrieval,
and simple aggregation as context and interference grow.

## Curriculum, evaluation, and metrics

Each track has six difficulty levels. `all` rotates a balanced set of tracks
through each batch and advances only after the overall answer-accuracy
threshold and every track's threshold are met. The dataset can also be run
with `--workspace-task-track <track>` to study one skill without dilution from
the others. Evaluation uses locally seeded ID and OOD episode generators so
the episode set is reproducible for paired model conditions.

Permutation generalization is additionally logged at periodic and final
evaluation in `workspace_task_metrics.csv`, using `permutation/<condition>`
track names. KV runs include matched memory-on and memory-off rows for all four
conditions. The label-only and ID episodes are generated from the same paired
structural seed; only the entity-label band changes. Length-only preserves the
lesson's entity count and label band while increasing temporal interference.

The trainer reports answer-token accuracy and episode exact match, overall and
per track. `./logs/run_<id>_workspace_task_metrics.csv` also records context
length, relevant updates, distractors, correction/novelty fractions, source
count, random candidate baseline, and priority-tie rate. The dataset-owned
`field_breakdown.csv` stratifies query type, context length, relevant-update
count, hop/source count, correction and novelty flags, priority conflicts,
distractor count, and candidate count.

Workspace-enabled runs additionally write
`workspace_dependency.csv` (or `_v2.csv` when resuming a legacy schema). Its
matched ID episodes are scored with full readback, readback bypassed, and
readback shuffled across batch items. The shuffle changes only whose slot
state is read; slot writes and state updates stay with their original examples.
Existing workspace slot/gate/router telemetry is still emitted by the shared
trainer. KV runs additionally retain the existing memory usage diagnostics.

Candidate-baseline percentages are simple random-choice references, not
calibrated statistical baselines. Use answer accuracy together with episode
exact match and the task strata. Memory-off alone does not establish that
memory contents were correct; compare it with the workspace readback
counterfactuals and, for KV, the existing KV dependency/shuffle diagnostics.
The core suite is symbolic. Natural-language transfer is deliberately deferred
until the symbolic controls are validated.

## Example Colab commands

Use the same backbone/MoE configuration and seed when comparing arms. These
examples use all task tracks and seed 0; remove `--workspace` for a no-workspace
arm. Set `--memory-type none` for the no-KV arm or `kv` for causal KV.

```bash
!python -m src.initium.core_training 0.0 --dataset-type workspace-capability --workspace-task-track all --split-graph --split-graph-variant mamba1 --split-graph-num-blocks 4 --memory-type none --moe --moe-placement all --moe-num-experts 8 --moe-top-k 1 --moe-capacity-factor 1.5 --moe-backbone-residual-scale 0.1 --seed 0 --total-steps 50000 --run-id-suffix wcap-workspace-none-s0 --checkpoint-every 10000 --workspace --workspace-num-slots 5 --workspace-train-mode coadapted
```

For KV, change `--memory-type none` to `--memory-type kv` and add
`--memory-heads 4 --memory-dk 64 --memory-dv 64 --memory-read-lag 1
--memory-inter-layer-read-every-block`. Keep `--moe-backbone-residual-scale
0.1` in workspace arms. Workspace reconsideration and prediction-error options
can be added as controlled variants using the core trainer's CLI help.

## Package map

- `episode.py`: token vocabulary, episode record, and boundary validation.
- `tracks/*.py`: one task generator/oracle per benchmark track.
- `curriculum.py`: per-track difficulty schedules and progression policy.
- `dataset.py`: `BaseDataset` adapter, batching, evaluation, ablations, and task
  metrics.
- `../../dataset_registry.py`: registration as `workspace-capability`.
- `../../core_training.py`: model compatibility checks and run-level CSV output.

Unit and integration coverage for the package lives in
`tests/unit/test_workspace_capability.py` and
`tests/integration/test_workspace_capability_pipeline.py`.

Run the focused checks from the repository root with:

```bash
python -m pytest tests/unit/test_workspace_capability.py tests/integration/test_workspace_capability_pipeline.py -q
```
