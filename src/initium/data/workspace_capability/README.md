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

For an input-path performance comparison, add `--workspace-token-id-input` to
the compact-input arm while keeping every other option and seed fixed. It stores
each generated stream as integer token IDs instead of a `(batch,time,320)`
float32 one-hot tensor. The split-graph input Linear uses the same learned
weights as an embedding lookup, including its bias and zero-row padding
semantics. Targets and auxiliary prediction objectives keep their existing
definitions. The default remains one-hot input, and the run ID gets the
`_tokenidinput` tag.

The compact input representation does not reduce the model's saved
back-propagation activations. If a longer lesson still exceeds GPU memory, the
training CLI also provides `--activation-offload-to-cpu`. This stores autograd
saved tensors in pinned host memory and restores them during backward, retaining
full BPTT and the same forward equations at the cost of host RAM and transfer
time. `--activation-offload-cpu-budget-mb` limits copied activation payload per
step (default 2048 MiB); saves over that cap stay on their original device. Add
both flags to matched input arms when measuring the offload effect. Console
telemetry reports per-step copied/resident saved-tensor bytes and, on Linux,
process RSS, peak RSS, and pinned RAM. Run IDs include the configured budget.

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

On `--workspace-task-track all`, the same fixed entity keys are populated for
every track at causal event boundaries: entity-state writes after an update,
relational-join writes the subject edge when its object arrives and the object
context at the fact boundary, nth-farthest writes after a complete vector, and
retrieval-aggregation writes after each key/value payload. These event writes
do not use answer labels. The all-track generator can expose all 64 entity IDs
in one retrieval episode, so semantic-address mode requires
`--workspace-num-slots 64` to avoid dropping writes on capacity overflow.
Permutation-specific state loss and pair commits remain active only on
permutation examples in the mixed track.

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
the `permutation` or `all` track. In an `all`-track run it applies only to
permutation examples; other tracks have no entity-to-location oracle target and
continue to train through their task loss. A fresh run is recommended because
the auxiliary decoder is a trainable model component.

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
In an `all`-track run, this update rule applies only to permutation examples;
the remaining four tracks retain their regular task output and use causal
event-boundary semantic writes.

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

### `capacity_bottleneck` — set size, tracking load, and n-back controls

Balanced subconditions cover symbolic change detection with set size `k`,
symbolic identity tracking with `k` marked objects and moving distractors, and
n-back lag. Change detection now interleaves non-probe updates before revealing
only the probed item's current value; tracking adds more targets, frames, and
moving non-targets. N-back streams grow independently of `n`. Each generator
records its subtask and `k` or lag, so results can be inspected by condition.
The tracking stream uses explicit symbolic object IDs, so it is an event-based
tracking analogue rather than a visual MOT replication.

### `belief_tracking` — agent-specific state and reality controls

The generator presents an initial world state, subsequent object moves, and
which agents observed each move. It balances false-belief queries (the queried
agent misses the final move), true-belief queries (the agent observes it), and
current-world controls. The curriculum adds more agents, objects, and repeated
moves, with at most ten moves per episode to bound sequence activation use.
Each agent receives a partial observation history; the final query is
balanced across false belief, true belief, and reality. This is a typed
symbolic task informed by ToM-bAbI/ToMi, not a reproduction of either
natural-language benchmark.

### `mixed_skill` — composed lookup, arithmetic, copy, and sort

An episode provides a key/value table followed by a dependent pipeline. The
initial lookup feeds arithmetic, copy, and sort stages; later operations reuse
prior stage results. The answer concatenates all intermediate outputs in order,
including both values emitted by sort. Results are absent from the input;
workspace address metadata marks table keys and pipeline stages only. The
curriculum grows from three to eight operations while keeping one compact
symbolic event per operation.

### `rule_switching` — card-feature feedback and task-set revision

Evidence trials show cards, a simulated feature choice, and noisy feedback.
All card features share one value-token family, so token type does not reveal
the active feature. The number of switches and feedback noise rise by lesson;
some switches are cued while others must be inferred from evidence. The final
query asks for the active feature value on a novel card. This is a compact
symbolic task, not a full human WCST protocol.

### `attentional_blink` — RSVP lag and symbolic mask conditions

Each item is an RSVP frame with two cued features and a scalar value. Targets
match the conjunction of both cue features; hard distractors often match one
feature, so neither parity nor a single-feature filter identifies them. A mask
condition adds a post-T2 noise-mask frame; intervening frames remain ordinary
distractors at every lag.
The generator does not expose target positions through semantic workspace
addresses or output-key labels. T2 lag and mask condition are reported
separately, including T1/T2 accuracy by lag and mask. Difficulty increases
the number of distractors, partial-match lure rate, and maximum target lag.
This remains a symbolic RSVP analogue rather than a visual display or a unique
test of global-workspace ignition. Curriculum advancement requires the
exact-match threshold at every observed lag/mask condition, not only on an
aggregate short/long-lag group. Both masked and unmasked conditions are
balanced at every lag.

### `variable_binding` — sequential assignment and copy-by-value

Variables receive scalar values, are reassigned, copied by value, or assigned a
modular sum of two current variables. The queried result follows multiple
copies and reassignments and is computed, not present as a scalar literal in
the input. Later lessons add variables and operations. The generated chain
records its depth and exact-match results are grouped by chain difficulty.

Each family is selectable directly through `--workspace-task-track` and is
included in `all`. Episodes for these six families are kept shorter than
the existing long permutation track, so they do not raise its full-BPTT
sequence-length peak. The existing lesson-5 activation-memory limit remains;
compact token-ID inputs reduce batch storage but do not remove that limit.

Use `--workspace-task-track new_all` to train on these six capability families together
without mixing in the original five tracks. In `new_all`, semantic entity
addressing needs at least 48 slots for the maximum set of world-state and
agent-belief keys; 48 slots are sufficient, while `all` still requires 64 for
retrieval aggregation.

## GNW functional probes

Seven additional tracks translate the consciousness-paper list into bounded
symbolic tasks: `ignition`, `psychological_refractory`, `masked_priming`,
`sustained_maintenance`, `no_report_confidence`, `bistable_perception`, and
`trace_conditioning`. The existing `attentional_blink` RSVP track is reused as
the eighth probe. Select any one with `--workspace-task-track <track>` or use
`--workspace-task-track gnw_all` to train only on this eight-track probe cohort.
The `all` mix contains all eighteen tracks; `new_all` remains the separate
six-track capability cohort.

The probes expose condition-stratified results through the existing
`query_types` metrics. Ignition varies repeated evidence strength; the RSVP
track varies lag and masking; the refractory task varies stimulus-onset gap;
masked priming crosses visibility and congruency; maintenance varies distractor
delay; no-report/confidence pairs an action response with an optional coarse
confidence response; rivalry uses a two-state hysteresis process; trace
conditioning varies the gap in a cue/outcome association. Track markers for
the seven appended tracks use an in-vocabulary extension code, so the model
input/output projection width stays at the existing 320-token size.

These are symbolic functional analogues, not human psychophysics. In
particular, the refractory task currently reports accuracy by onset gap rather
than reaction-time distributions; the no-report confidence label is a coarse
evidence-margin target and does not yet provide Brier score or type-2 ROC
analysis. Ignition's psychometric and workspace-state curves need to be read
from condition-stratified evaluation and workspace telemetry together. Passing
any or all of them would support functional workspace signatures only; it
would not establish phenomenal consciousness. The local papers most relevant
to these mappings include Sergent & Dehaene (2004), Sergent, Baillet & Dehaene
(2005), Pashler (1994), Dehaene et al. (2001, 2011), Naccache et al. (2002),
Dehaene, Lau & Kouider (2017), and Butlin et al. (2023).

The no-report task's action is now the majority choice supported by the noisy
evidence. The latent state is retained only for diagnostic alignment; a noisy
sample does not define a deterministic label for that hidden state. The
optional confidence target is the evidence-margin bin, not a calibrated
probability; the input now includes an explicit cue for whether confidence is
requested. The masked-priming answer is the response supported by the
presented target observation; the latent target remains available only for
perceptual-error metrics. Every evaluation reports GNW process signatures:
evidence-action and confidence-bin accuracy plus latent-truth alignment for
no-report, and response accuracy plus prime-intrusion rates by mask/congruency
condition for masked priming. Workspace runs record those same signatures for
normal, bypassed, and shuffled readback evaluations where available.

Two opt-in workspace interventions isolate task-specific mechanisms. Add
`--workspace-no-report-confidence-readout` to use an exact vote and
evidence-margin readout over the evidence values retrieved from semantic
workspace slots. Add `--workspace-masked-prime-trace` to store the masked prime
in its own workspace slot and supervise its query-time retrieval. The latter
intentionally lifts a masked item into the global workspace; compare it with
the default no-write condition as an architectural intervention. Both flags
require positive workspace key-value loss, entity-keyed writes, and semantic
entity addressing, and work with the individual track, `gnw_all`, or `all`.

### Functional signature diagnostics

Add `--workspace-functional-signature-diagnostics` to measure causal
contributions from enabled task-specific workspace pathways. It replays the
same balanced ID episodes with one enabled pathway disabled at a time and
writes per-track score changes as `feature_off/<name>` conditions in
`run_<id>_workspace_dependency.csv`. Existing `full`, `readback_bypassed`, and
`readback_shuffled` conditions remain, so a task score can be compared against
both a broken broadcast path and each explicit task helper. Only enabled
helpers are ablated. This adds one evaluation pass per active helper at every
tenth periodic evaluation and at the final evaluation; leave the flag off for
ordinary training.

Ignition also writes a response curve by evidence strength with predicted
access rate, target-report rate, response accuracy, and sample count to
`run_<id>_workspace_task_metrics.csv`. Interpret this as a task response
curve, not a direct measurement of conscious access. Condition-level GNW
process metrics remain in that file as well.

### Combined all-task run

`all` selects all eighteen workspace-capability tracks and does not include
graph traversal, which belongs to a separate dataset. The following run uses
the known 4-block Mamba-1/MoE/workspace settings and enables the available
task-specific helpers; periodic feature ablations report which tasks depend
on each helper. The helper flags are explicit scaffolds, so this run tests
their interaction with a shared workspace, not whether the model invented
those algorithms itself.

```bash
python -m src.initium.core_training 0.0 --dataset-type workspace-capability --workspace-task-track all --split-graph --split-graph-variant mamba1 --split-graph-num-blocks 4 --moe --moe-placement all --moe-num-experts 8 --moe-top-k 1 --moe-capacity-factor 1.5 --moe-backbone-residual-scale 0.1 --workspace --workspace-num-slots 64 --workspace-entity-keyed-writes --workspace-semantic-entity-addressing --workspace-key-value-loss-weight 0.1 --workspace-operation-executor --workspace-operation-diagnostics --workspace-nth-farthest-exact-readout --workspace-retrieval-exact-readout --workspace-exact-value-comparison --workspace-exact-selection --workspace-attentional-cue-readout --workspace-no-report-confidence-readout --workspace-masked-prime-trace --workspace-functional-signature-diagnostics --workspace-compile-recurrence --workspace-token-id-input --workspace-sparse-recurrence --seed 0 --total-steps 50000 --batch-size 16 --run-id-suffix wcap-all-functional-signatures-s0 --no-moe-diagnostics --no-segment-diagnostics --checkpoint-every 5000 --memory-type none --fp32
```

For a stricter learned-use comparison, repeat with the task-specific helper
flags removed while retaining generic semantic key-value supervision,
workspace readback, and `--workspace-functional-signature-diagnostics`. Keep
the seed and evaluation episodes matched; compare the two runs separately.

## Curriculum, evaluation, and metrics

Each track has six difficulty levels. `all` rotates the eighteen tracks evenly
across batches; `new_all` rotates the six capability tracks and `gnw_all`
rotates the eight GNW probes. Each multi-track
batch contains every selected track when `batch_size` is at least the selected
track count. Curriculum evaluation uses a small balanced set per required
challenge group. Advancement requires at least 80% episode exact match overall,
per track, and in each required group; token accuracy alone cannot advance a
lesson. Workspace readback-on, bypassed, and shuffled evaluations reuse the
same balanced episode objects and are logged by track and difficulty group.
The blink task additionally logs T1/T2 accuracy for each observed lag/mask
query type. This adds evaluation compute but keeps evaluation microbatching
bounded by the configured dataset batch size.
The dataset can also be run
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

### Retrieval exact readout

`--workspace-retrieval-exact-readout` is an opt-in positive control for the
`retrieval_aggregation` track. It supervises a scalar decoder from each entity's
addressed workspace slot after every initial write or update. At query time it
reads the requested entity slots and applies the requested recall, modulo-16
sum, even-count, or parity operation to the decoded values. For example, use
`--workspace-task-track all --workspace-num-slots 64
--workspace-entity-keyed-writes --workspace-semantic-entity-addressing
--workspace-retrieval-exact-readout` to test it alongside the other tracks.
The option requires at least 64 slots and a positive
`--workspace-retrieval-value-loss-weight` (default `0.1`). It does not require
KV memory.

Console output and `run_<id>_workspace_task_metrics.csv` report write-value
accuracy, query-slot lookup rate, queried-value accuracy, exact readout
accuracy, and invalid reads separately. This isolates payload preservation,
entity lookup, and the final aggregation operation; it is a task-specific
algorithmic control rather than a general learned aggregator.

### General key-value workspace readout

`--workspace-key-value-loss-weight 0.1` enables a vocabulary-sized value
decoder over semantic entity-addressed slot reads. It trains sparse value
reconstruction at explicit event commits and keyed reads at answer/query
positions. When an answer token has a direct `workspace_output_keys` mapping,
the task logits at that position are replaced with the slot readout, so the
answer depends on persistent state. The probe also respects the existing
workspace readback bypass and shuffle counterfactuals.

The first track mappings cover direct state answers for multi-agent belief,
variable binding, the scalar stages of mixed-skill pipelines, the two RSVP
targets, and symbolic multiple-object tracking. Capacity change-detection and
n-back compare two retrieved values through a generic equality operator;
rule switching selects a retrieved card feature using the retrieved rule.
These operation outputs are computed from slot-read predictions. Mixed-skill
two-value sorting is routed through two distinct keyed records when the
operation executor is enabled. Add
`--workspace-operation-executor` to execute mixed-skill and variable-binding
lookup/copy/modular-add/sort programs causally and write intermediate values
back into workspace slots. It requires a positive key-value loss weight; the
executor reads only input-derived operands, while sparse target values train
the committed slot contents. Without this flag, operation events continue to
use the ordinary task decoder. The key-value readout requires
`--workspace`, `--workspace-entity-keyed-writes`,
`--workspace-semantic-entity-addressing`, and
`--dataset-type workspace-capability`. It adds one small value head and sparse
metadata shaped `(batch, time, 4)`; it does not allocate another recurrent
slot bank. Training reports `L_ws_kv` alongside task loss.
The executor allocates a free semantic slot when an operation destination has
not yet been written by an input event, which allows the second sort result to
be committed independently.

Add `--workspace-operation-diagnostics` with the executor to report key-source
read accuracy, destination commit rate, and decoded committed-value accuracy
by operation at evaluation. It also reports final answer accuracy and episode
exact match for mixed-skill and variable-binding tracks. Diagnostic targets
are evaluation-only and are not passed to the executor. Rows are appended to
`run_<id>_workspace_task_metrics.csv`; the flag is off by default.

For capacity change detection, the output program compares the initial and
current location values and emits `BOOL_TRUE` exactly when they differ. The
optional `--workspace-exact-value-comparison` flag uses each addressed value
head's highest-scoring location token for that comparison, with a soft
straight-through surrogate used only for the training gradient. Evaluation
reports operand-value accuracy and comparison accuracy separately, so a failed
answer can be attributed to value retrieval or comparison. It requires
`--workspace-key-value-loss-weight > 0`, workspace entity-keyed writes, and
semantic entity addressing; supported tracks are `capacity_bottleneck`,
`new_all`, and `all`.

`--workspace-exact-selection` applies the same hard-forward/soft-gradient
approach to rule-switching selector programs. It chooses the most likely
stored rule token and then the most likely value token in that rule's candidate
slot. Evaluation reports rule/value operand accuracy separately from selected
output accuracy.

`--workspace-attentional-item-memory` stores every RSVP frame value in a slot
indexed by its order in the stream, including distractors. The write addresses
are derived from frame order and do not identify target locations. Sparse
key-value supervision trains the value reader at each frame boundary; answer
prediction still has to select the cue-matching targets through workspace
readback.

Add `--workspace-attentional-cue-readout` to make that retrieval path explicit.
It supervises separate color and shape decoders for each frame, while the
existing categorical value decoder learns the frame's scalar value,
then at the answer cue scores every occupied stream-order slot against the
visible color/shape conjunction. It selects the two highest-scoring matches in
temporal order and uses their scalar logits for T1/T2. The forward metadata
contains cue tokens and the answer step only; it does not contain target frame
indices, target values, or target labels. This flag implies
`--workspace-attentional-item-memory` and requires at least 26 slots, semantic
entity addressing, and positive key-value loss. Evaluation applies the same
readout to baseline and workspace-readback ablation passes.

`--workspace-sparse-recurrence` batches broadcast reads over spans with a
fixed workspace state, then executes the usual causal update at semantic write
and state-side operation steps. It reduces per-token recurrent launches while
preserving the read result at every token. It is disabled during per-step
workspace telemetry and falls back for unsupported recurrence modes.

Run the focused checks from the repository root with:

```bash
python -m pytest tests/unit/test_workspace_capability.py tests/integration/test_workspace_capability_pipeline.py -q
```
