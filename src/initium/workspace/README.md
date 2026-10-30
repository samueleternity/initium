# Workspace feature guide

This guide describes the workspace options wired through
`src/initium/core_training.py`, `src/initium/mamba_controller/split_graph_dnc.py`,
and `broadcast.py`. Workspace is a persistent slot memory attached to the
split-graph MoE path: selected specialist outputs can write to slots, and slot
readback is broadcast to the specialists. It is intended for controlled
experiments on persistent, reusable state.

## Minimal setup

Enable the feature with `--workspace`. It requires `--split-graph --moe` and a
MoE placement that includes the backbone (`all` or `backbone`); `combiner`
placement alone cannot host a workspace. The split-graph backbone residual
scale also scales the final workspace-enriched backbone MoE residual. For the
current capability and graph experiments, `--moe-backbone-residual-scale 0.1`
is the established comparison setting.

```bash
python -m src.initium.core_training 0.0 --split-graph --split-graph-variant mamba1 \
  --split-graph-num-blocks 4 --moe --moe-placement all \
  --moe-backbone-residual-scale 0.1 --workspace --workspace-num-slots 5 \
  --workspace-compile-recurrence
```

`--workspace-num-slots` sets persistent capacity. Use `0` only as a plumbing or
bypass control; keyed addressing needs at least one slot, and permutation
state experiments need at least 16. More slots increase memory and update
work. Set the count to the largest simultaneously live set the task needs.

## Write selection, addressing, and persistence

| Option | What it changes | When to use it |
| --- | --- | --- |
| Default workspace writes | The selected top-k MoE routes contribute learned value proposals to shared slots. Reads from the slots are broadcast back to specialists. | Baseline workspace behavior and general task-family comparisons. |
| `--workspace-no-selection` | Uses random, unweighted specialist writes instead of MoE-selected writes. | Control whether learned routing is responsible for useful writes. Keep the rest of the setup matched to the selected-write run. |
| `--workspace-entity-keyed-writes` | Stores address/value pairs and reads by address similarity. Addresses are learned from specialist representations. | Test whether persistent per-entity bindings help beyond shared content slots. It needs `--workspace` and at least one slot. |
| `--workspace-key-match-threshold VALUE` | Cosine-match threshold for treating an address as an existing slot; a weaker match allocates a free slot when available. Default: `0.5`. | Sweep when learned addresses merge unrelated items or fragment one item's writes. Use the same value across matched runs. |
| `--workspace-semantic-entity-addressing` | For workspace-capability permutation episodes, uses fixed one-hot entity-token IDs as addresses and event masks for writes. Slot payloads remain learned. Requires entity-keyed writes. | Diagnose entity-to-state binding with stable keys. Use `--workspace-task-track permutation` or `all`; do not treat this as a generic language-task address mode. |
| `--workspace-permutation-transactional` | Suppresses the two ordinary swap-commit writes, reads both addressed slot payloads from the same pre-commit state, then exchanges them together at the pair boundary. The semantic decoder supplies the actual query answer logits for both object→position and position→object queries. | A swap-aware permutation experiment testing paired state composition and decoder use. Requires semantic addressing, at least 16 slots, and the `permutation` or `all` task track. This is a distinct algorithmic variant. |
| `--workspace-reset-each-step` | Clears the slot state before every token's write/read cycle, removing persistence between tokens. | Persistence ablation. It is not a useful setting for normal memory training. |

Semantic entity addressing uses entity IDs present in the input stream as
address metadata. It does not inject oracle positions into the learned payload.
The semantic implementation bounds slot-content RMS to 1 and limits readback
RMS relative to its routed writer to reduce state-scale instability.

## Training the workspace

| Option | What it changes | When to use it |
| --- | --- | --- |
| `--workspace-train-mode coadapted` | Trains the backbone, MoE, workspace, and task head together. This is the default. | Primary end-to-end comparison. |
| `--workspace-train-mode frozen` | Freezes the substrate and task head and trains workspace parameters. Normally start from a compatible workspace-disabled checkpoint with `--resume`. | Isolate whether the workspace can learn useful writes/readout on a fixed representation. |
| `--workspace-train-scope workspace` | In frozen mode, trains all workspace parameters. | Default frozen-workspace experiment. |
| `--workspace-train-scope gate` | In frozen mode, trains only the selected write gate (plus its prediction-error write projection when active). Adds segment-level stability instrumentation. Requires `--segment-diagnostics`. | Gate-only Phase 4 stability/control studies; pair with the continuous or bistable gate. |
| `--workspace-bptt-horizon STEPS` | Detaches recurrent slot gradients every `STEPS` tokens while preserving the same forward state trajectory. `0` means full-sequence backpropagation. | Reduce activation/gradient memory for long sequences or measure the value of long-range gradient flow. |
| `--workspace-compile-recurrence` | Compiles the sequential slot-update step with CUDA `torch.compile`; token-only projections are prepared over time, while the state recurrence remains sequential. | Keep this enabled for normal workspace training: it has repeatedly improved training throughput. It supports semantic/keyed addressing, reconsideration, write salience, and transactional pair commits. It does not support random writes (`--workspace-no-selection`) or bistable gates. Omit it only for those unsupported modes or an explicit eager-versus-compiled control. |

Frozen mode and gate scope change which parameters are trainable, so give those
runs distinct run IDs. Do not compare a frozen workspace run directly to a
coadapted run as if only the memory mechanism changed.

## Write-gate variants

| Option | What it changes | When to use it |
| --- | --- | --- |
| `--workspace-gate-variant continuous` | Standard differentiable write/update gate. Default. | Baseline and compiled recurrence; use this for the first comparison. |
| `--workspace-gate-variant bistable` | Uses recurrent competitive settling with per-candidate background noise and hard forward write decisions with a straight-through gradient. Requires at least two slots and `--moe-diagnostics`. | Experimental stochastic-ignition comparison after the continuous baseline is stable. It is not supported by compiled recurrence. |
| `--workspace-bistable-selection-scale SCALE` | Scales the content-selection drive in the bistable gate. Default: `1.0`. | Selection-strength ablation. Change one scale at a time. |
| `--workspace-bistable-noise-scale SCALE` | Scales independent background noise. Default: `0.75`; `0` disables the noise contribution. | Noise ablation or tuning after verifying the ordinary bistable configuration. |
| `--workspace-bistable-gradient-mode truncated|unrolled` | `truncated` detaches intermediate settling states in the backward pass while preserving the forward settling trajectory; `unrolled` differentiates through each iteration. Default: `truncated`. | Use truncated for the stable first run. Use unrolled as a controlled gradient-path comparison. |
| `--workspace-bistable-deterministic` | Disables bistable gate noise for a deterministic diagnostic. | Debug or controlled inference; normal bistable evaluation is stochastic and seed-reproducible. |
| `--bistable-debug` | Emits detailed settling, gate-gradient, clipping, optimizer, and moment diagnostics. | Short, targeted debugging only; output and runtime overhead are high. Requires bistable mode. |
| `--bistable-anomaly-detection` and `--bistable-anomaly-step STEP` | Asks autograd to report the forward source of a failing backward operation, optionally for one absolute training step. | Use only to locate NaNs in bistable backward; it substantially slows training. |
| `--workspace-stability-max-activation VALUE` | Abort limit for instrumented activations in gate-scope stability runs. Default: `100000`. | Lower it when reproducing a known instability; it is not a general activation-clipping knob. |

## Reconsideration and prediction-error signals

| Option | What it changes | When to use it |
| --- | --- | --- |
| `--workspace-reconsideration` | Allows a routed proposal to force-commit into an eligible occupied slot when it passes the trigger rule, overriding the ordinary soft update gate. | Tasks with corrections, changing facts, or repeated updates where overwriting a previous binding is necessary. Requires `--moe-diagnostics` for trigger telemetry. |
| `--workspace-reconsideration-surprise-threshold VALUE` | Threshold in `[0,1]` for the normalized innovation trigger. Default: `0.5`. | Tune how readily an occupied slot may be revised; report trigger and changed fractions with the score. |
| `--workspace-reconsideration-trigger innovation` | Uses content innovation to decide whether an occupied slot merits reconsideration. Default. | First trigger to try; it requires no prediction-error head. |
| `--workspace-reconsideration-trigger random_rate_matched` | Randomly triggers at a rate matched to the innovation trigger. | Null control to check whether reconsideration gains come from meaningful trigger timing or just extra writes. |
| `--workspace-reconsideration-trigger prediction_error` | Uses detached causal input-prediction error to trigger reconsideration. | Test whether surprising input predicts a needed memory revision. Requires `--sgr-pred-loss-weight > 0`, split-graph Mamba-1 with at least two blocks, and `--moe-diagnostics`. |
| `--workspace-reconsideration-trigger prediction_error_random_rate_matched` | Uses a random trigger matched to the prediction-error firing rate. | Null control for prediction-error-triggered reconsideration. Has the same prediction-head requirements. |
| `--sgr-workspace-write-salience` | Adds detached causal prediction error to workspace write-gate logits, independently of reconsideration. | Test whether surprising inputs deserve stronger writes. Requires `--sgr-pred-loss-weight > 0`, workspace with at least one slot, and split-graph Mamba-1 with at least two blocks. It works with compiled recurrence. |

For prediction-error consumers, `--sgr-pred-loss-weight` must be positive so the
causal predictor is trained. Keep the predictor loss weight and signal settings
matched when comparing reconsideration and salience variants.

## State supervision and evaluation

| Option | What it logs or trains | When to use it |
| --- | --- | --- |
| `--workspace-semantic-state-diagnostics` | Replays held-out permutation prefixes, reports accepted entity slots, slot value RMS, decoded state versus oracle state, and a workspace-readback-bypassed comparison. | Diagnose whether writes were accepted and whether readback contains the expected entity state. Evaluation adds runtime, especially when many events are sampled. |
| `--workspace-state-diagnostic-max-events N` | Caps sampled state events per diagnostic episode. Default: `16`; `0` includes every event. | Reduce diagnostic cost while keeping a spread of initialization and swap points. |
| `--workspace-state-loss-weight WEIGHT` | Adds auxiliary cross-entropy for decoding each initialized entity's position after initialization and swap events. | Directly train and observe state content when using semantic addressing on permutation/all. It uses oracle state only as the auxiliary target. Use a fresh run when changing this model/loss configuration. |
| Transactional held-out metrics | With `--workspace-permutation-transactional`, the workspace task-metrics CSV records object→position and position→object answer accuracy, decoded state accuracy by committed swap depth, and successful pair-commit rate. | Use these metrics to separate answer-direction weaknesses, state degradation with repeated swaps, and failed address resolution. |
| `--eval-workspace-mode checkpoint|reset-each-step|bypass` | Evaluation-only counterfactual: use checkpoint behavior, reset persistent slots every step, or bypass workspace readback. | Measure dependence on persistent readback without retraining. Requires `--eval-only --workspace`; `checkpoint` is the normal behavior. |

## Suggested experiment order

1. Establish the ordinary coadapted workspace with continuous gate and persistent
   slots. Use `--moe-diagnostics` so routed writes and capacity are visible.
2. Compare selected writes with `--workspace-no-selection`, then compare shared
   writes with `--workspace-entity-keyed-writes` where the task has recurring
   entities.
3. On permutation, use semantic addressing and state diagnostics. Add
   `--workspace-state-loss-weight` only when the question is whether direct
   state supervision helps.
4. Run `--workspace-permutation-transactional` as a separate swap-aware model
   variant. Inspect both answer directions, state accuracy by swap depth, and
   pair-commit success rate.
5. Add reconsideration or prediction-error write salience as separate
   interventions with their matched random controls.
6. Try bistable last; it changes the write decision dynamics and cannot be
   combined with compiled recurrence.

The workspace's ordinary console diagnostics include write mass, active-write
fraction, gate mean, slot/readback scale, and address match/allocation/overflow
rates where applicable. Reconsideration and prediction-error write salience
also create `run_<id>_workspace_telemetry.csv`. Workspace-capability evaluation
strata are written to `run_<id>_workspace_task_metrics.csv`.
