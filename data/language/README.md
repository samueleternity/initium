# Language and learned world-state pipeline

This directory owns the standalone language run configuration and human-facing
data contracts. Python implementation lives in `src/initium/data/language/`.
It is intentionally separate from the existing `workspace-capability` dataset:
that remains the controlled episodic benchmark and does not become a broad text
corpus when trained for more steps.

## Included pipeline

1. `tools/build_language_corpus.py` builds a documented text corpus from a
   user-supplied manifest. It normalizes text, redacts common email/phone
   patterns, filters short/noisy rows, exact-deduplicates documents, assigns a
   deterministic 98/1/1 document split, trains a byte-level BPE tokenizer on
   train text only, writes tokenized JSONL shards, and records licenses and
   token counts. It never downloads data. Supply only material whose license
   permits this use; inspect the emitted report and held-out splits.
2. `state_records.py` builds a symbolic transition curriculum with exact before
   and after states, actions, independently rendered same-state text views, and
   logic labels. This teaches consistency within that simulator; it is not
   physical grounding. Real image/audio/action observations need a separately
   licensed adapter and synchronized state annotations.
3. `dialogue.py` reads multi-turn JSONL conversations and masks loss to assistant
   turns while retaining the complete preceding conversation in model input.
   Format and state policy match `lang_chat`'s transcript-carry behavior; adapt
   the runtime before changing that policy.
4. `objectives.py` and `heads.py` provide causal token loss, VICReg-style paired
   view loss, action-conditioned future-state loss, and supervised logic loss.
   `pipeline.py` provides weighted objective interleaving and document packing.
5. `memory.py` provides source/confidence/time-aware facts and requires explicit
   approval before durable user memory writes. Treat it as the inspectable
   product policy around latent workspace state, not as a trained capability.
6. `checkpoint.py` saves model, optimizer, scheduler, RNG, data position, config
   fingerprint, and tokenizer fingerprint for exact-run resume checks;
   `profiling.py` captures tokens/sec, peak CUDA memory, and inference latency.
   `evaluation.py` separates language, meaning, world model, memory,
   conversation, generalization, and data integrity suites, and records the six
   scale-readiness gates from the plan.
7. `collators.py` builds padded objective batches while keeping state-pair IDs
   explicit and assistant-only dialogue labels masked correctly. `model.py`
   assembles the requested split-graph Mamba1 model, parallel dense
   workspace, modulation path, causal output projection, and JEPA/logic heads.
   `trainer.py` and `train.py` connect all objectives to one optimizer loop.
   The runner applies composed neuromodulator arms, a validation-based
   retention signal, gradient and loss outlier gates, consolidation, held-out
   objective evaluations, and the configured checkpoint cadence.

## Build the corpus

Create a JSONL manifest (paths are relative to the manifest):

```json
{"path":"sources/books.txt","license":"CC-BY-4.0","source":"books-v1"}
```

Install the optional tokenizer dependency and build shards:

```powershell
pip install -e ".[text]"
python tools/build_language_corpus.py data/language/sources.jsonl data/language/build --vocab-size 32000
python tools/build_language_world_data.py data/language/build/world --count 10000 --seed 0
```

Input text files are treated as one document per line. This keeps deterministic
document-level splits and deduplication meaningful. Inspect
`build/corpus_manifest.json`. Supply licensed dialogue JSONL shards, one
conversation per line, with `conversation_id`, `split` (`train`, `validation`,
or `test`), and `turns` containing `{ "role": ..., "content": ... }` objects.
Keep held-out conversations separate by `split`.

```powershell
python -m initium.data.language.train --config data/language/run_config.json --corpus-manifest data/language/build/corpus_manifest.json --world-data-dir data/language/build/world --dialogue-jsonl data/language/dialogue.jsonl --output-dir data/language/runs/langall-M-N1-N2-N3-N5-N4
```

The supplied config enables all five objectives, so all five streams are
required. The run writes `latest.pt`, `resolved_config.json`, and per-objective
validation losses to `evaluations.jsonl`. Resume with `--resume` followed by the
`latest.pt` path. Checkpoints fingerprint the tokenizer and every source file,
and preserve model/optimizer state, neuromodulator state, RNG state, objective
samplers, consolidation anchors, and data cursors.

Core-style overrides include `--split-graph-num-blocks`,
`--split-graph-mamba-d-state`, `--workspace-num-slots`,
`--workspace-key-dim`, `--workspace-scan-chunk`, `--mod-targets`,
`--mod-blocks`, `--mod-rank`, `--mod-lambda`, `--neuromod-combine`,
`--neuromod-range`, `--neuromod-every`, `--neuromod-slew`,
`--neuromod-groups`, `--batch-size`, `--hidden-size`, `--lr`, `--lr-min`,
`--warmup-steps`, `--total-steps`, and `--checkpoint-every`. Run
`initium-train-language --help` for the full list. Overrides are validated and
stored in `resolved_config.json`; changing one produces a different checkpoint
fingerprint. Split-graph Mamba1, FP32, and token-ID input are fixed requirements
for this model assembly; block count and Mamba state sizes are configurable.

## Architecture candidate

`run_config.json` records the requested combination: split-graph Mamba1 with
four blocks, no DNC memory, token-ID input, 128-slot parallel dense workspace
on block 1 (64-dimensional keys, scan chunks of 32), modulation, and the four
neuromodulation arms with retention control, outlier gating, and consolidation
lambda 0.1. It also records seed 0, FP32, 130,000 steps, and 5,000-step
checkpoints. The code exposes `core_training_args()` for those architecture
flags; the blank `dataset-type` and `workspace-task-track` in the supplied
snippet are placeholders, not usable CLI values.

The standalone `initium.data.language.train` entrypoint trains this pipeline;
the existing `core_training` loop remains for episodic `BaseDataset` runs.
Retention control uses a rise in held-out text loss between evaluations as its
retention signal because broad corpus training has no episodic lesson
curriculum. The runner reports separate held-out losses for text, same-state
alignment, transitions, logic, and dialogue. A matched-control comparison and
the broader task suites remain required for the scale gate. Do not use
`lang_all` as a substitute: it remains the closed synthetic task curriculum.
A tokenizer change requires new embeddings/output projection; do not resume an
old checkpoint as if its vocabulary were compatible.

## Matched control and scale gates

Train the workspace candidate and a simpler control against the identical
tokenizer, documents, split, token budget, optimizer budget, and held-out suites.
The control should disable the dense workspace and JEPA heads while retaining
the same shared Mamba backbone and text objective. Track throughput, peak
activation and optimizer memory, checkpoint size/resume, and inference latency.
Scale only when all fields in `ScaleGateReport` pass and JEPA gains hold on
held-out paraphrases, unseen entities/combinations, and action outcomes.

Keep train/validation/test ids disjoint; reserve test data for final reporting.
Add contamination scans for every external evaluation set and preserve source
license records. JEPA alignment is only one metric, never the model score.
