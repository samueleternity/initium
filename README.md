# Initium

Initium is an experimental research framework for combining a Differentiable Neural Computer (DNC)/augmented memory with interchangeable sequence controllers and task/data pipelines. Its central idea is to keep an explicit, addressable external memory while exploring controllers such as LSTM, Mamba-family state-space models, and closed-form continuous-time (CfC) networks. The code also includes stochastic memory-write regularization, sparse expert layers, split-graph models, memory-structure experiments, and checkpoint-driven inference.

> **Research status:** Alpha and actively changing. Results below are experiment-specific observations, not guarantees that a configuration will reproduce on another machine, code revision, or dataset. This README is deliberately maintained as a working guide: update it when the code, default configuration, or evidence changes.

## What the project is investigating

The project studies how a neural controller can interact with a differentiable external memory on sequential tasks, especially graph traversal. It asks questions at several levels:

1. **Memory and controller:** Can a controller process a sequence and use DNC read/write operations to store and retrieve information needed later?
2. **Generalization:** How does performance change between the training curriculum and a held-out or out-of-distribution (OOD) evaluation distribution?
3. **Write regularization:** Does placing a probabilistic prior on each memory write improve generalization, and what are the costs or seed-to-seed variations?
4. **Compute structure:** Can the controller's sequence computation be parallelized or sparsified while preserving the sequential memory interface?
5. **Task breadth and deployment:** Can trained checkpoints be evaluated through a common inference interface across graph, text, audio, video, and multimodal tasks?

These are research questions, not all established project claims. In particular, supported dataset interfaces do not imply that every modality has been trained and validated to the same standard as graph traversal.

## High-level flow

```text
Configuration and random seed
        │
        ▼
Dataset registry ──► task episodes / curriculum ──► batch and sequence tensors
                                                          │
                                                          ▼
                                            Controller + DNC external memory
                                            ├─ controller processes sequence
                                            ├─ write head updates memory
                                            └─ read heads return memory content
                                                          │
                                                          ▼
                                   task loss + optional KL / MoE auxiliary losses
                                                          │
                                                          ▼
                           optimizer, evaluation, logs, checkpoints, curriculum
                                                          │
                                                          ▼
                      checkpoint loader ──► task inference ──► metrics and run logs
```

The training core delegates task construction and task-specific behavior to a dataset interface. The model combines a selected controller with DNC memory operations. During training, the loop computes task and optional auxiliary objectives, updates model parameters, periodically evaluates ID and OOD performance, advances the curriculum when its criteria are met, writes diagnostics, and saves resumable checkpoints. Inference reconstructs a model from a checkpoint, checks that the requested task and dimensions are supported, creates the task runner, and reports episode metrics. Optional inference caches can reuse eligible results or shared prefixes.

## Architecture and major components

### Differentiable external memory

The DNC supplies a memory matrix and differentiable read/write interface. The memory can retain information beyond a controller's immediate hidden state. Content-based addressing, read heads, and temporal-link structures are part of the underlying DNC implementation. The temporal link matrix has quadratic cost in the number of memory cells, which motivates the project's link-matrix ablations and memory-size investigations.

### Controllers

The training core exposes controller choices through its command-line interface and configuration:

- **LSTM:** baseline recurrent controller.
- **Mamba:** selective state-space controller.
- **Mamba-2 and Mamba-3:** additional state-space variants; current experiment notes report that these configurations underperformed in earlier tests and need further investigation.
- **CfC:** closed-form continuous-time controller.
- **Hybrid chains:** compositions of supported controller stages, such as a Mamba/CfC sequence.
- **Split-graph:** a parallel sequence backbone plus a separate combiner for the sequential memory-addressing work. It is an experimental way to expose more of the controller computation to parallel execution while retaining the memory interface and currently performs the best out of all options.

Some options require optional or hardware-sensitive dependencies, particularly `mamba-ssm` and `causal-conv1d`. A controller being selectable in code should not be read as evidence that it is equally mature or portable on every platform.

### Stochastic writes and KL regularization

The write-head extension models a write vector as a diagonal Gaussian. It can sample a write using the reparameterization trick and add a KL penalty to a prior distribution. The implementation supports a fixed standard-normal prior and a learned prior updated from recent write statistics on a periodic, detached snapshot schedule. The loss can include KL annealing and free bits; the learned-prior path logs its snapshot state separately.

The intended locality distinction is important: the per-write KL is computed from the current write distribution and a prior snapshot. In the learned-prior variant, the prior itself is fitted from recent writes periodically, outside the per-step gradient path. That makes the overall mechanism periodically data-conditioned even though the per-write calculation uses the current write and frozen prior snapshot.

Also KL regularization has specific, uninvestigated behaviour: it tends to stabilize grad_norms allowing the model to not collapse (detected once and needs further investigations) - it was mildly visible in graph-traversal experiments but its full strength occured in multimodal tasks where a model managed to collapse in lesson 14 with beta=0.0, meanwhile with beta=0.001(KL term active) the same model kept going with no issues.

### Other experimental mechanisms

- **Mixture of Experts (MoE):** sparse expert feed-forward capacity with routing and load-balancing diagnostics. The current project notes caution that this overhead may be unnecessary or harmful at smaller model sizes; its motivation is future scaling, not a demonstrated benefit for the current compact experiments.
- **Dynamic memory size (Dynamic-N):** usage-triggered growth of the memory cell array within configured bounds.
- **Link-matrix options:** ablations or sparsification of temporal-link computation.
- **Dynamic beta:** an optional bounded controller for the KL weight, distinct from a fixed-beta sweep.
- **Multimodal and real-data loaders:** interfaces for synthetic and selected real text, audio, video, graph, and multimodal inputs. Their practical coverage and validation are still evolving.

## Repository map

| Path | Role |
| --- | --- |
| `src/initium/core_training.py` | Dataset-agnostic training loop, CLI, evaluation, logging, checkpointing, and orchestration of model options. |
| `src/initium/config/` | Training, regularization, controller, and architecture defaults. |
| `src/initium/data/` | Dataset registry, task episodes, curriculum/data handling, codecs, and real-data utilities. |
| `src/initium/mamba_controller/` | DNC controller integration, Mamba variants, hybrid controllers, and split-graph implementation. |
| `src/initium/LNN_controller/` | CfC and hybrid continuous-time controller implementations. |
| `src/initium/memory_manipulation/` | Stochastic write heads, prior snapshots, memory resizing, link-matrix ablations, and compatibility helpers. |
| `src/initium/MoE/` | Sparse expert layer and its local usage notes. |
| `src/initium/inference/` | Checkpoint loading, task registry, inference engine, metrics, logging, and cache support. |
| `setup_wheels.py` | Environment/dependency setup entry point for specialized wheels. |
| `pyproject.toml` | Package metadata, base and optional dependencies, and development tooling groups. |

## Setup

The project targets Python 3.10 or later. Core dependencies are declared in `pyproject.toml`; accelerated state-space controllers may need platform-specific PyTorch/CUDA wheels and additional packages. Install the project and optional task dependencies in an environment appropriate to your hardware. For specialized dependency setup, inspect the repository's `setup_wheels.py` and its help output before running it; GPU package compatibility depends on the installed PyTorch, CUDA, and Python versions.

Typical editable installation for development:

```bash
python -m pip install -e .
```

Optional dependencies are grouped by modality, for example `text`, `video`, and `real-data` in the project metadata. Audio/video processing may also rely on system tools such as FFmpeg. Mamba-family components can require packages beyond the base install. The default training path is intended to be usable without those optional controller dependencies.

The source tree is packaged as `initium`; internal modules use the `initium.*` namespace.

## Training

The main entry point is `src/initium/core_training.py`. From the repository root, inspect the available arguments with:

```bash
python -m initium.core_training --help
```

The script accepts a positional KL weight (`beta`); omitting it runs the configured beta sweep. It also accepts a random seed and controller/task/architecture options. A representative graph-task run with the Mamba controller and the dynamic-memory/split-graph experiments enabled is:

```bash
python -m initium.core_training 0.0 --seed 0 --controller mamba --dynamic-n-mode --split-graph
```

This is an example of the CLI shape, not a universal best configuration. Check `src/initium/config/train_config.py` and `src/initium/config/controller_config.py` for the defaults in the checkout you're running. In particular, training duration, model size, beta sweep, dataset, evaluation cadence, and memory options may have changed since older experiment records were produced.

Training runs typically produce:

- periodic and final checkpoints, including model, optimizer, random-generator, curriculum, and architecture/prior state needed to resume or evaluate;
- step-level or periodic logs for task loss, KL, accuracy, OOD evaluation, and training health;
- additional CSV diagnostics for learned-prior snapshots and optional architecture features;
- a run identifier that encodes relevant settings to reduce accidental collisions.

Keep seeds, source revision, full command, configuration, device/dependency versions, and evaluation protocol with any result you intend to compare. A single final metric is not enough to characterize these experiments: curriculum position, checkpoint step, sampled versus deterministic writes, and evaluation sample count can change the interpretation.

## Evaluation and inference

The checkpoint-based inference CLI lives at `src/initium/inference/run_inference.py`. Its help output is the current reference for arguments:

```bash
python -m initium.inference.run_inference --help
```

The command accepts a checkpoint, task type, optional data link, episode count, seed, device, and task-specific settings. It can inspect checkpoint capabilities, compare reset and persistent experience modes, report memory-ablation behavior, and optionally configure inference caches. A basic invocation has this form:

```bash
python -m initium.inference.run_inference path/to/checkpoint.pt --dataset-type graph
```

The loader checks checkpoint/task compatibility and refuses unsupported task types or dimensions. Graph, text, audio, and video task runners are registered; multimodal execution is also represented in the code but its input dimensions depend on the selected modality combination. Verify the checkpoint's recorded dataset configuration before comparing results across task types.

## Experiments and current evidence

The supplied experiment records describe a staged investigation of a KL penalty on DNC memory writes. They are summarized here as results rather than by the names of the individual working files, so this overview can remain useful if those files are renamed or reorganized. -> If anyone wishes to see the original files, you can create an issue and will be contacted afterwards.

### Fixed-prior multi-seed study

The first study compared a no-KL anchor with a small fixed-prior KL weight over multiple random seeds with --controller lstm. It found meaningful heterogeneity: several seeds showed non-collapsed OOD gains, some showed little benefit, and two seeds showed genuine reversals. The strongest observed seed improved OOD triple accuracy by roughly 11–12 percentage points over its own anchor. The study therefore supports a **conditional** positive result for the tested configuration, not a claim that the effect is universal or statistically settled. The measured OOD perfect-traversal fraction also did not track triple accuracy uniformly, so the metric definition matters.

### Learned-prior study

The follow-up replaced the fixed prior with a prior periodically estimated from recent write statistics. Four deliberately selected seeds were tested: two previous reversal cases and two stronger fixed-prior cases. Outcomes were mixed: the two former reversals improved, one strong case was nearly flat, and the strongest fixed-prior result regressed by about five to six OOD points. Across the four seeds, the mean OOD change was small and not statistically distinguishable from zero; ID accuracy was lower by about 1.1–2.7 points in all four comparisons. Diagnostics did not indicate KL collapse. A regression-to-the-mean explanation is plausible (the reported correlation was large at n=4), but the sample is too small to distinguish that from a true repair of the earlier reversal cases. The structural audit found the learned prior's snapshot update was detached and periodic, with no second online posterior; that is a design characterization, not proof of a generalization benefit.

### Alternate Mamba-controller study

A separate study compared beta 0 and beta 0.001 in a Mamba, dynamic-memory, split-graph configuration across five seeds. The logs showed no statistically clear OOD or ID-to-OOD offset improvement from the KL term; beta 0 had a small nominal OOD edge in four of five seeds. The beta 0.001 runs consistently left the first curriculum lesson later. The comparison is provisional: only one seed reached the final curriculum lesson, four runs were unfinished, and there was no wide-evaluation sweep for that batch. This configuration should not be described as a completed replication of the earlier study because its controller, memory size, training budget, and evaluation coverage differ.

### Evidence boundaries and open work

- Results above apply to the exact runs and evaluation procedures recorded; they do not establish broad performance across modalities, hardware, or model scales.
- Phase 1's seed variation and Phase 2's four-seed selection limit aggregate claims. Report per-seed values and uncertainty alongside averages.
- The learned prior's apparent variance-dampening/mean-reversion effect needs more seeds and controls to separate it from failure-mode-specific repair.
- The alternate Mamba batch needs completion of unfinished seeds and late-curriculum evaluation before a gate-level conclusion.
- Mamba-2/Mamba-3 performance, MoE's useful scaling regime, memory-link trade-offs, dynamic memory growth, and real-data/multimodal behavior remain active investigation areas.
- Experiment status can change. Update this section when new runs are verified; retain the method, sample size, seed set, incompleteness, and uncertainty so summaries do not outrun evidence.

## Research foundations

The project draws on several complementary lines of work. These references motivate components and questions; their inclusion does not imply that Initium reproduces each paper's results.

### Differentiable memory and neural sequence models

- Graves et al., **“Hybrid computing using a neural network with dynamic external memory.”** DNC memory, differentiable addressing, and the graph-traversal motivation.
- Hasani et al., **“Closed-form Continuous-time Neural Networks.”** Continuous-time controller design.
- Gu and Dao, **“Mamba-3: Improved Sequence Modeling using State Space Principles.”** State-space sequence-modeling direction explored by controller variants.
- Dao and Gu, **“Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality.”** Structured state-space/sequence computation background.

### Delta-rule and state-space improvements

- **“Gated Delta Networks: Improving Mamba 2 with Delta Rule.”** Gated delta-rule sequence processing.
- **“Parallelizing Linear Transformers with the Delta Rule over Sequence Length.”** Parallelization ideas related to sequence-length computation.

### Sparse expert models

- Shazeer et al., **“Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer.”** Sparse expert routing foundations.
- Fedus et al., **“Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity.”** Simplified sparse expert scaling.
- **“Multi-Rate Mixture of Experts for Accelerating Liquid Neural Network Training.”** Motivation for combining expert computation with liquid-network training.
- **“MoE-Mamba: Efficient Selective State Space Models with Mixture of Experts.”** MoE and selective state-space combination.

### Stochastic regularization and the DNC baseline

- Higgins et al., **“beta-VAE: Learning Basic Visual Concepts with a Constrained Variational Framework.”** Gaussian KL regularization and reparameterized latent sampling adapted here to memory writes.
- Graves et al., **“Hybrid computing using a neural network with dynamic external memory.”** Baseline memory architecture and task family used by the Q21 study.
- Additional work on local credit assignment, priors, and continual-learning regularization informed the experiment design and interpretation. Consult the experiment notes for the specific comparisons and caveats.

## Development principles

Because the repository is new and research directions are still changing:

- Prefer describing mechanisms and measured outcomes over calling a configuration “best” without a scoped comparison.
- Distinguish implemented, exercised, and validated features. A registry entry is implementation coverage, not empirical validation.
- Preserve negative and mixed results; record the seed, comparison anchor, metric, evaluation window, and incomplete runs.
- Keep CLI help, configuration defaults, this README, and experiment records aligned as the project evolves.
- Treat research papers as motivation and technical background, not as evidence that this implementation matches their methods or results unless a reproduction has been explicitly checked.

## License

Initium is distributed under the GNU General Public License v3.0. See [`LICENSE`](LICENSE).

## Publishing to PyPI

Package versions come from `__version__` in `src/__init__.py`. After merging a version update into `main`, create a GitHub Release whose tag matches that version (for example, `v0.1.0`). The `Publish to PyPI` workflow builds and validates the wheel and source distribution, then publishes them automatically. It refuses releases whose tag points to a commit outside `main` or whose tag does not match the package version.

One-time setup: configure a PyPI Trusted Publisher for project `initium` with owner `samueleternity`, repository `initium`, workflow `publish-pypi.yml`, and GitHub environment `pypi`. This uses short-lived OIDC credentials; no PyPI API token needs to be stored in GitHub. The PyPI project name was checked and currently returns 404, but PyPI reserves it only after the first successful publication.