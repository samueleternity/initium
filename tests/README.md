# Test suite guide

This directory tests the model components, training/checkpoint behavior, and inference paths in `src/model`. Tests use small synthetic inputs so the CPU suite can run quickly without downloading datasets or pretrained models.

## Running tests

The repository's `just test` recipe runs:

```sh
pytest -m "not gpu"
```

This is the default CPU suite and matches the test command in GitHub Actions. To run one test module directly, use `pytest tests/path/to/test_file.py`. Pytest's configured test root is this directory, and `pyproject.toml` adds `src/model` to the import path because current production modules import sibling packages by their top-level names.

Tests marked `gpu` are excluded by `just test`. To select them explicitly, use `pytest -m gpu` on a machine with CUDA, the matching PyTorch build, and the required CUDA extensions installed. Selecting them on a CPU-only machine does not make them CPU-compatible.

## CPU and GPU coverage in CI

The current workflow in `.github/workflows/lint-and-test.yml` uses the standard `ubuntu-latest` GitHub-hosted runner, installs CPU-only PyTorch, and calls `setup_wheels.py --skip gpu-wheels`. It then runs `just test`, which deselects every test marked `gpu`.

As a result, GPU-only tests cannot execute on this workflow's current runner. They need a GPU-equipped runner, such as an external/self-hosted GPU VM, or a GitHub-hosted GPU larger runner if one is available to the repository's organization. GitHub documents GPU-powered larger runners as an organization/enterprise feature; standard runners used here do not provide a CUDA GPU. See [GitHub's larger-runner overview](https://docs.github.com/en/actions/concepts/runners/larger-runners) and [GPU runner specifications](https://docs.github.com/en/actions/reference/runners/larger-runners).

### Mamba status

The CPU-only CI environment does not run the Mamba tests. The repository pins `mamba-ssm` in `setup_wheels.py` and normally installs CUDA-matched Mamba and `causal-conv1d` wheels. A probe attempted to install `mamba-ssm==2.3.2.post1` without CUDA extensions and execute the controller integration cases on CPU. All five cases failed while importing the controllers because `selective_scan_cuda` was unavailable; none reached their forward/backward assertions. Mamba-2 and Mamba-3 also use fused Triton kernels in their full-sequence paths. For that reason, Mamba variants are not included in the CPU controller matrix.

Mamba-related tests that remain in this suite are explicitly marked `gpu`; they are intended for a CUDA runner with compatible extensions. Their presence does not mean GitHub's standard `ubuntu-latest` runner can execute them. On the current standard-runner CI, they are deselected by `just test`. The pinned [Mamba-2 implementation](https://github.com/state-spaces/mamba/blob/v2.3.2.post1/mamba_ssm/modules/mamba2.py) calls a fused scan kernel for full-sequence execution, and [Mamba-3](https://github.com/state-spaces/mamba/blob/v2.3.2.post1/mamba_ssm/modules/mamba3.py) dispatches fused SISO/MIMO kernels.

## Fixtures and shared setup

`conftest.py` provides the common test building blocks:

- `build_tiny_rnn` creates small LSTM, CfC, hybrid, or split-graph DNC models, adds deterministic stochastic write heads, and returns an optimizer for tests that need one.
- `tiny_model_kwargs` uses a small model configuration to keep unit and integration checks inexpensive.
- `synthetic_graph_edges` and `synthetic_kv_facts` provide local test data instead of external datasets.
- `configure_tiny_training` redirects logs/checkpoints to temporary directories and reduces training/evaluation settings for short end-to-end tests.
- `assert_run_artifacts` checks checkpoint/log creation and verifies that numeric log values are finite.
- `seeded_rng`, `tmp_checkpoint_dir`, `device`, and `gpu_device` provide deterministic random state, temporary storage, and device selection. `gpu_device` skips when CUDA is unavailable.

## Test inventory

### `unit/`

Fast tests of an individual component or contract.

- `test_capabilities.py`
  - `test_recorded_and_legacy_capabilities` checks reading recorded task capabilities, inferring legacy capabilities, and rejecting unsupported task types.
  - `test_dimension_mismatch_is_rejected` checks that a task whose input/output dimensions disagree with a checkpoint is rejected.
- `test_checkpoint_io.py`
  - `test_checkpoint_required_keys_and_override` checks required checkpoint fields and caller-provided model-config overrides.
  - `test_describe_checkpoint_includes_architecture_summary` checks that the human-readable checkpoint description includes controller, dimensions, link mode, and MoE status.
- `test_dynamic_n_controller.py`
  - `test_ema_growth_cooldown_and_state_roundtrip` checks usage EMA updates, growth thresholds, cooldown behavior, ceiling/floor constraints, growth history, and state save/restore.
- `test_link_matrix_ablation.py`
  - `test_dense_and_ablated_modes` checks that dense mode preserves upstream link behavior and ablated mode bypasses link updates.
  - `test_sparse_topk_keeps_largest_magnitude_per_row` checks row-wise top-k link sparsification.
  - `test_patch_layer_selection` checks that link-mode patches affect only the requested memory layer.
- `test_model_loader.py`
  - `test_loader_rebuilds_from_checkpoint_architecture` rebuilds an LSTM controller from its saved config and checks the restored state dict.
  - `test_legacy_loader_uses_controller_defaults` checks legacy checkpoints that omit newer configuration fields.
- `test_moe_layer.py`
  - `test_switch_moe_validation_routing_and_aux_gradient` checks expert-count validation, routing records, and gradient flow from the load-balancing auxiliary loss.
  - `test_capacity_drop_zeros_overflow_token_output` checks that tokens beyond expert capacity are dropped as zero outputs.
  - `test_multisource_moe_preserves_source_shapes` checks per-source shape preservation and routing diagnostics from a shared expert bank.
- `test_nvrtc_compat.py`
  - `test_patch_is_idempotent_and_cpu_parity` checks the NVRTC compatibility patch is idempotent and preserves CPU `prod`/`cumprod` behavior. The CUDA log-space branch is not covered here.
- `test_real_data_pipeline.py`
  - `test_bigram_pool_and_split_are_deterministic` checks bigram-matrix conversion into facts and deterministic, disjoint train/test splits.
- `test_resize_memory.py`
  - `test_resize_transplants_weights_head_and_optimizer_state` checks that resizing memory retains learned transforms and stochastic write heads, rebinds the model module, and migrates optimizer state to replacement parameters.
  - `test_resized_cell_count_is_checkpointed_and_reloaded` checks that a resized cell count is saved and reconstructed when loading a checkpoint.
- `test_stochastic_write_head.py`
  - `test_deterministic_forward_has_no_kl` checks deterministic writes return the mean projection and accumulate no KL loss.
  - `test_phase_one_and_general_kl_are_same_for_standard_prior` checks the general diagonal-Gaussian KL reduces to the original standard-normal formula and prior buffers receive no gradients.
  - `test_learned_prior_kl_matches_diagonal_gaussian_formula` checks KL values against a hand-computed nonstandard prior.
  - `test_snapshot_uses_controlled_sample_mean_and_clamped_variance` checks prior snapshot fitting, variance clamping, sample counts, and snapshot-step tracking.
  - `test_snapshot_refit_and_free_bits` checks free-bits behavior, snapshot refitting, finite prior variance, and prior-state roundtripping.

### `gradients/`

Checks that selected differentiable paths produce finite, nonzero gradients.

- `test_controller_cell_backward.py`
  - `test_cfc_cell_backward_is_finite` runs a short CfC sequence and checks its gradients.
  - `test_mamba_cell_backward_is_finite`, `test_mamba2_cell_backward_is_finite`, and `test_mamba3_cell_backward_is_finite` exercise recurrent cell steps and backward passes. These are marked `gpu` and require the matching Mamba package/extensions and a GPU runner.
- `test_kl_gradients.py`
  - `test_kl_path_reaches_shared_mean_parameters` checks that the KL objective reaches the shared mean projection.
  - `test_kl_head_gradcheck_lite` numerically checks the KL derivative with double-precision inputs.
- `test_moe_aux_loss_gradients.py`
  - `test_aux_loss_alone_trains_router` checks that the MoE load-balancing term alone creates a finite, nonzero router gradient.

### `integration/`

Checks behavior across multiple components without running a full production training workload.

- `test_controller_matrix.py`
  - `test_controller_forward_backward` is parameterized over LSTM, CfC, chained CfC, and split-graph CfC. It checks output sequence length and finite gradients. Mamba variants were removed from this CPU matrix after the pinned Mamba package failed to import without `selective_scan_cuda`.
- `test_dynamic_n_end_to_end.py`
  - `test_dynamic_n_decision_applies_live_resize` connects a Dynamic-N growth decision to a live memory resize and checks the resulting size and growth history.
- `test_moe_integration.py`
  - `test_cfc_with_moe_forward_and_aux_loss` checks a CfC+MoE DNC forward/backward path and the auxiliary loss.
  - `test_hybrid_mamba_cfc_with_moe` checks hybrid Mamba/CfC plus MoE, but is `gpu`-marked and requires the Mamba runtime on a GPU runner.
- `test_resize_x_link_matrix.py`
  - `test_link_mode_survives_resize_and_rebinds_module` checks that a link-matrix ablation remains active after memory resizing and that the model points to the replacement module.
- `test_split_graph_dnc.py`
  - `test_split_graph_forward_backward_and_no_memory_contract` checks split-graph output, backward finiteness, and zero read-vector input when memory pass-through is disabled.
  - `test_split_graph_resume_requires_memory_state` checks that resuming at a nonzero sequence step without memory state raises an error.
  - `test_split_graph_resumed_tail_matches_full_sequence` checks that resumed tail outputs match the corresponding outputs from a full-sequence run.

### `checkpoint/`

Checks persistence and restoration across process/model reconstruction boundaries.

- `test_checkpoint_roundtrip.py`
  - `test_model_state_roundtrip_rebuilds_independently` saves a model, loads it in a fresh Python subprocess, and compares the reconstructed model's outputs with the original.
- `test_resume_training.py`
  - `test_resume_preserves_step_and_curriculum` runs a short training session, resumes from its checkpoint, and checks step progression, run identity, and curriculum continuity.

### `smoke/`

Short end-to-end paths intended to catch wiring/configuration failures.

- `test_inference_smoke.py`
  - `test_inference_and_result_prefix_cache_smoke` builds a tiny checkpoint and exercises checkpoint loading, capability detection, graph inference, result caching, and shared-prefix state caching.
- `test_tiny_training_run.py`
  - `test_tiny_training_run` runs two training steps for LSTM, CfC, and chained CfC and checks generated logs/checkpoints.
  - `test_tiny_training_option_modes_independently` runs separate tiny jobs with MoE, Dynamic-N, and link-matrix ablation enabled.
  - `test_tiny_split_graph_cfc_run` smoke-tests a split-graph CfC training run.
  - `test_gpu_tiny_training_paths` covers Mamba, hybrid Mamba/CfC, and split-graph Mamba training configurations. It is `gpu`-marked and does not run in the standard CPU CI job.

## Current scope limits

- These tests validate small synthetic workloads and implementation contracts. They do not establish model quality or performance on full research datasets.
- The standard test job has no CUDA device and intentionally omits GPU wheel installation; GPU-only coverage requires a GPU-enabled runner.
- Mamba CPU support is not established by this suite. In the attempted CPU probe, the pinned `mamba-ssm==2.3.2.post1` package lacked `selective_scan_cuda` and failed before its controller tests could execute.
- Mypy is currently configured for `src/model`, not the tests. Pytest runs the tests, while repository-wide Ruff commands lint and format-check them.
- Production modules currently use top-level sibling imports from `src/model`; changing to a conventional `src/initium` package layout is outside this suite's scope.
