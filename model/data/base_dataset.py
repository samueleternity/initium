"""
file: data/base_dataset.py

Interface core_training.py talks to. Any new dataset (text, audio, video, ...)
implements BaseDataset and registers itself in dataset_registry.py; the core
training loop never imports anything dataset-specific.

Curriculum contract (returned by make_curriculum()):
    .lesson (int, 0-based), .table (sequence, len = number of lessons),
    .lesson_nr_cells (list[int], per-lesson memory size, overridable by core),
    .maybe_advance(model, device, step=None, optimizer=None)
        -> (lesson, id_score_pct, id_perfect_pct)
    Optional log-writer attributes set by core and read via getattr:
    advance_log_writer/_file, hop_log_writer/_file, field_log_writer/_file.
Datasets without a curriculum should return a 1-lesson curriculum.
"""


class BaseDataset:
    name = "base"
    input_dim: int = None          # model input/output width (DNC input_size)
    output_dim: int = None         # output_proj width
    advance_threshold: float = 0.85  # default accuracy floor for dynamic beta

    def make_curriculum(self):
        raise NotImplementedError

    def sample_batch(self, curriculum, batch_size):
        """-> (inputs, targets, mask) tensors, batch-first."""
        raise NotImplementedError

    def loss(self, output, target, mask):
        raise NotImplementedError

    def diversity(self, output, target, mask) -> float:
        raise NotImplementedError

    def set_output_proj(self, proj):
        raise NotImplementedError

    def evaluate_ood(self, model, device, num_episodes, rng, verbose_n=0, field_log=None):
        """-> (acc_pct, perfect_pct, breakdown_dict)"""
        raise NotImplementedError

    def evaluate_id_ablated(self, model, device, curriculum, lesson_idx):
        """Memory-off eval on lesson `lesson_idx` -> (acc_pct, perfect_pct)"""
        raise NotImplementedError

    def evaluate_id_combiner_stage_ablated(self, model, device, curriculum, lesson_idx, skip_stages):
        """Combiner-stage-off eval on lesson `lesson_idx` -> (acc_pct, perfect_pct).
        Optional: only implemented by datasets/models with a multi-stage
        split-graph combiner (see SplitGraphDNC)."""
        raise NotImplementedError

    def evaluate_robustness(self, model, device, curriculum, lesson_idx, perturbation, rng):
        """Same lesson-distribution eval as evaluate_id_ablated, but sampling
        PERTURBED episodes (modality-specific corruption) instead of
        ablating the model itself. -> (acc_pct, perfect_pct). Layer D
        ("Robustness") of the shared eval skeleton (Task/Generalization/
        Dependency/Robustness) -- optional per dataset, NotImplementedError
        by default."""
        raise NotImplementedError

    def field_log_header(self):
        """Column names (after the shared step/lesson/eval_type prefix
        core_training.py's field_breakdown CSV always writes) for THIS
        dataset's write_field_log() rows. Each dataset owns its own field
        schema (graph: src/edge/dst; the KV-chain family: value/cumsum) so
        core_training.py never hardcodes one dataset's column layout."""
        raise NotImplementedError

    def write_field_log(self, writer, file, step, lesson, eval_type, field_log):
        raise NotImplementedError