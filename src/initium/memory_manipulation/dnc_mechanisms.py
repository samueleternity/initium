"""Composable DNC read-addressing and write-value mechanisms.

The stock pytorch-dnc Memory is kept as the baseline. This subclass changes
only the read content score and/or the update applied to the selected rows;
usage, allocation, erase/write addressing, read modes and temporal links keep
their upstream implementations and semantics.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from dnc.memory import Memory
from dnc.util import σ, θ
from initium.memory_manipulation.nvrtc_compat import (
    pop_product_numerical_diagnostics,
    set_product_backward_trace,
)

READ_VARIANTS = ("cosine", "relational-mlp", "relational-residual")
WRITE_VARIANTS = ("learned", "kanerva-closed-form")


class MechanismMemory(Memory):
    """Stock DNC memory with independently selectable read and write rules.

    The memory interface and state are evaluated in fp32 even when the outer
    controller uses CUDA autocast. This protects cosine-addressing backward
    from low-precision intermediate gradients near the DNC norm epsilon.
    """

    def __init__(
        self,
        *args,
        read_variant: str = "cosine",
        write_variant: str = "learned",
        read_residual_scale: float = 1.0,
        read_residual_max_ratio: float = 0.5,
        relational_hidden_size: int | None = None,
        observation_variance: float = 1.0,
        **kwargs,
    ):
        if read_variant not in READ_VARIANTS:
            raise ValueError(f"Unknown DNC read variant: {read_variant!r}")
        if write_variant not in WRITE_VARIANTS:
            raise ValueError(f"Unknown DNC write variant: {write_variant!r}")
        if read_residual_scale < 0:
            raise ValueError("read_residual_scale must be non-negative")
        if read_residual_max_ratio < 0:
            raise ValueError("read_residual_max_ratio must be non-negative")
        if observation_variance <= 0:
            raise ValueError("observation_variance must be positive")
        super().__init__(*args, **kwargs)
        self.read_variant = read_variant
        self.write_variant = write_variant
        self.read_residual_scale = float(read_residual_scale)
        self.read_residual_max_ratio = float(read_residual_max_ratio)
        self.observation_variance = float(observation_variance)
        self._collect_stage_diagnostics = False
        self._stage_diagnostic_sums: dict[str, torch.Tensor] = {}
        self._stage_diagnostic_counts: dict[str, int] = {}
        self._stage_diagnostic_min: dict[str, torch.Tensor] = {}
        self._stage_diagnostic_max: dict[str, torch.Tensor] = {}
        self._backward_trace_enabled = False
        self._backward_trace_step: int | None = None
        self._backward_trace_first_bad_step: int | None = None
        self._backward_trace_reported_stages: set[str] = set()
        self._activation_peaks: dict[str, torch.Tensor] = {}
        self._read_weight_violation_count: torch.Tensor | None = None
        self._read_weight_raw_mass_max: torch.Tensor | None = None
        self._read_weight_mass_max: torch.Tensor | None = None
        self._memory_value_max: torch.Tensor | None = None
        self._read_vector_max: torch.Tensor | None = None
        self._link_row_mass_max: torch.Tensor | None = None
        self._link_column_mass_max: torch.Tensor | None = None
        self.relational_score: nn.Module | None = None
        if read_variant in {"relational-mlp", "relational-residual"}:
            hidden = relational_hidden_size or max(32, self.cell_size)
            self.relational_score = nn.Sequential(
                nn.Linear(2 * self.cell_size, hidden),
                nn.GELU(),
                nn.Linear(hidden, 1),
            )
            if read_variant == "relational-residual":
                # Preserve the DNC content-addressing behavior at step zero;
                # the learned relation function starts as a zero residual
                # and is then optimized from task feedback.
                output_layer = self.relational_score[-1]
                if not isinstance(output_layer, nn.Linear):
                    raise TypeError("relational score output layer must be linear")
                nn.init.zeros_(output_layer.weight)
                if output_layer.bias is not None:
                    nn.init.zeros_(output_layer.bias)
            if self.device is not None:
                self.relational_score.to(self.device)

    def forward(self, xi: torch.Tensor, hidden: dict[str, torch.Tensor]):
        """Run all DNC interface, addressing, and state math in fp32.

        Controller autocast is useful for the large sequence blocks, but the
        DNC cosine address divides by a norm product plus a small epsilon.
        Keeping its full forward/backward path in fp32 avoids casting large
        intermediate derivatives back to fp16. The memory state is fp32 by
        construction; the conversion also makes resumed/custom states safe.
        """
        fp32_hidden = {
            name: value.float() if value.is_floating_point() else value
            for name, value in hidden.items()
        }
        if xi.is_cuda:
            with torch.autocast(device_type="cuda", enabled=False):
                return super().forward(xi.float(), fp32_hidden)
        return super().forward(xi.float(), fp32_hidden)

    def begin_stage_diagnostics(self) -> None:
        """Collect lightweight read/write statistics until finish is called."""
        self._collect_stage_diagnostics = True
        self._stage_diagnostic_sums = {}
        self._stage_diagnostic_counts = {}
        self._stage_diagnostic_min = {}
        self._stage_diagnostic_max = {}

    def _record_stage_value(self, name: str, value: torch.Tensor) -> None:
        values = value.detach().float()
        value_mean = values.mean()
        value_min = values.min()
        value_max = values.max()
        if name in self._stage_diagnostic_sums:
            self._stage_diagnostic_sums[name] = self._stage_diagnostic_sums[name] + value_mean
            self._stage_diagnostic_min[name] = torch.minimum(
                self._stage_diagnostic_min[name], value_min
            )
            self._stage_diagnostic_max[name] = torch.maximum(
                self._stage_diagnostic_max[name], value_max
            )
            self._stage_diagnostic_counts[name] += 1
        else:
            self._stage_diagnostic_sums[name] = value_mean
            self._stage_diagnostic_min[name] = value_min
            self._stage_diagnostic_max[name] = value_max
            self._stage_diagnostic_counts[name] = 1

    def finish_stage_diagnostics(self) -> dict[str, float]:
        """Stop collection and return per-write/read means and extrema."""
        self._collect_stage_diagnostics = False
        result: dict[str, float] = {}
        for name, total in self._stage_diagnostic_sums.items():
            result[f"{name}_mean"] = float((total / self._stage_diagnostic_counts[name]).item())
            result[f"{name}_min"] = float(self._stage_diagnostic_min[name].item())
            result[f"{name}_max"] = float(self._stage_diagnostic_max[name].item())
        result["diagnostic_steps"] = float(max(self._stage_diagnostic_counts.values(), default=0))
        return result

    def set_backward_trace(self, enabled: bool, *, step: int | None = None) -> None:
        """Watch DNC operation boundaries for the first non-finite gradient.

        This is intended for short diagnostic runs: each watched gradient
        boundary synchronizes while checking for a non-finite value.
        """
        self._backward_trace_enabled = bool(enabled)
        self._backward_trace_step = step
        set_product_backward_trace(enabled, self._report_bad_gradient if enabled else None)

    def _report_bad_gradient(self, stage: str, grad: torch.Tensor) -> torch.Tensor:
        step = self._backward_trace_step
        if self._backward_trace_first_bad_step is not None:
            if step != self._backward_trace_first_bad_step:
                return grad
            if stage in self._backward_trace_reported_stages:
                return grad
        if bool(torch.isfinite(grad).all()):
            return grad
        if self._backward_trace_first_bad_step is None:
            self._backward_trace_first_bad_step = step

        finite = torch.isfinite(grad)
        finite_values = grad.detach().float()[finite]
        finite_max = (
            float(finite_values.abs().max().item()) if finite_values.numel() else float("nan")
        )
        finite_rms = (
            float(finite_values.square().mean().sqrt().item())
            if finite_values.numel()
            else float("nan")
        )
        bad_fraction = float((~finite).float().mean().item())
        bad_rows = []
        if grad.ndim >= 2:
            per_row_bad_fraction = (~finite).reshape(grad.shape[0], -1).float().mean(dim=1)
            bad_rows = [
                int(index)
                for index in torch.nonzero(per_row_bad_fraction > 0, as_tuple=False)
                .flatten()
                .detach()
                .cpu()
                .tolist()
            ]
        is_first = not self._backward_trace_reported_stages
        self._backward_trace_reported_stages.add(stage)
        label = "first" if is_first else "also"
        print(
            f"[DNC-BWD-TRACE] {label} non-finite gradient through "
            f"{stage} at step {step}: bad={bad_fraction:.3%} "
            f"finite_rms={finite_rms:.4g} finite_max={finite_max:.4g} "
            f"shape={tuple(grad.shape)} bad_batch_rows={bad_rows}"
        )
        return grad

    def _watch_backward(self, stage: str, value: torch.Tensor) -> torch.Tensor:
        if not self._backward_trace_enabled:
            return value
        peak = value.detach().float().abs().amax()
        previous = self._activation_peaks.get(stage)
        self._activation_peaks[stage] = peak if previous is None else torch.maximum(previous, peak)
        if not value.requires_grad:
            return value

        value.register_hook(lambda grad: self._report_bad_gradient(stage, grad))
        return value

    @staticmethod
    def _track_peak(previous: torch.Tensor | None, value: torch.Tensor) -> torch.Tensor:
        peak = value.detach().float().abs().amax()
        return peak if previous is None else torch.maximum(previous, peak)

    def pop_numerical_diagnostics(self) -> dict[str, float]:
        """Return/reset compact DNC invariant and magnitude metrics."""
        tensors = {
            "read_weight_raw_mass_abs_max": self._read_weight_raw_mass_max,
            "read_weight_mass_abs_max": self._read_weight_mass_max,
            "memory_abs_max": self._memory_value_max,
            "read_vector_abs_max": self._read_vector_max,
            "link_row_mass_max": self._link_row_mass_max,
            "link_column_mass_max": self._link_column_mass_max,
        }
        device_tensors = [value for value in tensors.values() if value is not None]
        values = torch.stack(device_tensors).cpu().tolist() if device_tensors else []
        result: dict[str, float] = {}
        index = 0
        for name, value in tensors.items():
            if value is not None:
                result[name] = float(values[index])
                index += 1
        if self._read_weight_violation_count is not None:
            result["read_weight_violation_calls"] = float(
                self._read_weight_violation_count.detach().cpu().item()
            )
        else:
            result["read_weight_violation_calls"] = 0.0
        self._read_weight_violation_count = None
        self._read_weight_raw_mass_max = None
        self._read_weight_mass_max = None
        self._memory_value_max = None
        self._read_vector_max = None
        self._link_row_mass_max = None
        self._link_column_mass_max = None
        result.update(
            {
                f"{stage}_abs_max": float(value.detach().cpu().item())
                for stage, value in self._activation_peaks.items()
            }
        )
        self._activation_peaks.clear()
        result.update(pop_product_numerical_diagnostics())
        return result

    def get_usage_vector(self, usage, free_gates, read_weights, write_weights):
        self._watch_backward("usage_input", usage)
        self._watch_backward("usage_free_gates", free_gates)
        self._watch_backward("usage_read_weights", read_weights)
        self._watch_backward("usage_write_weights", write_weights)
        updated = super().get_usage_vector(usage, free_gates, read_weights, write_weights)
        return self._watch_backward("usage_update", updated)

    def allocate(self, usage, write_gate):
        self._watch_backward("allocation_usage_input", usage)
        self._watch_backward("allocation_write_gate", write_gate)
        allocation, updated_usage = super().allocate(usage, write_gate)
        return self._watch_backward("allocation_weights", allocation), self._watch_backward(
            "allocation_usage", updated_usage
        )

    def write_weighting(self, memory, write_content_weights, allocation_weights, write_gate, allocation_gate):
        self._watch_backward("write_address_content_weights", write_content_weights)
        self._watch_backward("write_address_allocation_weights", allocation_weights)
        self._watch_backward("write_address_gate", write_gate)
        self._watch_backward("write_address_allocation_gate", allocation_gate)
        weights = super().write_weighting(
            memory, write_content_weights, allocation_weights, write_gate, allocation_gate
        )
        return self._watch_backward("write_weights", weights)

    def get_link_matrix(self, link_matrix, write_weights, precedence):
        updated = super().get_link_matrix(link_matrix, write_weights, precedence)
        if self._backward_trace_enabled:
            row_mass = updated.detach().float().clamp_min(0).sum(dim=-1).amax()
            column_mass = updated.detach().float().clamp_min(0).sum(dim=-2).amax()
            self._link_row_mass_max = (
                row_mass
                if self._link_row_mass_max is None
                else torch.maximum(self._link_row_mass_max, row_mass)
            )
            self._link_column_mass_max = (
                column_mass
                if self._link_column_mass_max is None
                else torch.maximum(self._link_column_mass_max, column_mass)
            )
        return self._watch_backward("temporal_link_matrix", updated)

    def update_precedence(self, precedence, write_weights):
        updated = super().update_precedence(precedence, write_weights)
        return self._watch_backward("precedence", updated)

    def content_weightings(self, memory, keys, strengths):
        self._watch_backward("content_address_memory", memory)
        self._watch_backward("content_address_keys", keys)
        self._watch_backward("content_address_strengths", strengths)
        # Preserve pytorch-dnc's exact operation order while exposing stages
        # hidden inside its cosine-addressing helper.
        scores = self._watch_backward("content_cosine_scores", θ(memory, keys))
        logits = self._watch_backward(
            "content_strength_scaled_scores", scores * strengths.unsqueeze(2)
        )
        weights = self._watch_backward("content_softmax_weights", σ(logits, 2))
        return self._watch_backward("content_weights", weights)

    def read_weightings(self, memory, content_weights, link_matrix, read_modes, read_weights):
        raw = super().read_weightings(memory, content_weights, link_matrix, read_modes, read_weights)
        if self._backward_trace_enabled:
            raw_mass = raw.detach().float().sum(dim=-1, keepdim=True)
            self._read_weight_raw_mass_max = self._track_peak(
                self._read_weight_raw_mass_max, raw_mass
            )

        self._watch_backward("read_weights_raw", raw)
        if self._backward_trace_enabled:
            self._read_weight_mass_max = self._track_peak(
                self._read_weight_mass_max, raw.sum(dim=-1, keepdim=True)
            )
            violation = (
                (raw < 0).any()
                | (raw.sum(dim=-1) > 1.0 + 1e-5).any()
                | (~torch.isfinite(raw).all())
            ).to(dtype=torch.float32)
            self._read_weight_violation_count = (
                violation
                if self._read_weight_violation_count is None
                else self._read_weight_violation_count + violation
            )
        return raw

    def read_vectors(self, memory, read_weights):
        if self._backward_trace_enabled:
            self._memory_value_max = self._track_peak(self._memory_value_max, memory)
        vectors = super().read_vectors(memory, read_weights)
        if self._backward_trace_enabled:
            self._read_vector_max = self._track_peak(self._read_vector_max, vectors)
        return self._watch_backward("read_vectors", vectors)

    def new(self, batch_size: int = 1):
        hidden = super().new(batch_size)
        if self.write_variant == "kanerva-closed-form":
            # Isotropic Gaussian covariance per row, represented by its scalar
            # variance. This is episode state, not a learned model parameter.
            hidden["write_posterior_variance"] = torch.ones(
                batch_size,
                self.nr_cells,
                1,
                device=hidden["memory"].device,
                dtype=hidden["memory"].dtype,
            )
        return hidden

    def clone(self, hidden):
        cloned = super().clone(hidden)
        if self.write_variant == "kanerva-closed-form":
            cloned["write_posterior_variance"] = hidden["write_posterior_variance"].clone()
        return cloned

    def erase(self, hidden):
        hidden = super().erase(hidden)
        if self.write_variant == "kanerva-closed-form":
            hidden["write_posterior_variance"].fill_(1.0)
        return hidden

    def _read_score_components(self, memory, keys):
        """Return cosine and learned score components before key-strength scaling."""
        if self.read_variant == "cosine":
            return self._dnc_cosine_scores(memory, keys), None
        batch, rows, width = memory.shape
        heads = keys.size(1)
        candidates = memory[:, None, :, :].expand(batch, heads, rows, width)
        queries = keys[:, :, None, :].expand(batch, heads, rows, width)
        pair = torch.cat((queries, candidates), dim=-1)
        relational_scores = self.relational_score(pair).squeeze(-1)
        if self.read_variant == "relational-residual":
            cosine_scores = self._dnc_cosine_scores(memory, keys)
            return cosine_scores, relational_scores
        return relational_scores, None

    def _read_similarity_scores(self, memory, keys):
        """Return one unscaled read score for every query and memory row."""
        base_scores, residual_scores = self._read_score_components(memory, keys)
        if residual_scores is not None:
            residual_scores = residual_scores - residual_scores.mean(dim=-1, keepdim=True)
            # Add epsilon before sqrt: the residual scorer is zero-initialized,
            # so RMS can be exactly zero on the first forward pass. Clamping
            # after sqrt leaves an infinite sqrt derivative in the graph.
            residual_rms = (residual_scores.square().mean(dim=-1, keepdim=True) + 1e-12).sqrt()
            normalized_residual = residual_scores / residual_rms
            cosine_rms = (base_scores.square().mean(dim=-1, keepdim=True) + 1e-12).sqrt()
            bounded_residual = (
                self.read_residual_max_ratio
                * cosine_rms
                * torch.tanh(self.read_residual_scale * normalized_residual)
            )
            return base_scores + bounded_residual
        return base_scores

    @staticmethod
    def _dnc_cosine_scores(memory, keys):
        """Match pytorch-dnc's cosine denominator, including its 1e-6 delta."""
        queries = keys.unsqueeze(2)
        candidates = memory.unsqueeze(1)
        dot = (queries * candidates).sum(dim=-1)
        query_norm = torch.linalg.vector_norm(queries, dim=-1)
        memory_norm = torch.linalg.vector_norm(candidates, dim=-1)
        return dot / (query_norm * memory_norm + 1e-6)

    def read_content_weightings(self, memory, keys, strengths):
        """Score rows, retaining DNC's key-strength softmax normalization."""
        self._watch_backward("read_content_memory", memory)
        self._watch_backward("read_content_keys", keys)
        self._watch_backward("read_content_strengths", strengths)
        scores = self._read_similarity_scores(memory, keys)
        self._watch_backward("read_content_scores", scores)
        weights = F.softmax(scores * strengths.unsqueeze(-1), dim=-1)
        return self._watch_backward("read_content_weights", weights)

    def read(self, read_keys, read_strengths, read_modes, hidden):
        self._watch_backward("memory_before_read", hidden["memory"])
        self._watch_backward("read_weights_previous", hidden["read_weights"])
        self._watch_backward("link_matrix_before_read", hidden["link_matrix"])
        self._watch_backward("read_keys", read_keys)
        self._watch_backward("read_strengths", read_strengths)
        self._watch_backward("read_modes", read_modes)
        if self.read_variant == "cosine":
            if self._collect_stage_diagnostics:
                # Use the upstream content-weighting implementation so the
                # baseline diagnostics match the actual DNC read path exactly.
                with torch.no_grad():
                    content_weights = self.content_weightings(
                        hidden["memory"], read_keys, read_strengths
                    )
                    entropy = -(
                        content_weights.clamp_min(1e-12) * content_weights.clamp_min(1e-12).log()
                    ).sum(dim=-1)
                    self._record_stage_value("read_content_entropy", entropy)
                    self._record_stage_value(
                        "read_content_max_weight", content_weights.max(dim=-1).values
                    )
                    self._record_stage_value(
                        "ordinary_read_score_cosine_correlation",
                        torch.ones_like(read_strengths),
                    )
            return super().read(read_keys, read_strengths, read_modes, hidden)
        if self._collect_stage_diagnostics:
            with torch.no_grad():
                memory = hidden["memory"]
                cosine, residual = self._read_score_components(memory, read_keys)
                scores = self._read_similarity_scores(memory, read_keys)
                cosine_centered = cosine - cosine.mean(dim=-1, keepdim=True)
                scores_centered = scores - scores.mean(dim=-1, keepdim=True)
                numerator = (cosine_centered * scores_centered).sum(dim=-1)
                denominator = torch.sqrt(
                    cosine_centered.square().sum(dim=-1) * scores_centered.square().sum(dim=-1)
                ).clamp_min(1e-12)
                self._record_stage_value(
                    "ordinary_read_score_cosine_correlation",
                    numerator / denominator,
                )
                if residual is not None:
                    effective_residual = scores - cosine
                    cosine_rms = cosine.square().mean(dim=-1).sqrt().clamp_min(1e-12)
                    residual_rms = effective_residual.square().mean(dim=-1).sqrt()
                    self._record_stage_value("relational_read_residual_rms", residual_rms)
                    self._record_stage_value(
                        "relational_read_residual_to_cosine_rms",
                        self.read_residual_scale * residual_rms / cosine_rms,
                    )
        content_weights = self.read_content_weightings(hidden["memory"], read_keys, read_strengths)
        if self._collect_stage_diagnostics:
            with torch.no_grad():
                entropy = -(
                    content_weights.clamp_min(1e-12) * content_weights.clamp_min(1e-12).log()
                ).sum(dim=-1)
                self._record_stage_value("read_content_entropy", entropy)
                self._record_stage_value(
                    "read_content_max_weight", content_weights.max(dim=-1).values
                )
        hidden["read_weights"] = self.read_weightings(
            hidden["memory"],
            content_weights,
            hidden["link_matrix"],
            read_modes,
            hidden["read_weights"],
        )
        read_vectors = self.read_vectors(hidden["memory"], hidden["read_weights"])
        return read_vectors, hidden

    def write(
        self,
        write_key,
        write_vector,
        erase_vector,
        free_gates,
        read_strengths,
        write_strength,
        write_gate,
        allocation_gate,
        hidden,
    ):
        self._watch_backward("memory_before_write", hidden["memory"])
        self._watch_backward("usage_before_write", hidden["usage_vector"])
        self._watch_backward("write_weights_previous", hidden["write_weights"])
        self._watch_backward("precedence_before_write", hidden["precedence"])
        self._watch_backward("link_matrix_before_write", hidden["link_matrix"])
        self._watch_backward("write_key", write_key)
        self._watch_backward("write_vector", write_vector)
        self._watch_backward("erase_vector", erase_vector)
        self._watch_backward("free_gates", free_gates)
        self._watch_backward("write_read_strengths", read_strengths)
        self._watch_backward("write_strength", write_strength)
        self._watch_backward("write_gate", write_gate)
        self._watch_backward("allocation_gate", allocation_gate)
        if self.write_variant == "learned":
            updated = super().write(
                write_key,
                write_vector,
                erase_vector,
                free_gates,
                read_strengths,
                write_strength,
                write_gate,
                allocation_gate,
                hidden,
            )
            self._watch_backward("memory_after_write", updated["memory"])
            self._watch_backward("usage_after_write", updated["usage_vector"])
            self._watch_backward("write_weights_after_write", updated["write_weights"])
            self._watch_backward("precedence_after_write", updated["precedence"])
            self._watch_backward("link_matrix_after_write", updated["link_matrix"])
            return updated

        # Keep DNC's usage, allocation and content-based write addressing.
        hidden["usage_vector"] = self.get_usage_vector(
            hidden["usage_vector"],
            free_gates,
            hidden["read_weights"],
            hidden["write_weights"],
        )
        write_content_weights = self.content_weightings(hidden["memory"], write_key, write_strength)
        allocation, _ = self.allocate(hidden["usage_vector"], allocation_gate * write_gate)
        hidden["write_weights"] = self.write_weighting(
            hidden["memory"],
            write_content_weights,
            allocation,
            write_gate,
            allocation_gate,
        )

        # Exact online posterior mean/covariance update for each independently
        # modeled row: prior N(memory_i, variance_i I), observation
        # N(write_vector, observation_variance I), with DNC write weight as
        # fractional observation precision. The learned projection supplies
        # the observation; Bayesian fusion replaces erase-and-add blending.
        weight = hidden["write_weights"].transpose(1, 2)
        variance = hidden["write_posterior_variance"]
        noise = self.observation_variance
        gain = (weight * variance) / (noise + weight * variance)
        observation = write_vector.expand_as(hidden["memory"])
        old_memory = hidden["memory"]
        posterior_memory = old_memory + gain * (observation - hidden["memory"])
        hidden["memory"] = posterior_memory
        hidden["write_posterior_variance"] = variance * noise / (noise + weight * variance)
        if self._collect_stage_diagnostics:
            posterior_variance = hidden["write_posterior_variance"]
            self._record_stage_value(
                "posterior_mean_update_abs", (posterior_memory - old_memory).abs()
            )
            self._record_stage_value("posterior_variance", posterior_variance)
            self._record_stage_value("posterior_variance_reduction", variance - posterior_variance)
            self._record_stage_value("write_weight_mass", weight)
            self._record_stage_value("active_write_cell_fraction", (weight > 1e-6).float())

        hidden["link_matrix"] = self.get_link_matrix(
            hidden["link_matrix"], hidden["write_weights"], hidden["precedence"]
        )
        hidden["precedence"] = self.update_precedence(hidden["precedence"], hidden["write_weights"])
        self._watch_backward("memory_after_write", hidden["memory"])
        self._watch_backward("usage_after_write", hidden["usage_vector"])
        self._watch_backward("write_weights_after_write", hidden["write_weights"])
        self._watch_backward("precedence_after_write", hidden["precedence"])
        self._watch_backward("link_matrix_after_write", hidden["link_matrix"])
        return hidden


def build_memory(
    *,
    input_size: int,
    nr_cells: int,
    cell_size: int,
    read_heads: int,
    device=None,
    independent_linears: bool = True,
    read_variant: str = "cosine",
    write_variant: str = "learned",
    read_residual_scale: float = 1.0,
    read_residual_max_ratio: float = 0.5,
    observation_variance: float = 1.0,
):
    """Construct a stock-compatible memory for the selected mechanism pair."""
    return MechanismMemory(
        input_size=input_size,
        nr_cells=nr_cells,
        cell_size=cell_size,
        read_heads=read_heads,
        device=device,
        independent_linears=independent_linears,
        read_variant=read_variant,
        write_variant=write_variant,
        read_residual_scale=read_residual_scale,
        read_residual_max_ratio=read_residual_max_ratio,
        observation_variance=observation_variance,
    )
