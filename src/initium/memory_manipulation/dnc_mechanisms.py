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

READ_VARIANTS = ("cosine", "relational-mlp")
WRITE_VARIANTS = ("learned", "kanerva-closed-form")


class MechanismMemory(Memory):
    """Stock DNC memory with independently selectable read and write rules."""

    def __init__(
        self,
        *args,
        read_variant: str = "cosine",
        write_variant: str = "learned",
        relational_hidden_size: int | None = None,
        observation_variance: float = 1.0,
        **kwargs,
    ):
        if read_variant not in READ_VARIANTS:
            raise ValueError(f"Unknown DNC read variant: {read_variant!r}")
        if write_variant not in WRITE_VARIANTS:
            raise ValueError(f"Unknown DNC write variant: {write_variant!r}")
        if observation_variance <= 0:
            raise ValueError("observation_variance must be positive")
        super().__init__(*args, **kwargs)
        self.read_variant = read_variant
        self.write_variant = write_variant
        self.observation_variance = float(observation_variance)
        self._collect_stage_diagnostics = False
        self._stage_diagnostic_sums: dict[str, torch.Tensor] = {}
        self._stage_diagnostic_counts: dict[str, int] = {}
        self._stage_diagnostic_min: dict[str, torch.Tensor] = {}
        self._stage_diagnostic_max: dict[str, torch.Tensor] = {}
        self.relational_score: nn.Module | None = None
        if read_variant == "relational-mlp":
            hidden = relational_hidden_size or max(32, self.cell_size)
            self.relational_score = nn.Sequential(
                nn.Linear(2 * self.cell_size, hidden),
                nn.GELU(),
                nn.Linear(hidden, 1),
            )
            if self.device is not None:
                self.relational_score.to(self.device)

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
            result[f"{name}_mean"] = float(
                (total / self._stage_diagnostic_counts[name]).item()
            )
            result[f"{name}_min"] = float(self._stage_diagnostic_min[name].item())
            result[f"{name}_max"] = float(self._stage_diagnostic_max[name].item())
        result["diagnostic_steps"] = float(
            max(self._stage_diagnostic_counts.values(), default=0)
        )
        return result

    def new(self, batch_size: int = 1):
        hidden = super().new(batch_size)
        if self.write_variant == "kanerva-closed-form":
            # Isotropic Gaussian covariance per row, represented by its scalar
            # variance. This is episode state, not a learned model parameter.
            hidden["write_posterior_variance"] = torch.ones(
                batch_size, self.nr_cells, 1,
                device=hidden["memory"].device,
                dtype=hidden["memory"].dtype,
            )
        return hidden

    def clone(self, hidden):
        cloned = super().clone(hidden)
        if self.write_variant == "kanerva-closed-form":
            cloned["write_posterior_variance"] = hidden[
                "write_posterior_variance"
            ].clone()
        return cloned

    def erase(self, hidden):
        hidden = super().erase(hidden)
        if self.write_variant == "kanerva-closed-form":
            hidden["write_posterior_variance"].fill_(1.0)
        return hidden

    def read_content_weightings(self, memory, keys, strengths):
        """Score candidate rows for reads; writes continue using DNC cosine."""
        if self.read_variant == "cosine":
            return self.content_weightings(memory, keys, strengths)
        batch, rows, width = memory.shape
        heads = keys.size(1)
        candidates = memory[:, None, :, :].expand(batch, heads, rows, width)
        queries = keys[:, :, None, :].expand(batch, heads, rows, width)
        pair = torch.cat((queries, candidates), dim=-1)
        logits = self.relational_score(pair).squeeze(-1)
        return F.softmax(logits * strengths.unsqueeze(-1), dim=-1)

    def read(self, read_keys, read_strengths, read_modes, hidden):
        if self.read_variant == "cosine":
            return super().read(read_keys, read_strengths, read_modes, hidden)
        if self._collect_stage_diagnostics:
            with torch.no_grad():
                batch, rows, width = hidden["memory"].shape
                heads = read_keys.size(1)
                candidates = hidden["memory"][:, None, :, :].expand(
                    batch, heads, rows, width
                )
                queries = read_keys[:, :, None, :].expand(batch, heads, rows, width)
                cosine = F.cosine_similarity(queries, candidates, dim=-1)
                relational = self.relational_score(
                    torch.cat((queries, candidates), dim=-1)
                ).squeeze(-1)
                cosine_centered = cosine - cosine.mean(dim=-1, keepdim=True)
                relational_centered = relational - relational.mean(dim=-1, keepdim=True)
                numerator = (cosine_centered * relational_centered).sum(dim=-1)
                denominator = torch.sqrt(
                    cosine_centered.square().sum(dim=-1)
                    * relational_centered.square().sum(dim=-1)
                ).clamp_min(1e-12)
                self._record_stage_value(
                    "ordinary_read_score_cosine_correlation", numerator / denominator
                )
        content_weights = self.read_content_weightings(
            hidden["memory"], read_keys, read_strengths
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
        if self.write_variant == "learned":
            return super().write(
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

        # Keep DNC's usage, allocation and content-based write addressing.
        hidden["usage_vector"] = self.get_usage_vector(
            hidden["usage_vector"],
            free_gates,
            hidden["read_weights"],
            hidden["write_weights"],
        )
        write_content_weights = self.content_weightings(
            hidden["memory"], write_key, write_strength
        )
        allocation, _ = self.allocate(
            hidden["usage_vector"], allocation_gate * write_gate
        )
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
        posterior_memory = old_memory + gain * (
            observation - hidden["memory"]
        )
        hidden["memory"] = posterior_memory
        hidden["write_posterior_variance"] = variance * noise / (
            noise + weight * variance
        )
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
        hidden["precedence"] = self.update_precedence(
            hidden["precedence"], hidden["write_weights"]
        )
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
        observation_variance=observation_variance,
    )
