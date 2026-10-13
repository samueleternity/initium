"""Stochastic competitive write ignition for shared workspaces.

The gate is an experimental, opt-in approximation of population ignition:
slot candidates receive a content-selection drive and independent background
noise, then settle through recurrent self-excitation and lateral inhibition.
Forward writes are binary; a straight-through estimator carries gradients
through the settled probabilities during training.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class BistableWriteGate(nn.Module):
    """Resolve slot-write candidates with recurrent competitive dynamics."""

    def __init__(
        self,
        *,
        selection_scale: float = 1.0,
        noise_scale: float = 0.75,
        iterations: int = 6,
        temperature: float = 0.4,
    ) -> None:
        super().__init__()
        if selection_scale < 0 or noise_scale < 0:
            raise ValueError("selection_scale and noise_scale must be >= 0")
        if iterations < 1:
            raise ValueError("iterations must be >= 1")
        if temperature <= 0:
            raise ValueError("temperature must be > 0")
        self.selection_scale = float(selection_scale)
        self.noise_scale = float(noise_scale)
        self.iterations = int(iterations)
        self.temperature = float(temperature)
        self.bias = nn.Parameter(torch.tensor(1.1))
        self.self_excitation_logit = nn.Parameter(torch.tensor(1.1))
        self.inhibition_logit = nn.Parameter(torch.tensor(0.5))
        self._last_diagnostics: dict[str, torch.Tensor] = {}
        self._noise_seed = int(torch.randint(0, 2**31 - 1, ()).item())
        self._noise_generators: dict[str, torch.Generator] = {}
        self._pending_generator_states: dict[str, torch.Tensor] = {}

    def get_extra_state(self) -> dict:
        """Persist the independent noise streams with model checkpoints."""
        states = dict(self._pending_generator_states)
        states.update(
            {key: generator.get_state().cpu() for key, generator in self._noise_generators.items()}
        )
        return {"noise_seed": self._noise_seed, "generator_states": states}

    def set_extra_state(self, state: dict) -> None:
        self._noise_seed = int(state["noise_seed"])
        self._pending_generator_states = {
            key: value.cpu() for key, value in state.get("generator_states", {}).items()
        }
        self._noise_generators.clear()

    def _sample_background(self, selection: torch.Tensor) -> torch.Tensor:
        device_key = str(selection.device)
        generator = self._noise_generators.get(device_key)
        if generator is None:
            generator = torch.Generator(device=selection.device)
            state = self._pending_generator_states.pop(device_key, None)
            if state is None:
                generator.manual_seed(self._noise_seed)
            else:
                try:
                    generator.set_state(state)
                except RuntimeError:
                    # CPU and CUDA generator states are not interchangeable;
                    # keep inference portable while preserving exact replay
                    # whenever a checkpoint returns to the same device.
                    generator.manual_seed(self._noise_seed)
            self._noise_generators[device_key] = generator
        return torch.randn(
            selection.shape,
            device=selection.device,
            dtype=selection.dtype,
            generator=generator,
        )

    def forward(
        self,
        selection: torch.Tensor,
        *,
        selection_confidence: torch.Tensor | None = None,
        candidate_mask: torch.Tensor | None = None,
        noise_scale: float | None = None,
        selection_scale: float | None = None,
        deterministic: bool = False,
        collect_diagnostics: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return hard gates and their pre-commit ignition probabilities.

        ``selection`` is a nonnegative (batch, slot) selection-strength
        tensor. The stochastic term is sampled independently per candidate
        and remains fixed through the recurrent settling iterations.
        """
        if selection.ndim != 2:
            raise ValueError("selection must have shape (batch, slots)")
        if selection.shape[-1] < 2:
            raise ValueError("bistable competition requires at least two slots")
        if candidate_mask is not None and tuple(candidate_mask.shape) != (selection.shape[0],):
            raise ValueError("candidate_mask must have shape (batch,)")
        scale = self.selection_scale if selection_scale is None else selection_scale
        noise = self.noise_scale if noise_scale is None else noise_scale
        if scale < 0 or noise < 0:
            raise ValueError("selection_scale and noise_scale must be >= 0")

        selection = selection.float().clamp_min(0.0)
        normalized = selection / selection.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        eps = torch.finfo(normalized.dtype).eps
        selection_logit = torch.logit(normalized.clamp(eps, 1.0 - eps))
        uniform_logit = torch.logit(
            torch.full_like(normalized[:, :1], 1.0 / normalized.shape[-1]).clamp(
                eps, 1.0 - eps
            )
        )
        selection_logit = selection_logit - uniform_logit
        if selection_confidence is None:
            confidence_logit = torch.zeros_like(selection_logit[:, :1])
        else:
            confidence = selection_confidence.float().reshape(-1, 1).clamp(eps, 1.0 - eps)
            confidence_logit = torch.logit(confidence)
        sampled_background = self._sample_background(selection)
        background = (
            sampled_background * noise
            if not deterministic and noise > 0
            else torch.zeros_like(selection)
        )
        drive = scale * (selection_logit + confidence_logit) + background + self.bias
        self_excitation = F.softplus(self.self_excitation_logit)
        inhibition = F.softplus(self.inhibition_logit)
        activity = torch.sigmoid(drive / self.temperature)
        for _ in range(self.iterations):
            population_activity = activity.sum(dim=-1, keepdim=True)
            lateral_input = population_activity - activity
            activity = torch.sigmoid(
                (drive + self_excitation * activity - inhibition * lateral_input)
                / self.temperature
            )

        hard = (activity >= 0.5).to(activity.dtype)
        committed = hard + activity - activity.detach()
        valid = (
            torch.ones(selection.shape[0], device=selection.device, dtype=torch.bool)
            if candidate_mask is None
            else candidate_mask.to(device=selection.device, dtype=torch.bool)
        )
        committed = committed * valid.unsqueeze(-1).to(committed.dtype)
        if collect_diagnostics:
            with torch.no_grad():
                flat = activity.detach()[valid]
                candidate_count = valid.sum().float() * selection.shape[-1]
                candidate_denom = candidate_count.clamp_min(1.0)
                histogram = (
                    torch.histc(flat, bins=10, min=0.0, max=1.0)
                    if flat.numel()
                    else activity.new_zeros((10,))
                )
                self._last_diagnostics = {
                    "gate_candidate_count": candidate_count,
                    "gate_probability_mean": flat.sum() / candidate_denom,
                    "gate_histogram": histogram / candidate_denom,
                    "gate_low_frac": (flat < 0.1).float().sum() / candidate_denom,
                    "gate_mid_frac": ((flat >= 0.1) & (flat <= 0.9)).float().sum()
                    / candidate_denom,
                    "gate_high_frac": (flat > 0.9).float().sum() / candidate_denom,
                    "write_frequency": hard.detach()[valid].sum() / candidate_denom,
                    "selection_strength_mean": normalized.detach()[valid].amax(dim=-1).sum()
                    / valid.sum().clamp_min(1),
                }
        else:
            self._last_diagnostics = {}
        return committed.to(selection.dtype), activity

    def last_diagnostics(self) -> dict[str, torch.Tensor]:
        return dict(self._last_diagnostics)
