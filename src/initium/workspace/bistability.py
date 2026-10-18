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
        gradient_mode: str = "truncated",
    ) -> None:
        super().__init__()
        if selection_scale < 0 or noise_scale < 0:
            raise ValueError("selection_scale and noise_scale must be >= 0")
        if iterations < 1:
            raise ValueError("iterations must be >= 1")
        if temperature <= 0:
            raise ValueError("temperature must be > 0")
        if gradient_mode not in ("truncated", "unrolled"):
            raise ValueError("gradient_mode must be 'truncated' or 'unrolled'")
        self.selection_scale = float(selection_scale)
        self.noise_scale = float(noise_scale)
        self.iterations = int(iterations)
        self.temperature = float(temperature)
        self.gradient_mode = gradient_mode
        self.bias = nn.Parameter(torch.tensor(1.1))
        self.self_excitation_logit = nn.Parameter(torch.tensor(1.1))
        self.inhibition_logit = nn.Parameter(torch.tensor(0.5))
        self._last_diagnostics: dict[str, torch.Tensor] = {}
        # Runtime-only detailed telemetry. The core trainer toggles this with
        # --bistable-debug; it does not affect the gate equations or checkpoints.
        self.debug_diagnostics = False
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
        valid = (
            torch.ones(selection.shape[0], device=selection.device, dtype=torch.bool)
            if candidate_mask is None
            else candidate_mask.to(device=selection.device, dtype=torch.bool)
        )
        scale = self.selection_scale if selection_scale is None else selection_scale
        noise = self.noise_scale if noise_scale is None else noise_scale
        if scale < 0 or noise < 0:
            raise ValueError("selection_scale and noise_scale must be >= 0")

        selection = selection.float().clamp_min(0.0)
        normalized = selection / selection.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        eps = torch.finfo(normalized.dtype).eps
        selection_logit = torch.logit(normalized.clamp(eps, 1.0 - eps))
        uniform_logit = torch.logit(
            torch.full_like(normalized[:, :1], 1.0 / normalized.shape[-1]).clamp(eps, 1.0 - eps)
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
        debug_iterations: list[dict[str, torch.Tensor]] = []
        if self.debug_diagnostics:
            with torch.no_grad():
                debug_iterations.append(self._iteration_diagnostics(activity, None, valid))
        for _ in range(self.iterations):
            # Keep the competitive settling trajectory intact in the forward
            # pass. Its recurrent Jacobian can be expansive by design, so the
            # default backward pass truncates gradients at each settling step
            # instead of multiplying them through the full positive-feedback
            # loop. ``unrolled`` remains available for controlled comparisons.
            previous_activity = activity.detach() if self.gradient_mode == "truncated" else activity
            population_activity = previous_activity.sum(dim=-1, keepdim=True)
            lateral_input = population_activity - previous_activity
            previous = activity
            activity = torch.sigmoid(
                (drive + self_excitation * previous_activity - inhibition * lateral_input)
                / self.temperature
            )
            if self.debug_diagnostics:
                with torch.no_grad():
                    debug_iterations.append(self._iteration_diagnostics(activity, previous, valid))

        hard = (activity >= 0.5).to(activity.dtype)
        # Parenthesize the zero-valued straight-through residual.  Evaluating
        # ``hard + activity - activity.detach()`` left-to-right can introduce
        # round-off around the hard 0/1 forward values when activity is added
        # before it is subtracted.
        committed = hard + (activity - activity.detach())
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
                if self.debug_diagnostics:
                    selected = normalized.detach()[valid]
                    mass = selection.detach()[valid]
                    logits = selection_logit.detach()[valid]
                    drives = drive.detach()[valid]
                    noise_values = background.detach()[valid]
                    if selected.numel():
                        debug = {
                            "debug_selection_scale": selection.new_tensor(float(scale)),
                            "debug_noise_scale": selection.new_tensor(float(noise)),
                            "debug_temperature": selection.new_tensor(self.temperature),
                            "debug_self_excitation": F.softplus(
                                self.self_excitation_logit.detach()
                            ),
                            "debug_inhibition": F.softplus(self.inhibition_logit.detach()),
                            "debug_selection_mass_min": mass.min(),
                            "debug_selection_mass_max": mass.max(),
                            "debug_selection_mass_zero_frac": (mass <= 0).float().mean(),
                            "debug_selection_prob_min": selected.min(),
                            "debug_selection_prob_max": selected.max(),
                            "debug_selection_prob_lt_1e4": (selected < 1e-4).float().mean(),
                            "debug_selection_prob_lt_1e3": (selected < 1e-3).float().mean(),
                            "debug_selection_prob_gt_099": (selected > 0.99).float().mean(),
                            "debug_selection_nonfinite_frac": (~torch.isfinite(selected))
                            .float()
                            .mean(),
                            "debug_selection_logit_absmax": logits.abs().max(),
                            "debug_drive_absmax": drives.abs().max(),
                            "debug_drive_rms": drives.square().mean().sqrt(),
                            "debug_drive_nonfinite_frac": (~torch.isfinite(drives)).float().mean(),
                            "debug_noise_rms": noise_values.square().mean().sqrt(),
                            "debug_confidence_min": confidence_logit.detach()[valid].min(),
                            "debug_confidence_max": confidence_logit.detach()[valid].max(),
                        }
                        for iteration, metrics in enumerate(debug_iterations):
                            for metric, value in metrics.items():
                                debug[f"debug_iter{iteration}_{metric}"] = value
                    else:
                        zero = selection.new_zeros(())
                        debug = {
                            "debug_selection_scale": zero,
                            "debug_noise_scale": zero,
                            "debug_temperature": zero,
                            "debug_self_excitation": zero,
                            "debug_inhibition": zero,
                            "debug_selection_mass_min": zero,
                            "debug_selection_mass_max": zero,
                            "debug_selection_mass_zero_frac": zero,
                            "debug_selection_prob_min": zero,
                            "debug_selection_prob_max": zero,
                            "debug_selection_prob_lt_1e4": zero,
                            "debug_selection_prob_lt_1e3": zero,
                            "debug_selection_prob_gt_099": zero,
                            "debug_selection_nonfinite_frac": zero,
                            "debug_selection_logit_absmax": zero,
                            "debug_drive_absmax": zero,
                            "debug_drive_rms": zero,
                            "debug_drive_nonfinite_frac": zero,
                            "debug_noise_rms": zero,
                            "debug_confidence_min": zero,
                            "debug_confidence_max": zero,
                        }
                        for iteration, metrics in enumerate(debug_iterations):
                            for metric in metrics:
                                debug[f"debug_iter{iteration}_{metric}"] = zero
                    self._last_diagnostics.update(debug)
        else:
            self._last_diagnostics = {}
        return committed.to(selection.dtype), activity

    def last_diagnostics(self) -> dict[str, torch.Tensor]:
        return dict(self._last_diagnostics)

    @staticmethod
    def _iteration_diagnostics(
        activity: torch.Tensor,
        previous: torch.Tensor | None,
        valid: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        detached = activity.detach()[valid]
        if not detached.numel():
            zero = activity.detach().new_zeros(())
            return {
                "mean": zero,
                "min": zero,
                "max": zero,
                "low_frac": zero,
                "high_frac": zero,
                "nonfinite_frac": zero,
                **({"delta_rms": zero, "delta_max": zero} if previous is not None else {}),
            }
        result = {
            "mean": detached.mean(),
            "min": detached.min(),
            "max": detached.max(),
            "low_frac": (detached < 0.01).float().mean(),
            "high_frac": (detached > 0.99).float().mean(),
            "nonfinite_frac": (~torch.isfinite(detached)).float().mean(),
        }
        if previous is not None:
            delta = (detached - previous.detach()[valid]).abs()
            result["delta_rms"] = delta.square().mean().sqrt()
            result["delta_max"] = delta.max()
        return result
