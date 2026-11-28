"""Minimal reusable optimizer-step orchestration for interleaved objectives.

The dataset-specific collators supply objective names and batches; the model
adapter supplies forward computations returning ``LossBundle``. Keeping that
boundary explicit lets the established episodic loop remain unchanged while
the broad-language pipeline evolves independently.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Callable, Iterable

import torch
from torch import nn

from initium.data.language.config import LossWeights
from initium.data.language.objectives import LossBundle


@dataclass
class StepResult:
    step: int
    loss: float
    components: dict[str, float]


class InterleavedTrainer:
    """Shared optimizer loop with LR modulation, gradient gates, and consolidation."""

    def __init__(self, model: nn.Module, optimizer: torch.optim.Optimizer, *,
                 weights: LossWeights = LossWeights(), grad_clip: float | None = 1.0,
                 neuromodulator=None, base_lr: float = 3e-4, min_lr: float = 3e-5,
                 total_steps: int = 130_000, warmup_steps: int = 1_000,
                 loss_outlier_gating: bool = False, gate_kappa: float = 2.0,
                 consolidation_lambda: float = 0.1, consolidation_enabled: bool = False,
                 gradient_gating: bool = False, modulation_alpha_warmup_steps: int = 3000):
        self.model, self.optimizer = model, optimizer
        self.weights, self.grad_clip = weights, grad_clip
        self.neuromodulator = neuromodulator
        self.base_lr, self.min_lr = float(base_lr), float(min_lr)
        self.total_steps, self.warmup_steps = int(total_steps), int(warmup_steps)
        self.loss_outlier_gating, self.gate_kappa = loss_outlier_gating, float(gate_kappa)
        self.consolidation_lambda = float(consolidation_lambda)
        self.consolidation_enabled = consolidation_enabled
        self.gradient_gating = gradient_gating
        self.modulation_alpha_warmup_steps = max(0, int(modulation_alpha_warmup_steps))
        self.step_index = 0
        self._pending_lesson_event = False
        self._pending_eval_signals = {}
        self.consolidation_anchor = ({
            name: value.detach().float().clone()
            for name, value in model.named_parameters() if value.requires_grad
        } if consolidation_enabled else {})
        self.consolidation_importance = ({
            name: torch.zeros_like(value, dtype=torch.float32)
            for name, value in model.named_parameters() if value.requires_grad
        } if consolidation_enabled else {})

    def _scheduled_lr(self, step: int) -> float:
        if self.warmup_steps and step < self.warmup_steps:
            return self.base_lr * (step + 1) / self.warmup_steps
        progress = min(max(step - self.warmup_steps, 0), max(self.total_steps - self.warmup_steps, 1))
        progress /= max(self.total_steps - self.warmup_steps, 1)
        return self.min_lr + 0.5 * (self.base_lr - self.min_lr) * (1 + math.cos(math.pi * progress))

    def _set_learning_rates(self, step: int):
        scale = self.neuromodulator.multipliers() if self.neuromodulator else {}
        base = self._scheduled_lr(step)
        for group in self.optimizer.param_groups:
            group["lr"] = (base * float(group.get("lr_multiplier", 1.0))
                           * scale.get(group.get("nm_group", "backbone"), 1.0))
        modulation = getattr(getattr(self.model, "backbone", None), "modulation_head", None)
        if modulation is not None:
            modulation.alpha_scale = min(1.0, step / max(1, self.modulation_alpha_warmup_steps))

    def _consolidation_penalty(self):
        if not self.neuromodulator or not self.consolidation_enabled:
            return next(self.model.parameters()).new_zeros(())
        multipliers = self.neuromodulator.multipliers()
        penalty = next(self.model.parameters()).new_zeros((), dtype=torch.float32)
        for name, parameter in self.model.named_parameters():
            if name not in self.consolidation_anchor:
                continue
            group = self._parameter_group(name)
            factor = multipliers.get(group, 1.0)
            if self.neuromodulator.mode != "consolidation":
                factor *= self.neuromodulator.r if group == "backbone" else 1.0 / self.neuromodulator.r
            penalty = penalty + factor * self.consolidation_importance[name].to(parameter.device).mul(
                (parameter.float() - self.consolidation_anchor[name].to(parameter.device)).square()
            ).sum()
        return 0.5 * self.consolidation_lambda * penalty

    @staticmethod
    def _parameter_group(name: str) -> str:
        if ".parallel_workspace." in name or ".workspace." in name:
            return "workspace"
        if ".modulation_head." in name:
            return "modulation"
        if name.startswith(("objective_heads.", "output_projection.")):
            return "task_head"
        return "backbone"

    def refresh_consolidation_anchor(self):
        """Start a new consolidation interval after a checkpoint/eval boundary."""
        self.consolidation_anchor = {
            name: value.detach().float().clone()
            for name, value in self.model.named_parameters() if value.requires_grad
        }
        for value in self.consolidation_importance.values():
            value.zero_()

    def state_dict(self):
        return {"step_index": self.step_index,
                "consolidation_anchor": self.consolidation_anchor,
                "consolidation_importance": self.consolidation_importance,
                "neuromodulator": self.neuromodulator.state_dict() if self.neuromodulator else None,
                "pending_lesson_event": self._pending_lesson_event,
                "pending_eval_signals": self._pending_eval_signals}

    def load_state_dict(self, state):
        self.step_index = int(state["step_index"])
        parameters = dict(self.model.named_parameters())
        self.consolidation_anchor = {
            name: value.to(parameters[name].device)
            for name, value in state["consolidation_anchor"].items()
        }
        self.consolidation_importance = {
            name: value.to(parameters[name].device)
            for name, value in state["consolidation_importance"].items()
        }
        if self.neuromodulator and state.get("neuromodulator"):
            self.neuromodulator.load_state_dict(state["neuromodulator"])
        self._pending_lesson_event = bool(state.get("pending_lesson_event", False))
        self._pending_eval_signals = dict(state.get("pending_eval_signals", {}))

    def step(self, objective_batches: dict[str, object],
             forward_loss: Callable[[str, object], LossBundle], *, lesson_event: bool = False,
             eval_signals: dict | None = None) -> StepResult:
        """Run one optimizer step from any supplied objective batches.

        ``forward_loss(name, batch)`` executes the shared model and relevant
        prediction head(s), returning scalar losses in ``LossBundle``. Batches
        can include multiple objectives, so their losses share one backward
        pass and update the same encoder.
        """
        if not objective_batches:
            raise ValueError("an optimizer step needs at least one objective batch")
        self.model.train()
        self._set_learning_rates(self.step_index)
        self.optimizer.zero_grad(set_to_none=True)
        losses = [forward_loss(name, batch) for name, batch in objective_batches.items()]
        components: dict[str, list[torch.Tensor]] = {}
        for bundle in losses:
            for name in ("text", "same_state", "transition", "logic", "dialogue", "auxiliary"):
                value = getattr(bundle, name)
                if value is not None:
                    components.setdefault(name, []).append(value)
        means = {name: torch.stack(values).mean() for name, values in components.items()}
        total = LossBundle(**means).total(self.weights) + self._consolidation_penalty()
        if not torch.isfinite(total):
            raise FloatingPointError(f"non-finite language objective at step {self.step_index + 1}")
        total.backward()
        params = [parameter for parameter in self.model.parameters() if parameter.grad is not None]
        if self.neuromodulator and self.consolidation_enabled:
            for name, parameter in self.model.named_parameters():
                if name in self.consolidation_importance and parameter.grad is not None:
                    grad = parameter.grad.detach().float()
                    self.consolidation_importance[name].mul_(0.99).addcmul_(grad, grad, value=0.01)
        grad_norm = torch.linalg.vector_norm(torch.stack([
            parameter.grad.detach().float().norm(2) for parameter in params
        ])) if params else total.new_zeros(())
        grad_norm_value = float(grad_norm.detach())
        if not math.isfinite(grad_norm_value):
            self.optimizer.zero_grad(set_to_none=True)
            raise FloatingPointError(f"non-finite gradient norm at step {self.step_index + 1}")
        clipped = self.grad_clip is not None and grad_norm_value > self.grad_clip
        if self.grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
        if self.neuromodulator and self.gradient_gating:
            if (self.neuromodulator.gn_ema is not None
                    and grad_norm_value > self.gate_kappa * max(self.neuromodulator.gn_ema, 1e-8)):
                for parameter in params:
                    parameter.grad.mul_(0.25)
        loss_value = float(total.detach())
        if (self.neuromodulator and self.loss_outlier_gating
                and self.neuromodulator.loss_observations >= 2
                and self.neuromodulator.loss_fast is not None):
            loss_std = math.sqrt(max(self.neuromodulator.loss_var or 0.0, 0.0))
            if loss_value > self.neuromodulator.loss_fast + 3.0 * loss_std:
                for parameter in params:
                    parameter.grad.mul_(0.25)
        self.optimizer.step()
        self.step_index += 1
        if self.neuromodulator:
            if lesson_event:
                self._pending_lesson_event = True
            if eval_signals:
                self._pending_eval_signals.update(eval_signals)
            self.neuromodulator.observe(loss_value, grad_norm_value, clipped)
            self.neuromodulator.update(self.step_index,
                                       lesson_event=self._pending_lesson_event,
                                       eval_signals=self._pending_eval_signals)
            if self.step_index % self.neuromodulator.update_every == 0:
                self._pending_lesson_event = False
                self._pending_eval_signals = {}
        return StepResult(self.step_index, float(total.detach()),
                          {name: float(value.detach()) for name, value in means.items()})

    def train(self, batches: Iterable[dict[str, object]],
              forward_loss: Callable[[str, object], LossBundle], *, max_steps: int | None = None):
        for objective_batches in batches:
            if max_steps is not None and self.step_index >= max_steps:
                break
            yield self.step(objective_batches, forward_loss)
