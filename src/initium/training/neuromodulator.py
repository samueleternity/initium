"""Detached, bounded learning-rate control for optimizer parameter groups."""

from __future__ import annotations

import math
import random


class Neuromodulator:
    """Track scalar training signals and produce rate-limited group multipliers.

    The controller is deliberately pure Python: ``observe`` accepts already
    synchronized scalar values and never reads tensors or participates in autograd.
    """

    def __init__(
        self,
        groups: list[str],
        r: float = 2.0,
        ema_fast: float = 0.9,
        ema_slow: float = 0.99,
        update_every: int = 50,
        slew: float = 0.05,
        mode: str = "off",
        seed: int = 0,
    ):
        if r < 1.0 or update_every < 1 or not 0.0 <= slew:
            raise ValueError("range must be >= 1, update_every >= 1, and slew >= 0")
        if mode not in {"off", "lr", "plasticity", "gate", "responsiveness", "consolidation", "random"}:
            raise ValueError(f"unknown neuromodulator mode: {mode}")
        self.groups = tuple(dict.fromkeys(groups))
        self.r = float(r)
        self.ema_fast_decay = float(ema_fast)
        self.ema_slow_decay = float(ema_slow)
        self.update_every = int(update_every)
        self.slew = float(slew)
        self.mode = mode
        self.mult = {group: 1.0 for group in self.groups}
        self.loss_fast: float | None = None
        self.loss_slow: float | None = None
        self.loss_var: float | None = None
        self.loss_observations = 0
        self.gn_ema: float | None = None
        self.gn_slow: float | None = None
        self.clipped_ema: float | None = None
        self.pred_error_ema: float | None = None
        self.pred_error_var: float | None = None
        self.step = 0
        self.rng = random.Random(int(seed))

    @staticmethod
    def _ema(previous: float | None, value: float, decay: float) -> float:
        return value if previous is None else decay * previous + (1.0 - decay) * value

    def observe(self, loss, grad_norm, clipped, pred_error=None):
        loss = float(loss)
        grad_norm = float(grad_norm)
        if math.isfinite(loss):
            if self.loss_fast is not None:
                residual = loss - self.loss_fast
                self.loss_var = self._ema(
                    self.loss_var, residual * residual, self.ema_slow_decay
                )
            else:
                self.loss_var = 0.0
            self.loss_fast = self._ema(self.loss_fast, loss, self.ema_fast_decay)
            self.loss_slow = self._ema(self.loss_slow, loss, self.ema_slow_decay)
            self.loss_observations += 1
        if math.isfinite(grad_norm):
            self.gn_ema = self._ema(self.gn_ema, grad_norm, self.ema_fast_decay)
            self.gn_slow = self._ema(self.gn_slow, grad_norm, self.ema_slow_decay)
        self.clipped_ema = self._ema(self.clipped_ema, float(bool(clipped)), self.ema_slow_decay)
        if pred_error is not None and math.isfinite(float(pred_error)):
            value = float(pred_error)
            if self.pred_error_ema is not None:
                residual = value - self.pred_error_ema
                self.pred_error_var = self._ema(
                    self.pred_error_var, residual * residual, self.ema_slow_decay
                )
            else:
                self.pred_error_var = 0.0
            self.pred_error_ema = self._ema(
                self.pred_error_ema, value, self.ema_fast_decay
            )

    def pred_error_z(self, value: float) -> float:
        if self.pred_error_ema is None or self.pred_error_var is None:
            return 0.0
        return (float(value) - self.pred_error_ema) / max(math.sqrt(self.pred_error_var), 1e-8)

    def _lr_target(self) -> float:
        if self.loss_fast is None or self.loss_slow is None or self.gn_ema is None:
            return 1.0
        progress = (self.loss_slow - self.loss_fast) / max(abs(self.loss_slow), 1e-8)
        unstable = (
            self.gn_slow is not None and self.gn_slow > 0
            and self.gn_ema > 2.0 * self.gn_slow
        ) or (self.clipped_ema is not None and self.clipped_ema > 0.2)
        if unstable:
            return 1.0 / self.r
        if progress < 0.005:
            return self.r
        return 1.0

    def update(self, step: int, lesson_event: bool = False, eval_signals=None):
        self.step = int(step)
        if self.mode == "off" or self.step % self.update_every:
            return self.multipliers()
        signals = eval_signals or {}
        targets = {group: 1.0 for group in self.groups}
        if self.mode == "random":
            for group in self.groups:
                targets[group] = math.exp(self.rng.uniform(-math.log(self.r), math.log(self.r)))
        elif self.mode == "lr":
            target = self._lr_target()
            targets = {group: target for group in self.groups}
        elif self.mode == "plasticity":
            target = self._lr_target()
            for group in self.groups:
                targets[group] = target
            if "backbone" in targets and lesson_event and signals.get("retention_drop", 0.0) > 0:
                targets["backbone"] = 1.0 / self.r
            dependency = signals.get("workspace_dependency_drop")
            if dependency is not None:
                if "workspace" in targets:
                    targets["workspace"] = self.r if dependency < 0.05 else 1.0 / self.r
                if "modulation" in targets:
                    targets["modulation"] = self.r if dependency < 0.05 else 1.0 / self.r
            if "moe" in targets:
                if signals.get("moe_capacity_drop_frac", 0.0) > 0.2:
                    targets["moe"] = 1.0 / self.r
                elif signals.get("moe_cv_load", 0.0) > 0.5:
                    targets["moe"] = self.r
        elif self.mode == "consolidation":
            if "backbone" in targets:
                targets["backbone"] = self.r
            for group in ("modulation", "workspace", "task_head"):
                if group in targets:
                    targets[group] = 1.0 / self.r
        log_min, log_max = -math.log(self.r), math.log(self.r)
        for group, target in targets.items():
            target = min(self.r, max(1.0 / self.r, target))
            delta = min(self.slew, max(-self.slew, target - self.mult[group]))
            candidate = min(self.r, max(1.0 / self.r, self.mult[group] + delta))
            self.mult[group] = math.exp(min(log_max, max(log_min, math.log(candidate))))
        return self.multipliers()

    def multipliers(self) -> dict[str, float]:
        if self.mode == "off":
            return {group: 1.0 for group in self.groups}
        return dict(self.mult)

    def diagnostics(self) -> dict[str, float | None]:
        return {
            "loss_fast": self.loss_fast,
            "loss_slow": self.loss_slow,
            "loss_var": self.loss_var,
            "loss_observations": self.loss_observations,
            "gn_ema": self.gn_ema,
            "clipped_ema": self.clipped_ema,
            "pred_error_ema": self.pred_error_ema,
            "pred_error_var": getattr(self, "pred_error_var", None),
        }

    def state_dict(self) -> dict:
        return {
            "groups": self.groups,
            "r": self.r,
            "ema_fast_decay": self.ema_fast_decay,
            "ema_slow_decay": self.ema_slow_decay,
            "update_every": self.update_every,
            "slew": self.slew,
            "mode": self.mode,
            "mult": dict(self.mult),
            "loss_fast": self.loss_fast,
            "loss_slow": self.loss_slow,
            "gn_ema": self.gn_ema,
            "gn_slow": self.gn_slow,
            "clipped_ema": self.clipped_ema,
            "pred_error_ema": self.pred_error_ema,
            "step": self.step,
            "rng_state": self.rng.getstate(),
        }

    def load_state_dict(self, state: dict):
        if tuple(state.get("groups", self.groups)) != self.groups:
            raise ValueError("neuromodulator checkpoint groups do not match configured groups")
        for key in ("mult", "loss_fast", "loss_slow", "loss_var", "loss_observations", "gn_ema", "gn_slow", "clipped_ema", "pred_error_ema", "pred_error_var", "step"):
            if key in state:
                setattr(self, key, dict(state[key]) if key == "mult" else state[key])
        if "rng_state" in state:
            self.rng.setstate(state["rng_state"])
