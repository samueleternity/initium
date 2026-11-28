"""Composable language, JEPA-style, transition, and logic losses."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from initium.data.language.config import LossWeights


def causal_lm_loss(logits: torch.Tensor, token_ids: torch.Tensor, *, ignore_index: int = -100):
    """Next-token loss at every unmasked document position (B,T,V; B,T)."""
    if logits.size(1) < 2:
        return logits.sum() * 0.0
    targets = token_ids[:, 1:].reshape(-1)
    if not targets.ne(ignore_index).any():
        return logits.sum() * 0.0
    return F.cross_entropy(logits[:, :-1].reshape(-1, logits.size(-1)),
                           targets, ignore_index=ignore_index)


def same_state_loss(first: torch.Tensor, second: torch.Tensor, *, variance_weight: float = 1.0,
                    covariance_weight: float = 0.04, eps: float = 1e-4) -> torch.Tensor:
    """View agreement plus VICReg variance/covariance anti-collapse regularization."""
    if first.shape != second.shape or first.ndim != 2:
        raise ValueError("paired state embeddings must have identical (batch, dim) shapes")
    invariance = F.mse_loss(first, second)
    z = torch.cat((first, second), dim=0)
    z = z - z.mean(dim=0)
    std = torch.sqrt(z.var(dim=0, unbiased=False) + eps)
    variance = F.relu(1.0 - std).mean()
    if z.size(0) < 2:
        covariance = z.new_zeros(())
    else:
        cov = z.T @ z / (z.size(0) - 1)
        off_diag = cov - torch.diag(torch.diagonal(cov))
        covariance = off_diag.square().sum() / z.size(1)
    return invariance + variance_weight * variance + covariance_weight * covariance


def transition_loss(predicted_future: torch.Tensor, target_future: torch.Tensor,
                    *, margin: float = 1.0) -> torch.Tensor:
    """Action-conditioned latent prediction with a negative-state separation term."""
    if predicted_future.shape != target_future.shape or predicted_future.ndim != 2:
        raise ValueError("transition embeddings must have identical (batch, dim) shapes")
    positive = F.smooth_l1_loss(predicted_future, target_future)
    if predicted_future.size(0) < 2:
        return positive
    negatives = target_future.roll(1, dims=0)
    pos_dist = (predicted_future - target_future).norm(dim=-1)
    neg_dist = (predicted_future - negatives).norm(dim=-1)
    valid = (target_future != negatives).any(dim=-1)
    contrast = F.relu(margin + pos_dist - neg_dist)
    return positive + (contrast[valid].mean() if valid.any() else positive.new_zeros(()))


def logic_state_loss(logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor | None = None):
    """Supervise discrete state/update/contradiction labels."""
    losses = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1),
                             reduction="none")
    if mask is None:
        return losses.mean()
    weights = mask.to(losses.dtype).reshape(-1)
    return (losses * weights).sum() / weights.sum().clamp_min(1)


@dataclass
class LossBundle:
    text: torch.Tensor | None = None
    same_state: torch.Tensor | None = None
    transition: torch.Tensor | None = None
    logic: torch.Tensor | None = None
    dialogue: torch.Tensor | None = None
    auxiliary: torch.Tensor | None = None

    def total(self, weights: LossWeights = LossWeights()) -> torch.Tensor:
        available = {"text": self.text, "same_state": self.same_state,
                     "transition": self.transition, "logic": self.logic,
                     "dialogue": self.dialogue}
        terms = [value * getattr(weights, name) for name, value in available.items()
                 if value is not None and getattr(weights, name) > 0]
        if self.auxiliary is not None:
            terms.append(self.auxiliary)
        if not terms:
            raise ValueError("no enabled objective losses were supplied")
        return torch.stack(terms).sum()
