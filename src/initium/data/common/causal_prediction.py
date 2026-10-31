"""Dataset-agnostic causal prediction helpers for split-graph models."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalInputPredictionHead(nn.Module):
    """Predict the next observed input vector from a causal backbone state.

    Supports dense, sparse one-hot, and concatenated multimodal inputs without
    assuming categorical token IDs.
    """

    def __init__(self, hidden_size: int, input_size: int):
        super().__init__()
        if hidden_size < 1 or input_size < 1:
            raise ValueError("hidden_size and input_size must be positive")
        self.projection = nn.Linear(hidden_size, input_size)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        with torch.autocast(device_type=features.device.type, enabled=False):
            return F.linear(
                features.float(),
                self.projection.weight.float(),
                self.projection.bias.float() if self.projection.bias is not None else None,
            )


def masked_input_prediction_loss(
    predictions: torch.Tensor,
    input_sequence: torch.Tensor,
    valid_mask: torch.Tensor,
) -> torch.Tensor:
    """Scale-normalized MSE for predicting observed input vectors one step ahead."""
    if predictions.ndim != 3 or input_sequence.ndim != 3 or valid_mask.ndim != 2:
        raise ValueError("predictions/input_sequence/valid_mask must have ranks 3/3/2")
    targets = input_sequence[:, 1:].float()
    if predictions.shape != targets.shape or valid_mask.shape != input_sequence.shape[:2]:
        raise ValueError("input prediction tensors have incompatible batch/time/feature shapes")
    valid = valid_mask[:, 1:].bool()
    residual_energy = (predictions.float() - targets).square().sum(dim=-1)
    target_energy = targets.square().sum(dim=-1).clamp_min(1.0)
    selected_loss = (residual_energy / target_energy)[valid]
    return selected_loss.sum() / valid.sum().clamp_min(1)


def causal_input_prediction_error(
    predictions: torch.Tensor,
    input_sequence: torch.Tensor,
    valid_mask: torch.Tensor,
) -> torch.Tensor:
    """Return detached, bounded relative error aligned to observed input t.

    ``predictions[:, t-1]`` predicts ``input_sequence[:, t]``. The normalized
    residual is invariant to input width and is zeroed for padding/invalid rows.
    """
    if predictions.ndim != 3 or input_sequence.ndim != 3 or valid_mask.ndim != 2:
        raise ValueError("predictions/input_sequence/valid_mask must have ranks 3/3/2")
    targets = input_sequence[:, 1:].float()
    if predictions.shape != targets.shape or valid_mask.shape != input_sequence.shape[:2]:
        raise ValueError("input prediction tensors have incompatible batch/time/feature shapes")
    prediction_values = predictions.float()
    residual = torch.linalg.vector_norm(prediction_values - targets, dim=-1)
    scale = (
        torch.linalg.vector_norm(prediction_values, dim=-1)
        + torch.linalg.vector_norm(targets, dim=-1)
    ).clamp_min(1e-6)
    relative_error = (residual / scale).clamp(0.0, 1.0).detach()
    relative_error = relative_error.masked_fill(~valid_mask[:, 1:].bool(), 0.0)
    return torch.cat(
        [relative_error.new_zeros((relative_error.shape[0], 1)), relative_error], dim=1
    )
