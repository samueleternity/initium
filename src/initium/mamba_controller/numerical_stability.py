"""Numerical guards shared by the Mamba controller and backbone paths.

Mamba blocks are pre-normalized, but that does not bound the mixer output.
An oversized mixer residual can grow the residual stream across blocks while
the final normalization keeps the externally visible output finite. Keep the
guard at the common mixer-to-residual boundary so every Mamba variant uses
the same policy.
"""

from __future__ import annotations

import math

import torch

# A token-wise RMS limit leaves ordinary activations unchanged while preventing
# a single learned mixer branch from overwhelming the residual stream. This
# is intentionally an architectural safety bound, not a training-time clamp
# on the SSM state or a replacement for gradient clipping.
MAMBA_MIXER_RESIDUAL_RMS_LIMIT = 10.0


def limit_mixer_residual_rms(
    residual: torch.Tensor,
    *,
    limit: float = MAMBA_MIXER_RESIDUAL_RMS_LIMIT,
) -> torch.Tensor:
    """Radially limit each token's FP32 mixer residual by its feature RMS.

    Uses max-scaled RMS arithmetic to avoid squaring large finite FP32 values.
    Non-finite inputs are deliberately not sanitized; they remain visible to
    the existing anomaly/numeric-trace machinery.
    """
    if not math.isfinite(limit) or limit <= 0.0:
        raise ValueError(f"Mamba mixer residual RMS limit must be finite and > 0, got {limit!r}")
    value = residual.float()
    max_abs = value.abs().amax(dim=-1, keepdim=True)
    safe_max = max_abs.clamp_min(torch.finfo(value.dtype).tiny)
    normalized = value / safe_max
    scaled_rms = normalized.square().mean(dim=-1, keepdim=True).sqrt()
    scale = (limit / safe_max) / scaled_rms.clamp_min(torch.finfo(value.dtype).tiny)
    scale = scale.clamp(max=1.0)
    return value * scale
