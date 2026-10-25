import pytest
import torch

from initium.mamba_controller.numerical_stability import (
    MAMBA_MIXER_RESIDUAL_RMS_LIMIT,
    limit_mixer_residual_rms,
)


def test_mixer_residual_limit_preserves_values_below_bound():
    residual = torch.tensor([[1.0, -2.0, 3.0], [0.0, 0.5, -0.25]])

    actual = limit_mixer_residual_rms(residual)

    torch.testing.assert_close(actual, residual)
    assert actual.dtype == torch.float32


def test_mixer_residual_limit_bounds_each_token_and_keeps_large_values_finite():
    residual = torch.tensor(
        [[1.0e30, -1.0e30, 1.0e30], [1.0e10, 0.0, 0.0]],
        dtype=torch.float32,
        requires_grad=True,
    )

    actual = limit_mixer_residual_rms(residual)
    token_rms = actual.square().mean(dim=-1).sqrt()

    assert torch.isfinite(actual).all()
    assert torch.all(token_rms <= MAMBA_MIXER_RESIDUAL_RMS_LIMIT)
    actual.sum().backward()
    assert residual.grad is not None
    assert torch.isfinite(residual.grad).all()


def test_mixer_residual_limit_promotes_half_precision_before_scaling():
    residual = torch.tensor([[60000.0, 60000.0]], dtype=torch.float16)

    actual = limit_mixer_residual_rms(residual)

    assert actual.dtype == torch.float32
    assert torch.isfinite(actual).all()
    assert float(actual.abs().max()) <= MAMBA_MIXER_RESIDUAL_RMS_LIMIT


@pytest.mark.parametrize("limit", [0.0, -1.0, float("inf"), float("nan")])
def test_mixer_residual_limit_rejects_invalid_limit(limit):
    with pytest.raises(ValueError, match="finite and > 0"):
        limit_mixer_residual_rms(torch.ones(1, 2), limit=limit)


def test_mixer_residual_limit_does_not_hide_nonfinite_inputs():
    residual = torch.tensor([[float("nan"), 1.0]])

    actual = limit_mixer_residual_rms(residual)

    assert torch.isnan(actual[0, 0])
