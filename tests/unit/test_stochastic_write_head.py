import torch
from initium.memory_manipulation.stochastic_write_head_v2 import (
    StochasticWriteHead,
    get_prior_state,
    load_prior_state,
)


def test_deterministic_forward_has_no_kl():
    head = StochasticWriteHead(3, 2, sample=False)
    x = torch.randn(4, 3)
    torch.testing.assert_close(head(x), head.mu_transform(x), rtol=0, atol=0)
    loss, _ = head.pop_kl()
    assert loss.item() == 0


def test_phase_one_and_general_kl_are_same_for_standard_prior():
    head = StochasticWriteHead(2, 2, sample=True)
    with torch.no_grad():
        head.mu_transform.weight.zero_()
        head.mu_transform.bias.copy_(torch.tensor([0.2, -0.4]))
        head.logvar_transform.weight.zero_()
        head.logvar_transform.bias.copy_(torch.tensor([-0.3, 0.1]))
    head.train()
    head(torch.zeros(1, 2))
    loss, _ = head.pop_kl()
    mu = head.mu_transform(torch.zeros(1, 2))
    logvar = head.logvar_transform(torch.zeros(1, 2))
    expected = (0.5 * (logvar.exp() + mu.square() - 1 - logvar)).sum(-1).mean()
    torch.testing.assert_close(loss, expected)
    loss.backward()
    assert head.prior_mu.grad is None
    assert head.prior_logvar.grad is None


def test_learned_prior_kl_matches_diagonal_gaussian_formula():
    head = StochasticWriteHead(2, 2, sample=True)
    with torch.no_grad():
        head.mu_transform.weight.zero_()
        head.mu_transform.bias.copy_(torch.tensor([0.5, -0.25]))
        head.logvar_transform.weight.zero_()
        head.logvar_transform.bias.copy_(torch.tensor([0.2, -0.4]))
        head.prior_mu.copy_(torch.tensor([0.1, 0.3]))
        head.prior_logvar.copy_(torch.tensor([-0.2, 0.5]))
    head(torch.zeros(1, 2))
    actual, _ = head.pop_kl()
    mu = head.mu_transform(torch.zeros(1, 2))
    logvar = head.logvar_transform(torch.zeros(1, 2))
    expected = (
        (
            0.5
            * (
                head.prior_logvar
                - logvar
                + (logvar.exp() + (mu - head.prior_mu).square()) / head.prior_logvar.exp()
                - 1
            )
        )
        .sum(-1)
        .mean()
    )
    torch.testing.assert_close(actual, expected)


def test_snapshot_uses_controlled_sample_mean_and_clamped_variance():
    head = StochasticWriteHead(2, 2)
    samples = torch.tensor([[1.0, 2.0], [3.0, 6.0], [5.0, 10.0]])
    head._recent_writes = [samples]
    result = head.update_prior_snapshot(19, min_logvar=-2, max_logvar=2)
    expected_var = samples.var(dim=0, unbiased=False).clamp(min=1e-6)
    torch.testing.assert_close(head.prior_mu, samples.mean(dim=0))
    torch.testing.assert_close(head.prior_logvar, expected_var.log().clamp(-2, 2))
    assert result["n_samples"] == 3
    assert head.last_snapshot_step == 19


def test_snapshot_refit_and_free_bits():
    head = StochasticWriteHead(2, 2, sample=True)
    head.train()
    with torch.no_grad():
        head.mu_transform.weight.zero_()
        head.mu_transform.bias.zero_()
        head.logvar_transform.weight.zero_()
        head.logvar_transform.bias.zero_()
    for _ in range(3):
        head(torch.zeros(2, 2))
    loss, diag = head.pop_kl(free_bits=0.2)
    assert abs(loss.item() - 0.4) < 1e-6
    assert diag["kl_mean"] == 0
    snapshot = head.update_prior_snapshot(11, min_logvar=-6, max_logvar=6)
    assert snapshot["n_samples"] == 6
    assert head.last_snapshot_step == 11
    assert torch.isfinite(head.prior_logvar).all()
    state = get_prior_state([head])
    restored = StochasticWriteHead(2, 2)
    load_prior_state([restored], state)
    torch.testing.assert_close(restored.prior_mu, head.prior_mu)
    torch.testing.assert_close(restored.prior_logvar, head.prior_logvar)
    assert restored.last_snapshot_step == 11
