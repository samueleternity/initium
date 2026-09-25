import torch

from initium.memory_manipulation.stochastic_write_head_v2 import StochasticWriteHead


def test_kl_path_reaches_shared_mean_parameters():
    head = StochasticWriteHead(2, 2, sample=True)
    x = torch.ones(1, 2)
    head(x)
    loss, _ = head.pop_kl()
    loss.backward()
    assert head.mu_transform.weight.grad is not None
    assert torch.isfinite(head.mu_transform.weight.grad).all()


def test_kl_head_gradcheck_lite():
    head = StochasticWriteHead(2, 2, sample=True).double()
    head.eval()
    head.train()

    def kl_for_input(x):
        head._kl_terms = []
        head._clamp_terms = []
        head(x)
        return head.pop_kl()[0]

    x = torch.tensor([[0.2, -0.3]], dtype=torch.double, requires_grad=True)
    assert torch.autograd.gradcheck(kl_for_input, (x,), eps=1e-6, atol=1e-4, rtol=1e-3)
