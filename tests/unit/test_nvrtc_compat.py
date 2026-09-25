import torch

from initium.memory_manipulation.nvrtc_compat import patch_prod_jiterator


def test_patch_is_idempotent_and_cpu_parity():
    original_prod, original_cumprod = torch.prod, torch.cumprod
    patch_prod_jiterator()
    patched_prod, patched_cumprod = torch.prod, torch.cumprod
    patch_prod_jiterator()
    assert torch._nvrtc_compat_patched is True
    x = torch.rand(2, 3, 4)
    torch.testing.assert_close(patched_prod(x, dim=1), original_prod(x, dim=1))
    torch.testing.assert_close(patched_cumprod(x, dim=1), original_cumprod(x, dim=1))
    # The CUDA log-space branch cannot be exercised by the CPU CI runner.
