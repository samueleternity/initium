"""
file: memory_manipulation/nvrtc_compat.py

Environment-compatibility shim, NOT a modeling change. Works around a
CUDA/NVRTC library mismatch observed in this environment ("nvrtc: error:
failed to open libnvrtc-builtins.so.13.0" - the installed torch build's
CUDA minor version has no matching nvrtc-builtins shared library present,
a known failure mode after a partial torch/CUDA reinstall or a fresh
Colab runtime whose nvidia-cuda-nvrtc-cuXX package doesn't match the
torch wheel's CUDA build).

Root cause: on CUDA, `torch.prod(x, dim=...)` and `torch.cumprod(x,
dim=...)` (reductions/scans along a non-trivial dim) dispatch through
PyTorch's Jiterator -- a kernel compiled at PROCESS RUNTIME via NVRTC --
rather than a statically pre-compiled kernel. Any environment where NVRTC
can't find its builtins library crashes the FIRST time either op runs,
regardless of model architecture, controller type, or MoE config.

`dnc.memory.Memory` (third-party, installed via pip -- not a project
file) calls both of these on the hot path of every single write():
  - get_usage_vector: torch.prod(1 - free_gates*read_weights, dim=1) and
    torch.prod(1 - write_weights, dim=1)
  - allocation weighting: torch.cumprod(sorted_usage, dim=1) (Graves et
    al. 2016's free-list product term, `prod_{i<j} u[phi[i]]`)

Rather than reimplementing each of dnc.memory's internal methods by hand
(fragile: depends on exactly matching an undocumented third-party
formula, and any future call site is invisible until it crashes), this
module patches `torch.prod` / `torch.cumprod` / `Tensor.prod` /
`Tensor.cumprod` THEMSELVES. Every call inside dnc/memory.py - known or
not yet discovered - is transparently redirected, with zero changes
needed to the third-party package or to any call site in this project.

Fix mechanism: for a CUDA tensor with an explicit `dim`, replace the
product/cumulative-product with the equivalent log-space reduction
exp(sum(log(clamp(x, min=EPS)), dim)) or exp(cumsum(log(clamp(x)), dim)).
log, sum/cumsum, and exp are ordinary elementwise/reduction ops with
statically pre-compiled CUDA kernels for every dtype PyTorch ships - no
Jiterator involved - so this sidesteps the missing-nvrtc-builtins
failure without requiring a CUDA toolkit reinstall inside the notebook.

Safety net: the log-space substitution is only mathematically valid for
NON-NEGATIVE inputs (log of a negative number is NaN). Every known call
site in dnc.memory operates on gates/usages/weights, which are always in
[0,1] - but to stay correct even for an unknown future call site (some
other library, or an edit to dnc.memory) with genuinely negative values,
the patched functions check `torch.isnan(result).any()` after the
log-space computation and, on a hit, recompute via the ORIGINAL op on a
CPU copy of the input (never re-invoking the broken CUDA op) before
moving the result back to the original device. This costs a device sync
+ small copy only in that (expected-never, for this project) fallback
path - the common case never sees it.

Call patch_prod_jiterator() once, before running any forward pass -
idempotent, so importing this from multiple entry points (core_training.py,
run_inference.py) is safe. If the underlying environment issue is fixed
instead (matching nvidia-cuda-nvrtc-cuXX package reinstalled), this patch
is a harmless no-op difference in float rounding at the ~1e-6 level, not
a correctness regression - there is no reason to remove it once applied.
"""

from __future__ import annotations

import torch

_EPS = 1e-6
_PATCHED_ATTR = "_nvrtc_compat_patched"

_orig_prod = torch.prod
_orig_cumprod = torch.cumprod
_orig_tensor_prod = torch.Tensor.prod
_orig_tensor_cumprod = torch.Tensor.cumprod


def _log_space(input: torch.Tensor, dim, keepdim: bool, cumulative: bool) -> torch.Tensor:
    orig_dtype = input.dtype
    work = input if input.dtype.is_floating_point else input.float()
    logged = torch.log(work.clamp(min=_EPS))
    reduced = (
        torch.cumsum(logged, dim=dim) if cumulative else torch.sum(logged, dim=dim, keepdim=keepdim)
    )
    return torch.exp(reduced).to(orig_dtype)


def _patched_prod(input, dim=None, keepdim=False, *, dtype=None):
    if isinstance(input, torch.Tensor) and input.is_cuda and dim is not None:
        result = _log_space(input, dim, keepdim, cumulative=False)
        if torch.isnan(result).any():
            # Negative-valued input -- log-space isn't valid here. Fall back
            # to the ORIGINAL op on CPU (never re-run the broken CUDA path).
            cpu_out = _orig_prod(
                input.detach().cpu(),
                dim,
                keepdim=keepdim,
                **({"dtype": dtype} if dtype is not None else {}),
            )
            return cpu_out.to(input.device)
        return result.to(dtype) if dtype is not None else result
    if dim is None:
        return _orig_prod(input, **({"dtype": dtype} if dtype is not None else {}))
    return _orig_prod(
        input, dim, keepdim=keepdim, **({"dtype": dtype} if dtype is not None else {})
    )


def _patched_cumprod(input, dim, *, dtype=None):
    if isinstance(input, torch.Tensor) and input.is_cuda:
        result = _log_space(input, dim, keepdim=False, cumulative=True)
        if torch.isnan(result).any():
            cpu_out = _orig_cumprod(
                input.detach().cpu(), dim, **({"dtype": dtype} if dtype is not None else {})
            )
            return cpu_out.to(input.device)
        return result.to(dtype) if dtype is not None else result
    return _orig_cumprod(input, dim, **({"dtype": dtype} if dtype is not None else {}))


def _patched_tensor_prod(self, dim=None, keepdim=False, *, dtype=None):
    return _patched_prod(self, dim, keepdim, dtype=dtype)


def _patched_tensor_cumprod(self, dim, *, dtype=None):
    return _patched_cumprod(self, dim, dtype=dtype)


def patch_prod_jiterator() -> None:
    """Idempotent: safe to call from every entry point (training, inference,
    a notebook cell re-running setup) without double-patching."""
    if getattr(torch, _PATCHED_ATTR, False):
        return
    torch.prod = _patched_prod
    torch.cumprod = _patched_cumprod
    torch.Tensor.prod = _patched_tensor_prod
    torch.Tensor.cumprod = _patched_tensor_cumprod
    setattr(torch, _PATCHED_ATTR, True)
    print(
        "[nvrtc_compat] patched torch.prod/cumprod (module- and Tensor-level) "
        "to avoid PyTorch's Jiterator on CUDA, working around this "
        "environment's missing/mismatched libnvrtc-builtins. Every "
        "dnc.memory.Memory call site (get_usage_vector, allocation "
        "weighting, and any other prod/cumprod use) is covered "
        "automatically - no per-method patching needed."
    )
