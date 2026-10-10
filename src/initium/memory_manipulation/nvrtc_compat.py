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

The log-space substitution is valid for NON-NEGATIVE inputs. Every known
call site in dnc.memory operates on gates/usages/weights in [0,1]. The old
implementation checked for NaNs after each product, which forced a CUDA to
CPU synchronization hundreds of times per long sequence; clamping also meant
that check did not detect negative inputs as its comment claimed. The normal
fast path now relies on the known non-negative DNC inputs and performs no
host synchronization. Set INITIUM_NVRTC_COMPAT_STRICT=1 before starting
Python to enable a slower negative-input check and CPU fallback for debugging
other call sites.

Call patch_prod_jiterator() once, before running any forward pass -
idempotent, so importing this from multiple entry points (core_training.py,
run_inference.py) is safe. If the underlying environment issue is fixed
instead (matching nvidia-cuda-nvrtc-cuXX package reinstalled), this patch
is a harmless no-op difference in float rounding at the ~1e-6 level, not
a correctness regression - there is no reason to remove it once applied.
"""

from __future__ import annotations

import os

import torch

_EPS = 1e-6
_PATCHED_ATTR = "_nvrtc_compat_patched"
_STRICT_NEGATIVE_CHECK = os.environ.get("INITIUM_NVRTC_COMPAT_STRICT", "0") == "1"
_PRODUCT_BACKWARD_TRACE: tuple[bool, object | None] = (False, None)
_PRODUCT_INPUT_STATS: dict[str, dict[str, torch.Tensor | int]] = {}

_orig_prod = torch.prod
_orig_cumprod = torch.cumprod
_orig_tensor_prod = torch.Tensor.prod
_orig_tensor_cumprod = torch.Tensor.cumprod


def _log_space(input: torch.Tensor, dim, keepdim: bool, cumulative: bool) -> torch.Tensor:
    orig_dtype = input.dtype
    # DNC gates/usages are non-negative. Accumulate in fp32 even under AMP.
    work = input.float()
    logged = torch.log(work.clamp(min=_EPS))
    reduced = (
        torch.cumsum(logged, dim=dim) if cumulative else torch.sum(logged, dim=dim, keepdim=keepdim)
    )
    return torch.exp(reduced).to(orig_dtype)


def set_product_backward_trace(enabled: bool, reporter=None) -> None:
    """Attach an optional diagnostic hook to patched product outputs."""
    global _PRODUCT_BACKWARD_TRACE
    _PRODUCT_BACKWARD_TRACE = (bool(enabled), reporter)


def _watch_product_output(
    result: torch.Tensor, input: torch.Tensor, operation: str
) -> torch.Tensor:
    enabled, reporter = _PRODUCT_BACKWARD_TRACE
    if enabled and reporter is not None and result.requires_grad:
        result.register_hook(lambda grad: reporter(operation, grad))
        if input.requires_grad:
            input.register_hook(lambda grad: reporter(f"{operation}_input_gradient", grad))
    return result


def _record_product_input(input: torch.Tensor, operation: str) -> None:
    enabled, _ = _PRODUCT_BACKWARD_TRACE
    if not enabled:
        return
    values = input.detach().float()
    metrics = {
        "min": values.amin(),
        "max": values.amax(),
        "lt_1e-6_fraction": (values.abs() < _EPS).float().mean(),
        "negative_fraction": (values < 0).float().mean(),
        "nonfinite_fraction": (~torch.isfinite(values)).float().mean(),
    }
    if operation in _PRODUCT_INPUT_STATS:
        previous = _PRODUCT_INPUT_STATS[operation]
        previous["min"] = torch.minimum(previous["min"], metrics["min"])
        previous["max"] = torch.maximum(previous["max"], metrics["max"])
        for name in ("lt_1e-6_fraction", "negative_fraction", "nonfinite_fraction"):
            previous[name] = previous[name] + metrics[name]
        previous["calls"] = int(previous["calls"]) + 1
    else:
        metrics["calls"] = 1
        _PRODUCT_INPUT_STATS[operation] = metrics


def pop_product_numerical_diagnostics() -> dict[str, float]:
    """Return/reset compact input summaries for patched product operations."""
    result: dict[str, float] = {}
    for operation, metrics in _PRODUCT_INPUT_STATS.items():
        calls = max(1, int(metrics["calls"]))
        result[f"{operation}_input_min"] = float(metrics["min"].detach().cpu().item())
        result[f"{operation}_input_max"] = float(metrics["max"].detach().cpu().item())
        for name in ("lt_1e-6_fraction", "negative_fraction", "nonfinite_fraction"):
            result[f"{operation}_input_{name}"] = float(
                (metrics[name] / calls).detach().cpu().item()
            )
        result[f"{operation}_calls"] = float(calls)
    _PRODUCT_INPUT_STATS.clear()
    return result


def _patched_prod(input, dim=None, keepdim=False, *, dtype=None):
    if isinstance(input, torch.Tensor) and input.is_cuda and dim is not None:
        _record_product_input(input, "torch_prod")
        if not input.dtype.is_floating_point:
            return _orig_prod(input, dim, keepdim=keepdim, dtype=dtype)
        if _STRICT_NEGATIVE_CHECK and bool((input < 0).any()):
            return _orig_prod(input.detach().cpu(), dim, keepdim=keepdim, dtype=dtype).to(
                input.device
            )
        result = _log_space(input, dim, keepdim, cumulative=False)
        result = result.to(dtype) if dtype is not None else result
        return _watch_product_output(result, input, "patched_torch_prod")
    if dim is None:
        return _orig_prod(input, **({"dtype": dtype} if dtype is not None else {}))
    return _orig_prod(
        input, dim, keepdim=keepdim, **({"dtype": dtype} if dtype is not None else {})
    )


def _patched_cumprod(input, dim, *, dtype=None):
    if isinstance(input, torch.Tensor) and input.is_cuda:
        _record_product_input(input, "torch_cumprod")
        if not input.dtype.is_floating_point:
            return _orig_cumprod(input, dim, dtype=dtype)
        if _STRICT_NEGATIVE_CHECK and bool((input < 0).any()):
            return _orig_cumprod(input.detach().cpu(), dim, dtype=dtype).to(input.device)
        result = _log_space(input, dim, keepdim=False, cumulative=True)
        result = result.to(dtype) if dtype is not None else result
        return _watch_product_output(result, input, "patched_torch_cumprod")
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
        "with the CUDA no-sync fast path to avoid PyTorch's Jiterator, working around this "
        "environment's missing/mismatched libnvrtc-builtins. Every "
        "dnc.memory.Memory call site (get_usage_vector, allocation "
        "weighting, and any other prod/cumprod use) is covered "
        "automatically - no per-method patching needed."
    )
