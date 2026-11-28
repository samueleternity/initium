"""Training and inference cost measurements used by scale-readiness reviews."""

from __future__ import annotations

import time

import torch


def measure_training_step(step_fn, *, warmup: int = 5, iterations: int = 20,
                          token_count_per_step: int = 0, device=None):
    if warmup < 0 or iterations < 1:
        raise ValueError("warmup must be nonnegative and iterations positive")
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    for _ in range(warmup):
        step_fn()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    start = time.perf_counter()
    for _ in range(iterations):
        step_fn()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - start
    result = {"iterations": iterations, "elapsed_seconds": elapsed,
              "step_seconds": elapsed / iterations,
              "tokens_per_second": token_count_per_step * iterations / max(elapsed, 1e-12)}
    if device.type == "cuda":
        result["peak_cuda_bytes"] = torch.cuda.max_memory_allocated(device)
    return result


def measure_inference_latency(infer_fn, *, warmup: int = 5, iterations: int = 50):
    for _ in range(warmup):
        infer_fn()
    samples = []
    for _ in range(iterations):
        start = time.perf_counter()
        infer_fn()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        samples.append(time.perf_counter() - start)
    ordered = sorted(samples)
    return {"iterations": iterations, "mean_seconds": sum(samples) / len(samples),
            "p50_seconds": ordered[len(ordered) // 2],
            "p95_seconds": ordered[min(int(len(ordered) * 0.95), len(ordered) - 1)]}
