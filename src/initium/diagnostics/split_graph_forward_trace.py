"""First-nonfinite tracing for split-graph forward execution.

This diagnostic synchronizes CUDA at observed boundaries. The trainer attaches
it only for a selected step or while replaying a batch that already failed a
normal trainer finite check. Dormant automatic tracing does not inspect tensors
or synchronize CUDA during training.
"""

from __future__ import annotations

import csv
import os
from datetime import datetime, timezone
from typing import Any, TextIO

import torch


class SplitGraphForwardTrace:
    """Write tensor boundary health rows for one absolute training step."""

    def __init__(
        self,
        log_dir: str,
        run_id: str,
        target_step: int | None,
        *,
        trigger_on_error: bool = False,
    ):
        self.target_step = target_step
        self.trigger_on_error = trigger_on_error
        self.triggered = False
        self.active = False
        self._step_seen = False
        self._rows_written = 0
        self.current_time_index: int | None = None
        self._session = datetime.now(timezone.utc).isoformat(timespec="seconds")
        path = os.path.join(log_dir, f"run_{run_id}_splitgraph_forward_trace.csv")
        self.path = path
        self._file: TextIO | None = None
        self._writer: Any | None = None

    def _ensure_writer(self) -> None:
        if self._file is not None:
            return
        self._file = open(self.path, "a", newline="", encoding="utf-8")
        self._writer = csv.writer(self._file)
        assert self._writer is not None
        if os.path.getsize(self.path) == 0:
            self._writer.writerow(
                [
                    "session_utc",
                    "step",
                    "boundary",
                    "time_index",
                    "dtype",
                    "shape",
                    "finite_count",
                    "element_count",
                    "nonfinite_count",
                    "max_abs_finite",
                    "first_nonfinite_index",
                ]
            )
            self._file.flush()

    def begin_step(self, step: int) -> None:
        self.active = self.target_step is not None and step == self.target_step
        self.current_time_index = None
        if self.active:
            self._step_seen = True
            self._rows_written = 0
            self._ensure_writer()
            print(
                f"[split-graph-forward-trace] tracing absolute optimizer step {step}; "
                "CUDA synchronization is enabled for observed boundaries",
                flush=True,
            )

    def activate_on_error(self, step: int, failure: str) -> bool:
        """Arm detailed tracing after an existing finite check detects failure."""
        if not self.trigger_on_error or self.triggered:
            return False
        self.triggered = True
        self.target_step = step
        self.active = True
        self._step_seen = True
        self._rows_written = 0
        self.current_time_index = None
        self._ensure_writer()
        print(
            "[split-graph-forward-trace] automatic trace triggered by "
            f"{failure} at step {step}; replaying the failed batch with boundary "
            f"checks enabled; output: {self.path}",
            flush=True,
        )
        return True

    def finish_error_replay(self, step: int) -> None:
        """Close a replay that did not reproduce a non-finite boundary."""
        if not self.active:
            return
        self.active = False
        print(
            "[split-graph-forward-trace] automatic replay completed without "
            f"reproducing a non-finite value at observed boundaries; "
            f"rows_written={self._rows_written}; original_failure_step={step}; "
            f"trace_csv={self.path}",
            flush=True,
        )
        self.close()

    def set_time_index(self, time_index: int | None) -> None:
        self.current_time_index = time_index

    def observe(self, boundary: str, tensor: torch.Tensor, time_index: int | None = None) -> None:
        if not self.active:
            return
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"numeric trace boundary {boundary!r} is not a tensor")

        value = tensor.detach()
        if time_index is None:
            time_index = self.current_time_index
        self._ensure_writer()
        assert self._file is not None and self._writer is not None
        finite_mask = torch.isfinite(value)
        safe_abs = value.float().abs().masked_fill(~finite_mask, 0.0)
        packed = torch.stack((finite_mask.sum().float(), safe_abs.amax())).cpu().tolist()
        finite_count, max_abs_finite = int(packed[0]), float(packed[1])
        element_count = value.numel()
        nonfinite_count = element_count - finite_count
        first_bad: str | tuple[int, ...] = ""
        if nonfinite_count:
            coordinates = torch.nonzero(~finite_mask, as_tuple=False)
            first_bad = tuple(int(index) for index in coordinates[0].cpu().tolist())

        self._writer.writerow(
            [
                self._session,
                self.target_step,
                boundary,
                time_index,
                str(value.dtype),
                tuple(value.shape),
                finite_count,
                element_count,
                nonfinite_count,
                max_abs_finite,
                first_bad,
            ]
        )
        self._file.flush()
        self._rows_written += 1
        print(
            "[split-graph-forward-trace] "
            f"step={self.target_step} "
            f"time_index={time_index if time_index is not None else '-'} "
            f"boundary={boundary} dtype={value.dtype} shape={tuple(value.shape)} "
            f"finite={finite_count}/{element_count} nonfinite={nonfinite_count} "
            f"max_abs_finite={max_abs_finite:.6g} "
            f"first_nonfinite_index={first_bad or '-'}",
            flush=True,
        )
        if nonfinite_count:
            failure = (
                "first observed non-finite split-graph forward boundary: "
                f"step={self.target_step}, time_index={time_index}, "
                f"boundary={boundary}, dtype={value.dtype}, shape={tuple(value.shape)}, "
                f"nonfinite={nonfinite_count}/{element_count}, "
                f"max_abs_finite={max_abs_finite:.6g}, first_index={first_bad}; "
                f"trace_csv={self.path}"
            )
            self.active = False
            self.close()
            raise FloatingPointError(failure)

    def end_step(self, step: int) -> None:
        if self.active:
            if self.trigger_on_error and self.triggered:
                return
            self.active = False
            if self._rows_written:
                message = (
                    f"step {step} completed without a non-finite value at observed "
                    f"boundaries; rows_written={self._rows_written}; trace_csv={self.path}"
                )
            else:
                message = (
                    f"WARNING: step {step} completed but no forward boundaries were observed; "
                    f"check that the active SplitGraphDNC implementation has trace hooks; "
                    f"trace_csv={self.path}"
                )
            print(f"[split-graph-forward-trace] {message}", flush=True)

    def close(self) -> None:
        if not self._step_seen and self.target_step is not None:
            print(
                f"[split-graph-forward-trace] requested step {self.target_step} was not "
                "reached; no forward trace rows were written",
                flush=True,
            )
        if self._file is not None and not self._file.closed:
            self._file.close()
