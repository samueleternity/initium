"""
file: inference/cache/cached_engine.py

InferenceEngine + caches. Only used when at least one cache is active, so with caching
off the plain engine (and its behaviour) is untouched.

Per episode:
  1. episode-level lookups   (ResultCache: skip the forward pass entirely)
  2. else compute, resuming from the longest cached prefix snapshot (PrefixStateCache)
     and snapshotting every not-yet-cached boundary on the way, in segments
  3. store the episode result in the episode-level caches

Correctness net: for the first `verify_hits` cache hits the episode is ALSO recomputed
without any cache and compared (scored outputs + carried state, tolerance verify_tol).
A mismatch (or NaN) prints a warning, uses the recomputed values, and disables all
caches for the remainder of the run.

Segment execution:
  MambaDNC / stock DNC : model(x[start:end], hidden, reset_experience=False)  (state passes through)
  SplitGraphDNC        : model(x[:end], hidden, ..., start_step=start)  -- the parallel
                         backbone is recomputed over the whole prefix (cheap, stateless
                         across calls, keeps outputs exact); only the sequential memory
                         loop is skipped for t < start.
"""

from __future__ import annotations

import inspect
import time

import torch

from src.initium.inference.cache.base_cache import EpisodeCtx
from src.initium.inference.cache.cache_config import (
    DEFAULT_VERIFY_HITS,
    DEFAULT_VERIFY_TOL,
    ROOT_CHAIN,
)
from src.initium.inference.cache.keys import chain_next, hash_tensor
from src.initium.inference.cache.state_utils import clone_tree, tree_max_abs_diff
from src.initium.inference.engine import FRESH_HIDDEN, InferenceEngine
from src.initium.inference.metrics import EpisodeResult


def model_supports_resume(model) -> bool:
    if hasattr(model, "backbone"):  # SplitGraphDNC
        return "start_step" in inspect.signature(model.forward).parameters
    return True  # MambaDNC / stock dnc: hx passes through


class CachedInferenceEngine(InferenceEngine):
    def __init__(
        self,
        model,
        output_proj,
        task,
        device,
        caches,
        ablate_memory: bool = False,
        verify_hits: int = DEFAULT_VERIFY_HITS,
        verify_tol: float = DEFAULT_VERIFY_TOL,
    ):
        super().__init__(model, output_proj, task, device, ablate_memory)
        self.caches = list(caches)
        self.resumable = model_supports_resume(model)
        self.verify_hits, self.verify_tol = verify_hits, verify_tol
        self.verify = {
            "checked": 0,
            "failures": 0,
            "max_output_diff": 0.0,
            "max_state_diff": 0.0,
            "disabled_after_failure": False,
        }
        self.notes: list[str] = []
        self._disabled = False
        self._warned_bounds = False

    # ---- small utils -----------------------------------------------------------
    def _note(self, msg: str) -> None:
        if msg not in self.notes:
            self.notes.append(msg)
            print(f"[cache] {msg}")

    def _sync(self) -> None:
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def cache_report(self, reset: bool = False) -> dict:
        return {c.name: c.report(reset=reset) for c in self.caches}

    def _select_active(self, reset_experience: bool):
        active = []
        for c in self.caches:
            ok, why = c.applicable(reset_experience=reset_experience, resumable=self.resumable)
            if ok:
                active.append(c)
            else:
                self._note(f"{c.name} cache inactive: {why}")
        return active

    def _valid_boundaries(self, ep) -> list[int]:
        T = ep.input_seq.shape[0]
        bs = sorted({int(b) for b in ep.cache_boundaries})
        ok = [b for b in bs if 0 < b < T and float(ep.mask[:b].sum()) == 0.0]
        if len(ok) != len(bs) and not self._warned_bounds:
            self._warned_bounds = True
            self._note(
                "some declared cache boundaries were dropped (out of range, or a scored "
                "step lies before them)"
            )
        return ok

    # ---- model execution ---------------------------------------------------------
    @torch.no_grad()
    def _forward_segment(
        self, x_full: torch.Tensor, start: int, end: int, hidden, reset_experience: bool
    ):
        kw = dict(reset_experience=reset_experience, pass_through_memory=not self.ablate_memory)
        if hasattr(self.model, "backbone"):  # SplitGraphDNC
            x = x_full[:end].unsqueeze(0).to(self.device)
            if start > 0:
                kw["start_step"] = start
        else:
            x = x_full[start:end].unsqueeze(0).to(self.device)
        output, new_hidden = self.model(x, hidden, **kw)
        output = output.transpose(0, 1).contiguous().squeeze(0)
        output = self.output_proj(output)
        return output.float().cpu(), new_hidden

    def _compute_full(self, ctx: EpisodeCtx, hidden):
        """Uncached reference: one forward over the whole episode."""
        x = ctx.episode.input_seq
        t0 = time.perf_counter()
        out, hid = self._forward_segment(x, 0, x.shape[0], hidden, ctx.reset_experience)
        self._sync()
        return out, hid, (time.perf_counter() - t0) * 1000.0

    def _compute(self, ctx: EpisodeCtx, hidden, active):
        x = ctx.episode.input_seq
        T = x.shape[0]
        use_bounds = ctx.reset_experience and self.resumable and bool(ctx.boundaries)
        resume = None
        if use_bounds:
            for c in active:
                r = c.find_resume(ctx)
                if r is not None and (resume is None or r.position > resume.position):
                    resume = r
        start = resume.position if resume is not None else 0
        hid = resume.hidden if resume is not None else hidden
        seg_reset = ctx.reset_experience and resume is None
        cuts = [b for b in ctx.boundaries if b > start] if use_bounds else []
        cuts.append(T)

        cold_ms = resume.saved_ms if resume is not None else 0.0  # cost as if computed from scratch
        segs, pos = [], start
        for cut in cuts:
            t0 = time.perf_counter()
            seg, hid = self._forward_segment(x, pos, cut, hid, seg_reset)
            self._sync()
            cold_ms += (time.perf_counter() - t0) * 1000.0
            segs.append(seg)
            if cut < T:  # snapshot BEFORE the next segment mutates state
                for c in active:
                    c.store_boundary(ctx, cut, hid, cold_ms)
            pos, seg_reset = cut, False
        out = torch.cat(segs, dim=0)
        if start > 0:  # prefix rows are never scored (validated)
            out = torch.cat([torch.zeros(start, out.shape[1]), out], dim=0)
        return out, hid, resume, cold_ms

    # ---- verification -------------------------------------------------------------
    def _maybe_verify(self, ctx, incoming_hidden, out, new_hidden, event, state_valid):
        v = self.verify
        if self._disabled or v["checked"] >= self.verify_hits:
            return out, new_hidden, event
        ref_out, ref_hidden, _ = self._compute_full(ctx, clone_tree(incoming_hidden))
        mask = ctx.episode.mask == 1
        d_out = (out[mask] - ref_out[mask]).abs().max().item() if bool(mask.any()) else 0.0
        d_state = tree_max_abs_diff(new_hidden, ref_hidden) if state_valid else 0.0
        v["checked"] += 1
        v["max_output_diff"] = max(v["max_output_diff"], d_out)
        v["max_state_diff"] = max(v["max_state_diff"], d_state)
        if d_out <= self.verify_tol and d_state <= self.verify_tol:  # NaN-safe form
            return out, new_hidden, event
        v["failures"] += 1
        v["disabled_after_failure"] = True
        self._disabled = True
        self._note(
            f"VERIFICATION FAILED at episode {ctx.index} ({event}): max|d output|={d_out:.3e}, "
            f"max|d state|={d_state:.3e} > tol {self.verify_tol:g}. Caches disabled for the "
            f"rest of this run; using the recomputed result."
        )
        return ref_out, ref_hidden, event + ":VERIFY-FAILED"

    # ---- one episode --------------------------------------------------------------
    def _run_one(self, ctx: EpisodeCtx, hidden, active):
        for c in active:
            hit = c.lookup_episode(ctx)
            if hit is not None:
                new_hidden = FRESH_HIDDEN if ctx.reset_experience else hit.hidden
                return self._maybe_verify(
                    ctx,
                    hidden,
                    hit.output,
                    new_hidden,
                    f"hit:{c.name}",
                    state_valid=not ctx.reset_experience,
                )
        out, new_hidden, resume, cold_ms = self._compute(ctx, hidden, active)
        event = "miss"
        if resume is not None:
            event = f"hit:{resume.cache_name}@{resume.position}"
            out, new_hidden, event = self._maybe_verify(
                ctx, hidden, out, new_hidden, event, state_valid=True
            )
        if not self._disabled:
            for c in active:
                c.store_episode(ctx, out, new_hidden, cold_ms)
        return out, new_hidden, event

    def run(
        self,
        episodes,
        reset_experience: bool,
        verbose_n: int = 0,
        progress_every: int = 0,
        label: str = "",
    ) -> list[EpisodeResult]:
        active = self._select_active(reset_experience)
        want_bounds = any(c.wants_boundaries for c in active)
        hidden, chain = FRESH_HIDDEN, ROOT_CHAIN
        results: list[EpisodeResult] = []
        run_items = run_correct = 0
        for i, ep in enumerate(episodes):
            if reset_experience:
                hidden = FRESH_HIDDEN
            t0 = time.perf_counter()
            act = [] if self._disabled else active
            ctx = EpisodeCtx(
                index=i,
                episode=ep,
                reset_experience=reset_experience,
                in_hash=hash_tensor(ep.input_seq) if act else "",
                chain=chain,
                boundaries=self._valid_boundaries(ep) if (want_bounds and act) else [],
            )
            out, hidden, event = self._run_one(ctx, hidden, act)
            self._sync()
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            if not reset_experience:  # fresh mode: chain stays "root" (state is always fresh)
                chain = chain_next(chain, ctx.in_hash)

            score = self.task.score_episode(out, ep, verbose=(i < verbose_n))
            results.append(EpisodeResult(i, score, elapsed_ms, reset_experience, cache_event=event))
            run_items += score.n_items
            run_correct += score.n_correct
            if progress_every and ((i + 1) % progress_every == 0 or i + 1 == len(episodes)):
                print(
                    f"[engine{label}] episode {i + 1}/{len(episodes)} | "
                    f"running item acc {100.0 * run_correct / max(run_items, 1):.2f}%"
                )
        return results
