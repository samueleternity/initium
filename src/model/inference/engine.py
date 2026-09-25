"""
file: inference/engine.py

Runs episodes through a loaded model. Knows nothing about any specific task;
it only calls task.score_episode().

Two regimes (the reset_experience switch):
  reset_experience=True  : every episode starts from hidden=(None,None,None) with
                           reset_experience=True -> independent episodes, exactly
                           like the periodic OOD eval in training.
  reset_experience=False : the (controller state, memory, last read) returned by
                           episode i is fed into episode i+1 with
                           reset_experience=False -> memory content and recurrent
                           state persist across the whole run. NO weights change;
                           any "learning" is in-context via memory + controller state.
"""

from __future__ import annotations

import time

import torch

from inference.metrics import EpisodeResult

FRESH_HIDDEN = (None, None, None)


class InferenceEngine:
    def __init__(
        self,
        model,
        output_proj,
        task,
        device,
        ablate_memory: bool = False,
        combiner_skip_stages: frozenset | None = None,
    ):
        self.model, self.output_proj, self.task = model, output_proj, task
        self.device, self.ablate_memory = device, ablate_memory
        self.combiner_skip_stages = combiner_skip_stages
        self.model.eval()
        self.output_proj.eval()

    @torch.no_grad()
    def _forward(self, episode, hidden, reset_experience: bool):
        x = episode.input_seq.unsqueeze(0).to(self.device)  # (1, T, input_dim)
        kw = dict(reset_experience=reset_experience, pass_through_memory=not self.ablate_memory)
        if self.combiner_skip_stages:
            kw["combiner_skip_stages"] = self.combiner_skip_stages
        output, new_hidden = self.model(x, hidden, **kw)
        output = output.transpose(0, 1).contiguous().squeeze(0)  # (T, input_dim)
        output = self.output_proj(output)  # (T, output_dim)
        return output.float().cpu(), new_hidden

    def run(
        self,
        episodes,
        reset_experience: bool,
        verbose_n: int = 0,
        progress_every: int = 0,
        label: str = "",
    ) -> list[EpisodeResult]:
        hidden = FRESH_HIDDEN
        results: list[EpisodeResult] = []
        run_items = run_correct = 0
        for i, ep in enumerate(episodes):
            if reset_experience:
                hidden = FRESH_HIDDEN
            t0 = time.perf_counter()
            out, hidden = self._forward(ep, hidden, reset_experience)
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
            elapsed_ms = (time.perf_counter() - t0) * 1000.0

            score = self.task.score_episode(out, ep, verbose=(i < verbose_n))
            results.append(EpisodeResult(i, score, elapsed_ms, reset_experience))
            run_items += score.n_items
            run_correct += score.n_correct
            if progress_every and ((i + 1) % progress_every == 0 or i + 1 == len(episodes)):
                print(
                    f"[engine{label}] episode {i + 1}/{len(episodes)} | "
                    f"running item acc {100.0 * run_correct / max(run_items, 1):.2f}%"
                )
        return results
