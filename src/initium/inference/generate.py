"""Autoregressive rollout helpers for classic-track token tasks."""

from __future__ import annotations

import torch


@torch.no_grad()
def generate(
    model,
    output_proj,
    task,
    prompt_seq: torch.Tensor,
    max_new_steps: int,
    device: torch.device,
    temperature: float = 1.0,
    top_k: int = 0,
    ablate_memory: bool = False,
):
    """Generate token IDs while carrying DNC/controller state between steps.

    SplitGraphDNC keeps its full-prefix backbone contract and resumes the
    sequential DNC loop with ``start_step``; recurrent controllers consume
    the newly appended single timestep directly.
    """
    if temperature <= 0:
        raise ValueError("temperature must be > 0")
    if max_new_steps < 1 or top_k < 0:
        raise ValueError("max_new_steps must be >=1 and top_k must be >=0")
    model.eval()
    output_proj.eval()
    x = prompt_seq.detach().clone().to(device).unsqueeze(0)
    hidden = (None, None, None)
    generated = []
    first_logits = None
    split_graph = hasattr(model, "backbone")

    for index in range(max_new_steps):
        if index == 0:
            model_input = x
            reset = True
            kwargs = {}
        elif split_graph:
            model_input = x
            reset = False
            kwargs = {"start_step": x.shape[1] - 1}
        else:
            model_input = x[:, -1:, :]
            reset = False
            kwargs = {}
        output, hidden = model(
            model_input,
            hidden,
            reset_experience=reset,
            pass_through_memory=not ablate_memory,
            **kwargs,
        )
        logits = output_proj(output[-1]).float()
        if index == 0:
            first_logits = logits.squeeze(0).detach().cpu()
        if hasattr(task, "sample_token"):
            token = task.sample_token(logits.squeeze(0), temperature, top_k)
        else:
            logits = logits / temperature
            if top_k:
                k = min(top_k, logits.shape[-1])
                threshold = torch.topk(logits, k, dim=-1).values[..., -1, None]
                logits = logits.masked_fill(logits < threshold, float("-inf"))
            token = int(torch.multinomial(torch.softmax(logits, dim=-1), 1).item())
        generated.append(token)
        if index + 1 < max_new_steps:
            next_position = x.shape[1]
            next_input = task.encode_generated_token(token, next_position).to(device).view(1, 1, -1)
            x = torch.cat([x, next_input], dim=1)
    return {"tokens": generated, "first_logits": first_logits}
