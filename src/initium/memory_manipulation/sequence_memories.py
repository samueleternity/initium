"""Sequence-parallel associative memories for split-graph controllers.

The modules in this file deliberately consume controller features rather than
feeding reads back into the controller. Their ``step`` methods are the
reference recurrence; ``forward`` keeps state in fp32 and is suitable for
whole-sequence training.
"""

from __future__ import annotations

from typing import NamedTuple

import torch
from torch import nn
from torch.nn import functional as F


class KVState(NamedTuple):
    keys: torch.Tensor  # [B, H, S, K]
    values: torch.Tensor  # [B, H, S, V]


class DeltaState(NamedTuple):
    matrix: torch.Tensor  # [B, H, V, K], always fp32


class SequenceMemory(nn.Module):
    """Common interface for memories driven by ``[B, T, D]`` features."""

    def __init__(self):
        super().__init__()
        self._collect_memory_usage = False
        self._usage_head_sums: dict[str, torch.Tensor] = {}
        self._usage_scalar_sums: dict[str, torch.Tensor] = {}
        self._usage_counts: dict[str, int] = {}

    def begin_stage_diagnostics(self) -> None:
        self._collect_memory_usage = True
        self._usage_head_sums = {}
        self._usage_scalar_sums = {}
        self._usage_counts = {}

    def _record_head_usage(self, name: str, values: torch.Tensor) -> None:
        if not self._collect_memory_usage:
            return
        values = values.detach().float()
        if values.ndim < 2 or values.shape[1] != getattr(self, "heads", values.shape[1]):
            raise ValueError(f"{name} must have head dimension at axis 1")
        reduce_dims = (0, *range(2, values.ndim))
        per_head = values.mean(dim=reduce_dims) if reduce_dims else values
        if name in self._usage_head_sums:
            self._usage_head_sums[name] = self._usage_head_sums[name] + per_head
        else:
            self._usage_head_sums[name] = per_head
        self._usage_counts[name] = self._usage_counts.get(name, 0) + 1

    def _record_usage_scalar(self, name: str, value: torch.Tensor | float) -> None:
        if not self._collect_memory_usage:
            return
        value = torch.as_tensor(value).detach().float().mean()
        if name in self._usage_scalar_sums:
            self._usage_scalar_sums[name] = self._usage_scalar_sums[name] + value
        else:
            self._usage_scalar_sums[name] = value
        self._usage_counts[name] = self._usage_counts.get(name, 0) + 1

    def finish_stage_diagnostics(self) -> dict[str, float]:
        self._collect_memory_usage = False
        result: dict[str, float] = {}
        for name, total in self._usage_head_sums.items():
            means = total / max(self._usage_counts[name], 1)
            values = means.detach().cpu().tolist()
            for head, value in enumerate(values):
                result[f"{name}_head_{head}_mean"] = float(value)
            if values:
                result[f"{name}_highest_mean_head"] = float(
                    max(range(len(values)), key=values.__getitem__)
                )
                result[f"{name}_lowest_mean_head"] = float(
                    min(range(len(values)), key=values.__getitem__)
                )
        for name, total in self._usage_scalar_sums.items():
            result[f"{name}_mean"] = float((total / max(self._usage_counts[name], 1)).cpu().item())
        if hasattr(self, "heads"):
            for head in range(self.heads):
                prefix = f"kv_head_{head}_relative_position_bin_"
                bin_keys = sorted(
                    name
                    for name in self._usage_scalar_sums
                    if name.startswith(prefix) and name.endswith("_mass")
                )
                if bin_keys:
                    indices = [int(name[len(prefix) : len(prefix) + 2]) for name in bin_keys]
                    masses = [result[f"{name}_mean"] for name in bin_keys]
                    result[f"kv_head_{head}_most_used_relative_position_bin"] = float(
                        indices[max(range(len(masses)), key=masses.__getitem__)]
                    )
                    result[f"kv_head_{head}_least_used_relative_position_bin"] = float(
                        indices[min(range(len(masses)), key=masses.__getitem__)]
                    )
        result["diagnostic_steps"] = float(max(self._usage_counts.values(), default=0))
        self._usage_head_sums = {}
        self._usage_scalar_sums = {}
        self._usage_counts = {}
        return result

    def forward(self, h_seq: torch.Tensor, state=None, lag: int = 1):
        raise NotImplementedError

    def step(self, h_t: torch.Tensor, state=None, lag: int = 1):
        raise NotImplementedError

    def reset_state(self, batch_size: int, *, device=None) -> KVState | DeltaState:
        raise NotImplementedError

    def pop_numerical_diagnostics(self) -> dict[str, float]:
        return {}


class GatedDeltaMemory(SequenceMemory):
    """Multi-head gated delta-rule memory with bounded recurrent state.

    The default sequence implementation evaluates the exact reference
    recurrence in chunks. CUDA runs can opt into FLA's fused WY scan without
    changing the public interface.
    """

    def __init__(
        self,
        input_size: int,
        heads: int = 4,
        key_size: int = 64,
        value_size: int = 64,
        chunk_size: int = 64,
        gate: bool = True,
        decay_scale: float = 1.0,
        use_fla: bool = False,
    ):
        super().__init__()
        if min(input_size, heads, key_size, value_size, chunk_size) <= 0:
            raise ValueError("memory dimensions and chunk_size must be positive")
        if decay_scale <= 0:
            raise ValueError("decay_scale must be positive")
        if use_fla and chunk_size not in {16, 32, 64}:
            raise ValueError("FLA delta backend requires chunk_size 16, 32, or 64")
        self.input_size = input_size
        self.heads = heads
        self.key_size = key_size
        self.value_size = value_size
        self.chunk_size = chunk_size
        self.use_gate = gate
        self.decay_scale = float(decay_scale)
        self.use_fla = bool(use_fla)
        self.query = nn.Linear(input_size, heads * key_size)
        self.key = nn.Linear(input_size, heads * key_size)
        self.value = nn.Linear(input_size, heads * value_size)
        self.write_gate = nn.Linear(input_size, heads)
        self.decay_gate = nn.Linear(input_size, heads)
        self.output = nn.Linear(heads * value_size, input_size)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def reset_state(self, batch_size: int, *, device=None) -> DeltaState:
        return DeltaState(torch.zeros(
            batch_size, self.heads, self.value_size, self.key_size,
            device=device, dtype=torch.float32,
        ))

    def _project(self, h_t: torch.Tensor):
        b = h_t.shape[0]
        q = F.normalize(self.query(h_t).view(b, self.heads, self.key_size).float(), dim=-1)
        k = F.normalize(self.key(h_t).view(b, self.heads, self.key_size).float(), dim=-1)
        v = self.value(h_t).view(b, self.heads, self.value_size).float()
        beta = torch.sigmoid(self.write_gate(h_t).float()).view(b, self.heads, 1, 1)
        alpha = torch.exp(-F.softplus(self.decay_gate(h_t).float()) * self.decay_scale)
        alpha = alpha.view(b, self.heads, 1, 1) if self.use_gate else torch.ones_like(beta)
        return q, k, v, beta, alpha

    def step(self, h_t: torch.Tensor, state=None, lag: int = 1):
        if lag not in (0, 1):
            raise ValueError("lag must be 0 or 1")
        if h_t.ndim != 2 or h_t.shape[-1] != self.input_size:
            raise ValueError(f"h_t must have shape [B, {self.input_size}]")
        state = self.reset_state(h_t.shape[0], device=h_t.device) if state is None else state
        matrix = state.matrix.float()
        q, k, v, beta, alpha = self._project(h_t)
        prediction = torch.einsum("bhvk,bhk->bhv", matrix, k)
        delta = beta[..., 0] * (v - alpha[..., 0] * prediction)
        updated = alpha * matrix + delta.unsqueeze(-1) @ k.unsqueeze(-2)
        read = torch.einsum("bhvk,bhk->bhv", updated if lag == 0 else matrix, q)
        read = read.to(h_t.dtype).reshape(h_t.shape[0], -1)
        return self.output(read), DeltaState(updated)

    def forward(self, h_seq: torch.Tensor, state=None, lag: int = 1):
        if h_seq.ndim != 3 or h_seq.shape[-1] != self.input_size:
            raise ValueError(f"h_seq must have shape [B, T, {self.input_size}]")
        if lag not in (0, 1):
            raise ValueError("lag must be 0 or 1")
        state = self.reset_state(h_seq.shape[0], device=h_seq.device) if state is None else state
        b, t, _ = h_seq.shape
        # Project the entire sequence in batched GEMMs. The state update is
        # kept as the exact reference recurrence and carried across bounded
        # chunks; fused WY scans can replace this block without changing the
        # state or projection interface.
        q = F.normalize(
            self.query(h_seq).view(b, t, self.heads, self.key_size).float(), dim=-1
        ).transpose(1, 2)
        k = F.normalize(
            self.key(h_seq).view(b, t, self.heads, self.key_size).float(), dim=-1
        ).transpose(1, 2)
        v = self.value(h_seq).view(b, t, self.heads, self.value_size).float().transpose(1, 2)
        beta = torch.sigmoid(self.write_gate(h_seq).float()).transpose(1, 2).unsqueeze(-1)
        if self.use_gate:
            alpha = torch.exp(
                -F.softplus(self.decay_gate(h_seq).float()) * self.decay_scale
            ).transpose(1, 2).unsqueeze(-1)
        else:
            alpha = torch.ones_like(beta)
        fast_result = (
            self._fla_forward(q, k, v, beta, alpha, state, lag) if t > 0 else None
        )
        if fast_result is not None:
            reads, state = fast_result
            if self._collect_memory_usage:
                shaped_reads = reads.float().reshape(
                    b, h_seq.shape[1], self.heads, self.value_size
                ).transpose(1, 2)
                self._record_head_usage("delta_read_rms", shaped_reads.square().mean(dim=(-1, -2)).sqrt())
                self._record_head_usage("delta_write_strength", beta.squeeze(-1))
                self._record_head_usage("delta_decay_alpha", alpha.squeeze(-1))
                self._record_head_usage(
                    "delta_state_rms", state.matrix.square().mean(dim=(-1, -2)).sqrt()
                )
                self._record_usage_scalar(
                    "delta_state_elements", self.heads * self.value_size * self.key_size
                )
            return self.output(reads.flatten(-2).to(h_seq.dtype)), state, {}
        read_sequence = []
        for start in range(0, h_seq.shape[1], self.chunk_size):
            for t in range(start, min(start + self.chunk_size, h_seq.shape[1])):
                matrix = state.matrix.float()
                q_t, k_t, v_t = q[:, :, t], k[:, :, t], v[:, :, t]
                beta_t, alpha_t = beta[:, :, t], alpha[:, :, t]
                read_before = torch.einsum("bhvk,bhk->bhv", matrix, q_t)
                prediction = torch.einsum("bhvk,bhk->bhv", matrix, k_t)
                delta = beta_t * (v_t - alpha_t * prediction)
                updated = alpha_t.unsqueeze(-1) * matrix + delta.unsqueeze(-1) @ k_t.unsqueeze(-2)
                read = (
                    torch.einsum("bhvk,bhk->bhv", updated, q_t)
                    if lag == 0
                    else read_before
                )
                read_sequence.append(read.flatten(1).to(h_seq.dtype))
                state = DeltaState(updated)
        if not read_sequence:
            return h_seq.new_zeros(b, 0, self.input_size), state, {}
        reads = torch.stack(read_sequence, dim=1)
        if self._collect_memory_usage:
            shaped_reads = reads.float().reshape(
                b, h_seq.shape[1], self.heads, self.value_size
            ).transpose(1, 2)
            self._record_head_usage("delta_read_rms", shaped_reads.square().mean(dim=(-1, -2)).sqrt())
            self._record_head_usage("delta_write_strength", beta.squeeze(-1))
            self._record_head_usage("delta_decay_alpha", alpha.squeeze(-1))
            self._record_head_usage(
                "delta_state_rms", state.matrix.square().mean(dim=(-1, -2)).sqrt()
            )
            self._record_usage_scalar(
                "delta_state_elements", self.heads * self.value_size * self.key_size
            )
        return self.output(reads), state, {}

    def _fla_forward(self, q, k, v, beta, alpha, state, lag: int):
        """Use FLA's fused chunk scan when it is available and compatible."""
        if not self.use_fla:
            return None
        if not q.is_cuda:
            raise RuntimeError("The FLA delta backend requires CUDA tensors")
        try:
            from fla.ops.gated_delta_rule import chunk_gated_delta_rule
        except ImportError as exc:
            raise RuntimeError(
                "delta_use_fla=True requires flash-linear-attention to be installed"
            ) from exc
        # FLA stores state as [B,H,K,V], while this module's reference uses
        # [B,H,V,K]. It emits reads after the token write. For lag=1, insert a
        # no-op token and shift writes one position right so query t observes
        # the exact state S_(t-1), while the final state still includes every
        # write from the input sequence.
        q = q.transpose(1, 2).contiguous()
        k = k.transpose(1, 2).contiguous()
        v = v.transpose(1, 2).contiguous()
        beta = beta.squeeze(-1).transpose(1, 2).contiguous()
        log_decay = alpha.squeeze(-1).log().transpose(1, 2).contiguous()
        initial_state = state.matrix.transpose(-1, -2).contiguous()
        if lag == 1:
            b, t, heads, key_size = q.shape
            no_query = q.new_zeros(b, 1, heads, key_size)
            no_key = k.new_zeros(b, 1, heads, key_size)
            no_value = v.new_zeros(b, 1, heads, self.value_size)
            q = torch.cat((q, no_query), dim=1)
            k = torch.cat((no_key, k), dim=1)
            v = torch.cat((no_value, v), dim=1)
            beta = torch.cat((beta.new_zeros(b, 1, heads), beta), dim=1)
            log_decay = torch.cat((log_decay.new_zeros(b, 1, heads), log_decay), dim=1)
        reads, final_state = chunk_gated_delta_rule(
            q=q,
            k=k,
            v=v,
            g=log_decay,
            beta=beta,
            scale=1.0,
            initial_state=initial_state,
            output_final_state=True,
            chunk_size=self.chunk_size,
        )
        if lag == 1:
            reads = reads[:, : q.shape[1] - 1]
        return reads, DeltaState(final_state.transpose(-1, -2).contiguous())


class CausalKVMemory(SequenceMemory):
    """Dense causal KV memory using scaled dot-product attention.

    Training may replace ``value`` with a Gaussian stochastic write head.
    Keys remain deterministic so memory addressing is not sampled along with
    stored content.
    """

    def __init__(
        self,
        input_size: int,
        heads: int = 4,
        key_size: int = 64,
        value_size: int = 64,
        topk: int | None = None,
        position_mode: str = "none",
    ):
        super().__init__()
        if min(input_size, heads, key_size, value_size) <= 0:
            raise ValueError("memory dimensions must be positive")
        if topk is not None and topk <= 0:
            raise ValueError("topk must be positive")
        if position_mode not in {"none", "rope"}:
            raise ValueError("position_mode must be 'none' or 'rope'")
        if position_mode == "rope" and key_size % 2:
            raise ValueError("RoPE requires an even memory key size")
        self.input_size = input_size
        self.heads = heads
        self.key_size = key_size
        self.value_size = value_size
        self.topk = topk
        self.position_mode = position_mode
        self.query = nn.Linear(input_size, heads * key_size)
        self.key = nn.Linear(input_size, heads * key_size)
        self.value = nn.Linear(input_size, heads * value_size)
        self.output = nn.Linear(heads * value_size, input_size)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def reset_state(self, batch_size: int, *, device=None) -> KVState:
        return KVState(
            torch.empty(batch_size, self.heads, 0, self.key_size, device=device),
            torch.empty(batch_size, self.heads, 0, self.value_size, device=device),
        )

    def _apply_rope(self, x: torch.Tensor, start_position: int) -> torch.Tensor:
        if self.position_mode == "none" or x.shape[-2] == 0:
            return x
        positions = torch.arange(
            start_position, start_position + x.shape[-2], device=x.device, dtype=torch.float32
        )
        inv_frequency = torch.exp(
            torch.arange(0, self.key_size, 2, device=x.device, dtype=torch.float32)
            * (-torch.log(torch.tensor(10_000.0, device=x.device)) / self.key_size)
        )
        angles = positions[:, None] * inv_frequency[None, :]
        cos, sin = angles.cos().to(x.dtype)[None, None], angles.sin().to(x.dtype)[None, None]
        pairs = x.reshape(*x.shape[:-1], self.key_size // 2, 2)
        even, odd = pairs[..., 0], pairs[..., 1]
        rotated = torch.stack((even * cos - odd * sin, even * sin + odd * cos), dim=-1)
        return rotated.flatten(-2)

    def _qkv(self, h: torch.Tensor, start_position: int = 0):
        b, t, _ = h.shape
        q = self.query(h).view(b, t, self.heads, self.key_size).transpose(1, 2)
        k = self.key(h).view(b, t, self.heads, self.key_size).transpose(1, 2)
        v = self.value(h).view(b, t, self.heads, self.value_size).transpose(1, 2)
        q = self._apply_rope(q, start_position)
        k = self._apply_rope(k, start_position)
        return q, k, v

    def _attention(self, q, keys, values, allowed):
        allowed = allowed.expand(-1, q.shape[1], -1, -1)
        if self.topk is not None and keys.shape[-2] > self.topk:
            scores = torch.matmul(q.float(), keys.float().transpose(-2, -1))
            scores.mul_(self.key_size**-0.5)
            scores.masked_fill_(~allowed, torch.finfo(scores.dtype).min)
            indices = scores.topk(min(self.topk, keys.shape[-2]), dim=-1).indices
            selected = torch.zeros_like(allowed).scatter_(-1, indices, True)
            allowed = allowed & selected
        # SDPA supports a boolean allow-mask and chooses flash/memory-efficient
        # kernels where available. All-masked rows (first token at lag=1) are
        # explicitly zeroed after the kernel to avoid backend-dependent
        # softmax behavior (including NaNs on some implementations).
        has_allowed_key = allowed.any(dim=-1, keepdim=True)
        out = F.scaled_dot_product_attention(q, keys, values, attn_mask=allowed)
        if self._collect_memory_usage:
            self._record_usage_scalar("kv_history_tokens", keys.shape[-2])
            query_count = min(q.shape[-2], 16)
            query_indices = torch.linspace(
                0, max(q.shape[-2] - 1, 0), steps=query_count, device=q.device
            ).round().long().unique()
            diagnostic_q = q.index_select(-2, query_indices).float()
            diagnostic_allowed = allowed.index_select(-2, query_indices)
            scores = torch.matmul(diagnostic_q, keys.float().transpose(-2, -1))
            scores.mul_(self.key_size**-0.5)
            scores.masked_fill_(~diagnostic_allowed, torch.finfo(scores.dtype).min)
            probabilities = torch.softmax(scores, dim=-1)
            probabilities = torch.where(
                diagnostic_allowed, probabilities, torch.zeros_like(probabilities)
            )
            probabilities = probabilities / probabilities.sum(dim=-1, keepdim=True).clamp_min(1e-12)
            entropy = -(probabilities * probabilities.clamp_min(1e-12).log()).sum(dim=-1)
            self._record_head_usage("kv_attention_entropy", entropy)
            self._record_head_usage("kv_attention_max_weight", probabilities.amax(dim=-1))
            position_mass = probabilities.sum(dim=(0, 2))
            position_mass = position_mass / position_mass.sum(dim=-1, keepdim=True).clamp_min(1e-12)
            bins = min(16, keys.shape[-2])
            if bins:
                positions = torch.arange(keys.shape[-2], device=keys.device)
                bin_indices = torch.div(positions * bins, keys.shape[-2], rounding_mode="floor")
                bin_mass = torch.zeros(
                    self.heads, bins, device=keys.device, dtype=position_mass.dtype
                ).scatter_add_(1, bin_indices.expand(self.heads, -1), position_mass)
                for head in range(self.heads):
                    for index in range(bins):
                        self._record_usage_scalar(
                            f"kv_head_{head}_relative_position_bin_{index:02d}_mass",
                            bin_mass[head, index],
                        )
        return torch.where(has_allowed_key, out, torch.zeros_like(out))

    def step(self, h_t: torch.Tensor, state=None, lag: int = 1):
        if lag not in (0, 1):
            raise ValueError("lag must be 0 or 1")
        h = h_t.unsqueeze(1)
        state = self.reset_state(h_t.shape[0], device=h_t.device) if state is None else state
        q, k, v = self._qkv(h, start_position=state.keys.shape[-2])
        if state.keys.shape[-2] == 0:
            state = KVState(state.keys.to(dtype=k.dtype), state.values.to(dtype=v.dtype))
        keys = torch.cat((state.keys, k), dim=-2)
        values = torch.cat((state.values, v), dim=-2)
        n_old = state.keys.shape[-2]
        allowed = torch.ones(1, 1, 1, keys.shape[-2], dtype=torch.bool, device=h.device)
        if lag:
            allowed[..., n_old:] = False
        read = self._attention(q, keys, values, allowed)
        read = read.transpose(1, 2).reshape(h_t.shape[0], 1, -1).squeeze(1)
        return self.output(read), KVState(keys, values)

    def forward(self, h_seq: torch.Tensor, state=None, lag: int = 1):
        if h_seq.ndim != 3 or h_seq.shape[-1] != self.input_size:
            raise ValueError(f"h_seq must have shape [B, T, {self.input_size}]")
        if lag not in (0, 1):
            raise ValueError("lag must be 0 or 1")
        b, t, _ = h_seq.shape
        state = self.reset_state(b, device=h_seq.device) if state is None else state
        old = state.keys.shape[-2]
        q, k, v = self._qkv(h_seq, start_position=old)
        if state.keys.shape[-2] == 0:
            state = KVState(state.keys.to(dtype=k.dtype), state.values.to(dtype=v.dtype))
        keys = torch.cat((state.keys, k), dim=-2)
        values = torch.cat((state.values, v), dim=-2)
        positions = torch.arange(t, device=h_seq.device)[:, None] + old
        key_positions = torch.arange(old + t, device=h_seq.device)[None, :]
        allowed = key_positions <= positions if lag == 0 else key_positions < positions
        allowed = allowed[None, None, :, :].expand(b, 1, -1, -1)
        reads = self._attention(q, keys, values, allowed)
        if self._collect_memory_usage:
            self._record_head_usage("kv_key_rms", keys.square().mean(dim=(-1, -2)).sqrt())
            self._record_head_usage("kv_value_rms", values.square().mean(dim=(-1, -2)).sqrt())
            self._record_head_usage(
                "kv_read_rms", reads.square().mean(dim=(-1, -2)).sqrt()
            )
        reads = reads.transpose(1, 2).reshape(b, t, -1)
        return self.output(reads), KVState(keys, values), {}
