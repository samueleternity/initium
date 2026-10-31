"""Causal KV sidecar for committed workspace updates.

This channel stores up to two committed slot records per token. Records from
the current episode are first collected in a causal
warm-up pass, then read by a second backbone pass. Only records produced by
the second pass are committed to the persistent state returned to the caller.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class WorkspaceKVState:
    keys: torch.Tensor
    values: torch.Tensor
    valid: torch.Tensor


class WorkspaceKVBridge(nn.Module):
    """A masked causal KV bank written from committed workspace content."""

    def __init__(
        self,
        input_size: int,
        heads: int,
        key_size: int,
        value_size: int,
        *,
        position_mode: str = "none",
        record_size: int | None = None,
        records_per_token: int = 2,
    ):
        super().__init__()
        if min(input_size, heads, key_size, value_size) <= 0:
            raise ValueError("workspace KV dimensions must be positive")
        if position_mode not in {"none", "rope"}:
            raise ValueError("workspace KV position_mode must be 'none' or 'rope'")
        if position_mode == "rope" and key_size % 2:
            raise ValueError("RoPE requires an even workspace KV key size")
        if records_per_token <= 0:
            raise ValueError("records_per_token must be positive")
        self.input_size = int(input_size)
        self.record_size = int(input_size if record_size is None else record_size)
        if self.record_size <= 0:
            raise ValueError("workspace KV record_size must be positive")
        self.heads = int(heads)
        self.key_size = int(key_size)
        self.value_size = int(value_size)
        self.records_per_token = int(records_per_token)
        self.position_mode = position_mode
        self.query = nn.Linear(input_size, heads * key_size)
        self.key = nn.Linear(self.record_size, heads * key_size)
        self.value = nn.Linear(self.record_size, heads * value_size)
        self.output = nn.Linear(heads * value_size, input_size)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def reset_state(self, batch_size: int, *, device=None, dtype=None) -> WorkspaceKVState:
        parameter = self.key.weight
        dtype = parameter.dtype if dtype is None else dtype
        return WorkspaceKVState(
            torch.empty(batch_size, self.heads, 0, self.key_size, device=device, dtype=dtype),
            torch.empty(batch_size, self.heads, 0, self.value_size, device=device, dtype=dtype),
            torch.empty(batch_size, 0, device=device, dtype=torch.bool),
        )

    def _rope(self, x: torch.Tensor, start: int) -> torch.Tensor:
        if self.position_mode == "none" or x.shape[-2] == 0:
            return x
        positions = torch.arange(start, start + x.shape[-2], device=x.device, dtype=torch.float32)
        inv = torch.exp(
            torch.arange(0, self.key_size, 2, device=x.device, dtype=torch.float32)
            * (-torch.log(torch.tensor(10_000.0, device=x.device)) / self.key_size)
        )
        angles = positions[:, None] * inv[None, :]
        cos, sin = angles.cos().to(x.dtype)[None, None], angles.sin().to(x.dtype)[None, None]
        pairs = x.reshape(*x.shape[:-1], self.key_size // 2, 2)
        even, odd = pairs[..., 0], pairs[..., 1]
        return torch.stack((even * cos - odd * sin, even * sin + odd * cos), dim=-1).flatten(-2)

    def append(
        self,
        records: torch.Tensor,
        commit_mask: torch.Tensor,
        state: WorkspaceKVState | None,
    ) -> WorkspaceKVState:
        if records.ndim != 4 or records.shape[-1] != self.record_size:
            raise ValueError(f"records must have shape [B, T, K, {self.record_size}]")
        if records.shape[2] != self.records_per_token:
            raise ValueError("records_per_token does not match workspace record tensor")
        if commit_mask.shape != records.shape[:3]:
            raise ValueError("commit_mask must match record batch/time axes")
        batch, time, records_per_token, _ = records.shape
        state = (
            self.reset_state(batch, device=records.device, dtype=records.dtype)
            if state is None
            else state
        )
        if state.keys.shape[:2] != (batch, self.heads):
            raise ValueError("workspace KV state batch/head dimensions do not match records")
        old = state.keys.shape[-2]
        flat_records = records.reshape(batch, time * records_per_token, self.record_size)
        flat_mask = commit_mask.reshape(batch, time * records_per_token)
        safe_records = flat_records * flat_mask.unsqueeze(-1).to(records.dtype)
        keys = (
            self.key(safe_records)
            .view(batch, time * records_per_token, self.heads, self.key_size)
            .transpose(1, 2)
        )
        values = (
            self.value(safe_records)
            .view(batch, time * records_per_token, self.heads, self.value_size)
            .transpose(1, 2)
        )
        keys = self._rope(keys, old)
        valid = torch.cat(
            (state.valid, flat_mask.to(device=records.device, dtype=torch.bool)), dim=-1
        )
        return WorkspaceKVState(
            torch.cat((state.keys.to(keys.dtype), keys), dim=-2),
            torch.cat((state.values.to(values.dtype), values), dim=-2),
            valid,
        )

    def read_prior(
        self,
        queries: torch.Tensor,
        state: WorkspaceKVState,
        *,
        query_start_position: int,
    ) -> torch.Tensor:
        """Read prior records with a strict same-token exclusion (lag=1)."""
        if queries.ndim != 3 or queries.shape[-1] != self.input_size:
            raise ValueError(f"queries must have shape [B, T, {self.input_size}]")
        batch, time, _ = queries.shape
        if state.keys.shape[0] != batch or state.valid.shape != (batch, state.keys.shape[-2]):
            raise ValueError("workspace KV state shape does not match query batch")
        query = self.query(queries).view(batch, time, self.heads, self.key_size).transpose(1, 2)
        query = self._rope(query, query_start_position)
        key_count = state.keys.shape[-2]
        if key_count == 0 or time == 0:
            return queries.new_zeros((batch, time, self.input_size))
        query_positions = query_start_position + self.records_per_token * torch.arange(
            time, device=queries.device
        )
        key_positions = torch.arange(key_count, device=queries.device)
        allowed = (key_positions[None, :] < query_positions[:, None])[
            None, None, :, :
        ] & state.valid[:, None, None, :]
        has_prior = allowed.any(dim=-1, keepdim=True)
        safe_allowed = allowed | (~has_prior & (key_positions == 0)[None, None, None, :])
        read = F.scaled_dot_product_attention(
            query,
            state.keys.to(query.dtype),
            state.values.to(query.dtype),
            attn_mask=safe_allowed,
        )
        read = torch.where(has_prior, read, torch.zeros_like(read))
        read = read.transpose(1, 2).reshape(batch, time, self.heads * self.value_size)
        return self.output(read)
