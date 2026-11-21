"""Dense parallel workspace with chunked affine recurrence and soft readback."""

from __future__ import annotations

import math
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F


def _affine_scan(
    a: torch.Tensor,
    b: torch.Tensor,
    c0: torch.Tensor | None = None,
    chunk_size: int = 32,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Chunked affine recurrence c_t = a_t * c_{t-1} + b_t in fp32.

    Computes the recurrence in parallel across tokens using intra-chunk lower-triangular
    matrix operations and an inter-chunk sequential scan over chunks.

    Args:
        a: Multiplicative decay factors, shape (B, T, S).
        b: Additive update vectors, shape (B, T, S, D).
        c0: Initial carried state, shape (B, S, D). If None, defaults to zeros.
        chunk_size: Intra-chunk block size C.

    Returns:
        c: Full slot state sequence of shape (B, T, S, D).
        c_T: Carried final slot state of shape (B, S, D) at timestep T-1.
    """
    B, T, S, D = b.shape
    C = int(chunk_size)
    if C <= 0:
        raise ValueError("chunk_size must be positive")

    a_fp32 = a.float()
    b_fp32 = b.float()
    if c0 is None:
        c_prev = torch.zeros(B, S, D, device=b.device, dtype=torch.float32)
    else:
        c_prev = c0.float()

    pad_len = (C - (T % C)) % C
    if pad_len > 0:
        a_pad = torch.ones(B, pad_len, S, device=a.device, dtype=torch.float32)
        a_padded = torch.cat([a_fp32, a_pad], dim=1)
        b_pad = torch.zeros(B, pad_len, S, D, device=b.device, dtype=torch.float32)
        b_padded = torch.cat([b_fp32, b_pad], dim=1)
    else:
        a_padded = a_fp32
        b_padded = b_fp32

    total_T = a_padded.shape[1]
    Nc = total_T // C

    a_chunks = a_padded.view(B, Nc, C, S)
    b_chunks = b_padded.view(B, Nc, C, S, D)

    la = torch.log(a_chunks.clamp(min=1e-5)).clamp(min=-9.0)
    L = torch.cumsum(la, dim=2)

    diff = L.unsqueeze(3) - L.unsqueeze(2)
    causal_mask = torch.tril(torch.ones(C, C, device=a.device, dtype=torch.bool))
    decay = torch.where(
        causal_mask.view(1, 1, C, C, 1),
        torch.exp(diff),
        torch.zeros_like(diff),
    )

    c_local = torch.einsum("bnijs,bnjsd->bnisd", decay, b_chunks)

    exp_L = torch.exp(L)
    c_chunks_out: list[torch.Tensor] = []
    for n in range(Nc):
        c_chunk = c_local[:, n] + exp_L[:, n].unsqueeze(-1) * c_prev.unsqueeze(1)
        c_chunks_out.append(c_chunk)
        c_prev = c_chunk[:, -1]

    c_all = torch.cat(c_chunks_out, dim=1)
    c = c_all[:, :T]
    c_T = c[:, -1]
    return c, c_T


class ParallelWorkspace(nn.Module):
    """Dense parallel workspace reading residual stream features and updating slots."""

    def __init__(
        self,
        d_model: int,
        num_slots: int = 5,
        key_dim: int = 64,
        chunk: int = 32,
        read_heads: int = 1,
    ) -> None:
        super().__init__()
        if d_model <= 0 or num_slots <= 0 or key_dim <= 0 or chunk <= 0:
            raise ValueError("dimensions must be positive")
        if read_heads != 1:
            raise ValueError("read_heads must be 1 for current ParallelWorkspace")

        self.d_model = int(d_model)
        self.num_slots = int(num_slots)
        self.key_dim = int(key_dim)
        self.chunk = int(chunk)
        self.read_heads = int(read_heads)

        self.eval_bypass: bool = False
        self.eval_shuffle_readback: bool = False

        self.slot_keys = nn.Parameter(torch.empty(self.num_slots, self.key_dim))
        self.read_keys = nn.Parameter(torch.empty(self.num_slots, self.key_dim))

        self.Wk = nn.Linear(self.d_model, self.key_dim)
        self.Wv = nn.Linear(self.d_model, self.d_model)
        self.Wu = nn.Linear(self.d_model, self.num_slots)
        self.Wq = nn.Linear(self.d_model, self.key_dim)
        self.Wo = nn.Linear(self.d_model, self.d_model)

        self.read_gate = nn.Parameter(torch.zeros(()))
        self.readout_norm = nn.LayerNorm(self.d_model, elementwise_affine=False)

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.normal_(self.slot_keys, std=1.0 / math.sqrt(self.key_dim))
        nn.init.normal_(self.read_keys, std=1.0 / math.sqrt(self.key_dim))
        nn.init.xavier_uniform_(self.Wk.weight)
        nn.init.zeros_(self.Wk.bias)
        nn.init.xavier_uniform_(self.Wv.weight)
        nn.init.zeros_(self.Wv.bias)
        nn.init.xavier_uniform_(self.Wu.weight)
        nn.init.zeros_(self.Wu.bias)
        nn.init.xavier_uniform_(self.Wq.weight)
        nn.init.zeros_(self.Wq.bias)
        nn.init.zeros_(self.Wo.weight)
        nn.init.zeros_(self.Wo.bias)
        nn.init.zeros_(self.read_gate)

    def init_state(
        self,
        batch_size: int,
        *,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        """Create a zeroed carried slot state tensor of shape (B, S, D)."""
        return torch.zeros(
            batch_size,
            self.num_slots,
            self.d_model,
            device=device,
            dtype=dtype or torch.float32,
        )

    def forward(
        self,
        h: torch.Tensor,
        state: torch.Tensor | None = None,
        *,
        valid_mask: torch.Tensor | None = None,
        bypass: bool | None = None,
        shuffle_readback: bool | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
        """Forward pass for dense parallel workspace across all (B, T) timesteps.

        Args:
            h: Input features from backbone residual stream, shape (B, T, D).
            state: Initial slot state, shape (B, S, D). Defaults to zeros if None.
            valid_mask: Boolean or binary mask of valid tokens, shape (B, T) or (B, T, 1).
                Zero-padded tokens do not write to slots (u = 0).
            bypass: Override for evaluation broadcast bypass.
            shuffle_readback: Override for evaluation batch readback shuffling.

        Returns:
            delta_ws: Additive residual stream update of shape (B, T, D).
            c_T: Carried final slot state of shape (B, S, D).
            aux: Telemetry dictionary holding r_t, c, alpha, w, u.
        """
        B, T, _ = h.shape
        h_fp32 = h.float()

        # 1. Assignment: w = softmax_s(Wk(h) * κ^T / sqrt(key_dim)), shape (B, T, S)
        k_proj = self.Wk(h_fp32)
        w_scores = torch.matmul(k_proj, self.slot_keys.t()) / math.sqrt(self.key_dim)
        w = F.softmax(w_scores, dim=-1)

        # 2. Gate: u = σ(Wu(h)) * w, clamped to [0, 1 - 1e-3]
        u = torch.sigmoid(self.Wu(h_fp32)) * w
        u = u.clamp(min=0.0, max=1.0 - 1e-3)
        if valid_mask is not None:
            mask = valid_mask.unsqueeze(-1) if valid_mask.ndim == 2 else valid_mask
            u = u * mask.to(dtype=u.dtype)

        # 3. Proposal: p = Wv(h), shape (B, T, D)
        p = self.Wv(h_fp32)

        # 4. Recurrence per slot: c_t = (1 - u_t) * c_{t-1} + u_t * p_t
        a = 1.0 - u
        b = u.unsqueeze(-1) * p.unsqueeze(-2)
        c, c_T = _affine_scan(a, b, c0=state, chunk_size=self.chunk)

        # 5. Read: α = softmax_s(Wq(h) * κ'^T / sqrt(key_dim))
        q_proj = self.Wq(h_fp32)
        q_scores = torch.matmul(q_proj, self.read_keys.t()) / math.sqrt(self.key_dim)
        alpha = F.softmax(q_scores, dim=-1)

        c_norm = self.readout_norm(c)
        r_t = torch.einsum("bts,btsd->btd", alpha, c_norm)

        # Handle bypass and shuffle evaluation conditions
        is_bypass = self.eval_bypass if bypass is None else bool(bypass)
        is_shuffle = (
            self.eval_shuffle_readback
            if shuffle_readback is None
            else bool(shuffle_readback)
        )

        c_aux = c
        if is_shuffle and B > 1:
            r_t = torch.roll(r_t, shifts=1, dims=0)
            c_aux = torch.roll(c_aux, shifts=1, dims=0)

        delta_ws = torch.sigmoid(self.read_gate) * self.Wo(r_t)

        if is_bypass:
            delta_ws = torch.zeros_like(delta_ws)
            r_t = torch.zeros_like(r_t)
            c_aux = torch.zeros_like(c_aux)

        delta_ws = delta_ws.to(dtype=h.dtype)

        aux: dict[str, Any] = {
            "r_t": r_t,
            "c": c_aux.detach(),
            "alpha": alpha.detach(),
            "w": w.detach(),
            "u": u.detach(),
        }

        return delta_ws, c_T, aux
