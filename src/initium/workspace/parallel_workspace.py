"""Dense, fully parallel workspace: per-token slot writes via a chunked affine scan."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _affine_scan(
    a: torch.Tensor,
    b: torch.Tensor,
    c0: torch.Tensor,
    chunk: int,
    tril_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """c_t = a_t * c_{t-1} + b_t. a: (B,T,S), b: (B,T,S,D), c0: (B,S,D) -> (B,T,S,D)."""
    B, T, S, D = b.shape
    pad = (-T) % chunk
    if pad:  # identity padding at the end (a=1, b=0)
        a = F.pad(a, (0, 0, 0, pad), value=1.0)
        b = F.pad(b, (0, 0, 0, 0, 0, pad))
    n = (T + pad) // chunk
    L = a.log().clamp_min(-9.0).view(B, n, chunk, S).cumsum(2)
    diff = L.unsqueeze(3) - L.unsqueeze(2)  # (B,n,i,j,S) = L_i - L_j
    if tril_mask is None:
        tril_mask = torch.ones(chunk, chunk, dtype=torch.bool, device=b.device).tril()
    tril = tril_mask.view(1, 1, chunk, chunk, 1)
    decay = diff.masked_fill(~tril, float("-inf")).exp()
    loc = torch.einsum("bnijs,bnjsd->bnisd", decay, b.view(B, n, chunk, S, D))
    tot, ends = L[:, :, -1].exp(), loc[:, :, -1]
    prev, carry = [], c0
    for i in range(n):  # n = T/chunk iterations only
        prev.append(carry)
        carry = tot[:, i, :, None] * carry + ends[:, i]
    c = loc + L.exp().unsqueeze(-1) * torch.stack(prev, 1).unsqueeze(2)
    return c.reshape(B, n * chunk, S, D)[:, :T]


class ParallelWorkspace(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_slots: int = 5,
        key_dim: int = 64,
        chunk: int = 32,
        moe_feed: bool = False,
    ):
        super().__init__()
        if min(d_model, num_slots, key_dim, chunk) < 1:
            raise ValueError("parallel workspace dimensions must be positive")
        self.d_model, self.num_slots, self.key_dim, self.chunk = d_model, num_slots, key_dim, chunk
        self.scale = key_dim**-0.5
        # This mask is invariant across batches and steps. A non-persistent
        # buffer follows .to(device) without changing checkpoint state dicts.
        self.register_buffer(
            "_scan_tril_mask",
            torch.ones(chunk, chunk, dtype=torch.bool).tril(),
            persistent=False,
        )
        self.slot_keys = nn.Parameter(torch.randn(num_slots, key_dim))
        self.read_keys = nn.Parameter(torch.randn(num_slots, key_dim))
        self.Wk = nn.Linear(d_model, key_dim)
        self.Wv = nn.Linear(d_model, d_model)
        self.Wu = nn.Linear(d_model, num_slots)
        self.Wq = nn.Linear(d_model, key_dim)
        self.Wo = nn.Linear(d_model, d_model)
        self.read_gate = nn.Parameter(torch.zeros(()))
        self.readout_norm = nn.LayerNorm(d_model, elementwise_affine=False)
        self.probe = nn.Linear(d_model, 16)
        nn.init.zeros_(self.Wo.weight)
        nn.init.zeros_(self.Wo.bias)
        self.moe_feed = bool(moe_feed)
        if self.moe_feed:
            self.moe_norm = nn.LayerNorm(d_model)
            self.Wm = nn.Linear(d_model, d_model, bias=False)
            nn.init.zeros_(self.Wm.weight)

    def init_state(self, batch: int, device=None) -> torch.Tensor:
        return torch.zeros(batch, self.num_slots, self.d_model, device=device, dtype=torch.float32)

    def forward(self, h, state, valid_mask=None, bypass=False, shuffle_readback=False, routed=None):
        seg = 128
        if torch.is_grad_enabled() or h.shape[1] <= seg:
            return self._forward_seg(h, state, valid_mask, bypass, shuffle_readback, routed)
        deltas, readouts = [], []
        for s in range(0, h.shape[1], seg):
            d, state, aux = self._forward_seg(
                h[:, s : s + seg],
                state,
                None if valid_mask is None else valid_mask[:, s : s + seg],
                bypass,
                shuffle_readback,
                None if routed is None else routed[:, s : s + seg],
            )
            deltas.append(d)
            if "readout" in aux:
                readouts.append(aux["readout"])
        return (
            torch.cat(deltas, 1),
            state,
            ({"readout": torch.cat(readouts, 1)} if readouts else {}),
        )

    def _forward_seg(
        self, h, state, valid_mask=None, bypass=False, shuffle_readback=False, routed=None
    ):
        """h: (B,T,D); state: (B,S,D). Returns (delta (B,T,D) fp32, new_state, aux)."""
        with torch.autocast(device_type=h.device.type, enabled=False):
            h = h.float()
            rd = None if routed is None else self.Wm(self.moe_norm(routed.detach().float()))
            src = h
            w = (self.Wk(h) @ self.slot_keys.t() * self.scale).softmax(-1)
            u = (torch.sigmoid(self.Wu(h)) * w).clamp(max=1.0 - 1e-3)
            if valid_mask is not None:  # padding never writes
                u = u * valid_mask.unsqueeze(-1).to(u.dtype)
            c = _affine_scan(
                1.0 - u,
                u.unsqueeze(-1) * self.Wv(src).unsqueeze(2),
                state.float(),
                self.chunk,
                self._scan_tril_mask,
            )
            new_state = c[:, -1]
            if bypass:
                return torch.zeros_like(h), new_state, {}
            cr = self.readout_norm(c)
            if shuffle_readback and cr.shape[0] > 1:
                cr = cr.roll(1, 0)
            alpha = (
                self.Wq(h if rd is None else h + rd) @ self.read_keys.t() * self.scale
            ).softmax(-1)
            r = torch.einsum("bts,btsd->btd", alpha, cr)
            delta = torch.sigmoid(self.read_gate) * self.Wo(r)
            self.last_diag = {
                "delta_rms": delta.detach().square().mean().sqrt(),
                "Wo_rms": self.Wo.weight.detach().square().mean().sqrt(),
                "read_gate": torch.sigmoid(self.read_gate.detach()),
                **(
                    {"Wm_rms": self.Wm.weight.detach().square().mean().sqrt()}
                    if self.moe_feed
                    else {}
                ),
            }
        aux = {"readout": r}
        if torch.is_grad_enabled():
            si = w.argmax(-1)
            aux["written"] = c.gather(
                2, si[..., None, None].expand(-1, -1, 1, c.shape[-1])
            ).squeeze(2)
        return delta, new_state, aux
