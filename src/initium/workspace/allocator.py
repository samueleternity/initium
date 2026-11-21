"""Learned key allocator: per-token has-key / write flag + pointer.

Pointer candidates: earlier positions of the segment (self == NEW) and keys already held in carried slots.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class KeyAllocator(nn.Module):
    def __init__(self, d_model: int, keys: int):
        super().__init__()
        self.keys, self.d = int(keys), int(d_model)
        self.has, self.write = nn.Linear(d_model, 1), nn.Linear(d_model, 1)
        self.q = nn.Linear(d_model, d_model, bias=False)
        self.k = nn.Linear(d_model, d_model, bias=False)
        self.slot = nn.Linear(d_model, d_model, bias=False)
        self.key_emb = nn.Parameter(torch.randn(self.keys, d_model) / d_model**0.5)
        self.register_buffer("use_prob", torch.zeros(()))  # persisted anneal state

    def forward(self, feats, state):
        B, T, _ = feats.shape
        addr, content = state.split((self.keys, self.d), dim=-1)
        occ = addr.square().sum(-1) > 1e-12
        kid = addr.argmax(-1)
        occ_keys = (
            torch.zeros(B, self.keys, dtype=torch.int32, device=feats.device).scatter_add(
                1, kid, occ.int()
            )
            > 0
        )
        ck = torch.zeros(B, self.keys, self.d, device=feats.device, dtype=feats.dtype).scatter_add(
            1, kid.unsqueeze(-1).expand(-1, -1, self.d), self.slot(content) * occ.unsqueeze(-1)
        )
        q, s = self.q(feats), 1.0 / self.d**0.5
        pos = torch.einsum("btd,bsd->bts", q, self.k(feats)) * s
        pos = pos.masked_fill(~torch.ones(T, T, dtype=torch.bool, device=feats.device).tril(), -1e4)
        key = torch.einsum("btd,bkd->btk", q, ck + self.key_emb) * s
        key = key.masked_fill(~occ_keys.unsqueeze(1), -1e4)
        return (
            self.has(feats).squeeze(-1),
            self.write(feats).squeeze(-1),
            torch.cat((pos, key), -1),
            occ_keys,
        )

    def loss(self, out, ids, wm, valid):
        has, wr, ptr, occ_keys = out
        T = has.shape[1]
        t = torch.arange(T, device=ids.device)
        tgt = ids.ge(0) & valid
        same = (
            (ids.unsqueeze(2) == ids.unsqueeze(1))
            & tgt.unsqueeze(1)
            & (t[None, None, :] < t[None, :, None])
        )
        prev = (same * (t + 1)).amax(-1) - 1
        in_carry = occ_keys.gather(1, ids.clamp_min(0))
        target = torch.where(in_carry, T + ids, torch.where(prev >= 0, prev, t.expand_as(ids)))
        n, m = valid.sum().clamp_min(1), tgt.sum().clamp_min(1)
        l_has = (
            F.binary_cross_entropy_with_logits(has, tgt.float(), reduction="none") * valid
        ).sum() / n
        l_wr = (
            F.binary_cross_entropy_with_logits(wr, wm.float(), reduction="none") * tgt
        ).sum() / m
        l_ptr = (
            F.cross_entropy(ptr.transpose(1, 2), target.clamp_min(0), reduction="none") * tgt
        ).sum() / m
        with torch.no_grad():
            stats = {
                "pointer_acc": ((ptr.argmax(-1) == target) & tgt).sum() / m,
                "flag_acc": (
                    (((has > 0) == tgt) & valid).sum() / n + ((wr > 0) == wm).mul(tgt).sum() / m
                )
                * 0.5,
            }
        return l_has + l_wr + l_ptr, stats

    @torch.no_grad()
    def predict(self, out):
        has, wr, ptr, occ_keys = out
        B, T = has.shape
        t = torch.arange(T, device=has.device).expand(B, T)
        hv = has > 0
        has_prob = has.sigmoid()
        flag_conf = torch.where(hv, has_prob, 1.0 - has_prob)
        pointer_conf = ptr.softmax(-1).amax(-1)
        confidence = torch.minimum(
            flag_conf, torch.where(hv, pointer_conf, torch.ones_like(pointer_conf))
        )
        sel = torch.where(hv, ptr.argmax(-1), t)
        is_key = sel >= T
        root = torch.where(is_key, t, sel)
        for _ in range(max(1, (T - 1).bit_length())):
            root = root.gather(1, root)
        is_new = hv & (sel == t)
        # New semantic addresses must use actual free ids. Counting occupied
        # keys is incorrect when a carried workspace contains sparse key ranges.
        new_roots = is_new & root.eq(t)
        new_rank = new_roots.long().cumsum(1) - 1
        free_ids = (~occ_keys).long().argsort(dim=1, descending=True)
        free_count = (~occ_keys).sum(1, keepdim=True)
        new_key = free_ids.gather(1, new_rank.clamp_min(0).clamp_max(self.keys - 1))
        new_key = torch.where(new_rank < free_count, new_key, torch.zeros_like(new_key))
        key_node = torch.where(is_key, sel - T, new_key)
        root_key_ref = is_key.gather(1, root)
        root_has_capacity = free_count.expand(-1, T).gather(1, root).gt(0)
        valid_key = hv & (root_key_ref | root_has_capacity)
        pid = torch.where(
            valid_key, key_node.gather(1, root).clamp(0, self.keys - 1), torch.full_like(t, -1)
        )
        return pid, (wr > 0) & hv, confidence
