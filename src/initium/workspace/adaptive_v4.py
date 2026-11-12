"""Adaptive workspace V4: primitive operators, set reductions, chained multi-operation events."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F
import math

from initium.workspace.adaptive_v3 import AdaptiveWorkspaceOperatorBank as _V3

_NAMES = (
    "noop", "copy", "gt", "sum", "max", "min", "count_even", "parity",
    "sub", "and", "or", "xor", "not", "mul", "neg", "inc", "dec", "eq", "member", "cond_copy", "halt",
)
(_COPY, _GT, _SUM, _MAX, _MIN, _CNT, _PAR, _SUB, _AND, _OR, _XOR, _NOT, _MUL, _NEG, _INC, _DEC, _EQ, _MEM, _CW, _HALT) = range(1, 21)
_LEFT = (_COPY, _GT, _SUB, _AND, _OR, _XOR, _NOT, _MUL, _NEG, _INC, _DEC, _EQ, _MEM, _CW)
_RIGHT = (_GT, _SUB, _AND, _OR, _XOR, _MUL, _EQ, _CW)
_LIT, _RUN, _NF = 64, 128, 10


def _tree_prod(f):
    while f.shape[1] > 1:
        if f.shape[1] % 2:
            f = torch.cat((f, torch.ones_like(f[:, :1])), 1)
        f = f[:, 0::2] * f[:, 1::2]
    return f[:, 0]


def _norm(x):
    x = x.clamp_min(0)
    return x / x.sum(-1, keepdim=True).clamp_min(1e-6)


def _ctree(re, im):
    while re.shape[1] > 1:
        if re.shape[1] % 2:
            re = torch.cat((re, torch.ones_like(re[:, :1])), 1)
            im = torch.cat((im, torch.zeros_like(im[:, :1])), 1)
        re, im = (re[:, 0::2] * re[:, 1::2] - im[:, 0::2] * im[:, 1::2],
                  re[:, 0::2] * im[:, 1::2] + im[:, 0::2] * re[:, 1::2])
    return re[:, 0], im[:, 0]


def _sum_dist(P, m, C, S):
    mm = m.unsqueeze(-1)
    re, im = _ctree((1 - mm) + mm * (P @ C), -mm * (P @ S))
    return _norm((re @ C - im @ S) / C.shape[0])


def _count_dist(P, m, C, S):
    q = (m * P[..., 0::2].sum(-1)).unsqueeze(-1)
    re, im = _ctree((1 - q) * C[0] + q * C[1], -q * S[1])
    return _norm((re @ C - im @ S) / C.shape[0])


def _sub_dist(a, b, C, S):
    ar, ai, br, bi = a @ C, -(a @ S), b @ C, -(b @ S)
    re, im = ar * br + ai * bi, ai * br - ar * bi
    return _norm((re @ C - im @ S) / C.shape[0])


def _extreme_dist(P, m, largest):
    s = P.cumsum(-1) if largest else P.flip(-1).cumsum(-1).flip(-1)
    t = (1 - m).unsqueeze(-1) + m.unsqueeze(-1) * s
    g = torch.exp(torch.log(t.clamp_min(1e-6)).sum(1))
    pmf = g - F.pad(g[:, :-1], (1, 0)) if largest else g - F.pad(g[:, 1:], (0, 1))
    return _norm(pmf)


def _voc(x, base):
    return F.pad(x, (base, 320 - base - x.shape[-1]))


class AdaptiveWorkspaceOperatorBankV4(nn.Module):
    operator_names = _NAMES
    max_ops = 2 # if needed set higher, but it will tank the performance 
    teacher_width = 44
    use_backbone_context = False
    begin_sequence = _V3.begin_sequence
    last_route_steps = _V3.last_route_steps
    pop_aux_loss = _V3.pop_aux_loss
    last_diagnostics = _V3.last_diagnostics
    _accumulate_diagnostics = _V3._accumulate_diagnostics
    _target_slot_indices = staticmethod(_V3._target_slot_indices)
    _entity_destination_weights = staticmethod(_V3._entity_destination_weights)

    def __init__(self, d_model: int, address_dim: int, compile_ops: bool = False) -> None:
        super().__init__()
        self.d_model, self.address_dim, self.scale = int(d_model), int(address_dim), 12.0
        self.compile_ops, self._compiled, self._failed = bool(compile_ops), {}, set()
        K, A, C, w = self.max_ops, self.address_dim, len(_NAMES), 3 * d_model
        self.event_embedding = nn.Embedding(320, d_model)
        self.event_proj = nn.Linear(8 * d_model, d_model)
        self.ctx_norm = nn.LayerNorm(w)
        self.sub_embed = nn.Embedding(K, w)
        self.router = nn.Linear(w, C)
        self.bind = nn.Linear(w, 3 * A, bias=False)
        self.dest_head = nn.Linear(w, A)
        self.literal_key = nn.Parameter(torch.randn(A))
        self.dest_gain = nn.Parameter(torch.zeros(K, 5))
        self.left_gain = nn.Parameter(torch.zeros(K, _NF))
        self.right_gain = nn.Parameter(torch.zeros(K, _NF))
        self.set_gain = nn.Parameter(torch.tensor([4.0, 0.0]))
        self.set_blk = nn.Parameter(torch.tensor([-2.0, -6.0, -2.0]))
        nn.init.zeros_(self.sub_embed.weight)
        nn.init.zeros_(self.dest_head.weight)
        nn.init.zeros_(self.dest_head.bias)
        nn.init.xavier_uniform_(self.bind.weight)
        with torch.no_grad():
            self.router.bias.zero_()
            self.router.bias[0] = 3.0
            self.dest_gain[0, 0] = 4.0
            self.left_gain[0, 1] = 4.0
            self.right_gain[0, 8] = 4.0
            self.left_gain[1:, 9] = 3.0  # later sub-ops default to reading the previous result
        self.eval_bypass = False
        for n in (16, 32):
            t = 2 * math.pi * torch.outer(torch.arange(n), torch.arange(n)).double() / n
            self.register_buffer(f"C{n}", t.cos().float(), persistent=False)
            self.register_buffer(f"S{n}", t.sin().float(), persistent=False)
        self._last_aux_loss = None
        self._last_training_diagnostics = {}
        self._last_training_diagnostic_counts = {}
        self._last_route_steps = []
        self._route_sum = None
        self._route_n = 0.0

    def parse_steps(self, tokens):
        """Parse all candidate steps in one batched call. tokens: (n, B, L)."""
        n, B, L = tokens.shape
        ps = self._parse(tokens.long().reshape(n * B, L))
        return {
            k: (torch.stack(v, 0).view(4, n, B).transpose(0, 1) if k == "a" else v.view(n, B, *v.shape[1:]))
            for k, v in ps.items()
        }

    @staticmethod
    def pick_parse(pre, i):
        return {k: ([v[i][j] for j in range(4)] if k == "a" else v[i]) for k, v in pre.items()}

    def _call(self, key, fn, *args):
        if not (self.compile_ops and self.training and args[0].is_cuda) or key in self._failed:
            return fn(*args)
        c = self._compiled.get(key)
        if c is None:
            c = self._compiled[key] = torch.compile(
                fn, fullgraph=True, dynamic=False, options={"max_autotune": False}
            )
        try:
            return c(*args)
        except Exception as exc:  # inductor/triton failure: keep the equivalent eager math
            self._failed.add(key)
            import warnings

            warnings.warn(f"V4 '{key}' compile failed ({type(exc).__name__}); using eager.")
            return fn(*args)

    def _dist_math(self, P, V, wl, wr, m, op_probs, W, bias, C16, S16, C32, S32):
        B, C = P.shape[0], len(_NAMES)
        dev = P.device
        PL, PR = (wl.unsqueeze(-1) * P).sum(1), (wr.unsqueeze(-1) * P).sum(1)
        l1, r1 = PL[:, 1], PR[:, 1]
        p_gt = (PL[:, 1:] * PR.cumsum(-1)[:, :-1]).sum(-1)
        vec_copy = (wl.unsqueeze(-1) * V).sum(1)
        d_copy = F.softmax(F.linear(vec_copy, W, bias), -1)
        tot = _sum_dist(P, m, C16, S16)
        odd = tot[:, 1::2].sum(-1)

        def bit(p):
            return _voc(torch.stack((1 - p, p), -1), 136)
        
        ar = torch.arange(16, device=dev)
        mul_idx = (ar[:, None] * ar[None, :] % 16).reshape(-1)
        d_mul = _voc(
            torch.zeros(B, 16, device=dev, dtype=P.dtype).index_add(
                1, mul_idx, (PL.unsqueeze(2) * PR.unsqueeze(1)).reshape(B, -1)
            ), 136)
        d_neg = _voc(PL[:, (-ar) % 16], 136)
        d_inc = _voc(PL.roll(1, -1), 136)
        d_dec = _voc(PL.roll(-1, -1), 136)
        p_eq = (PL * PR).sum(-1)
        match = (P * PL.unsqueeze(1)).sum(-1)
        p_in = 1 - torch.exp(torch.log((1 - m * match).clamp_min(1e-6)).sum(1))

        dists = torch.stack(
            (
                d_copy, bit(p_gt), _voc(tot, 136),
                _voc(_extreme_dist(P, m, True), 136), _voc(_extreme_dist(P, m, False), 136),
                _voc(_count_dist(P, m, C32, S32), 152), _voc(torch.stack((1 - odd, odd), -1), 184),
                _voc(_sub_dist(PL, PR, C16, S16), 136),
                bit(l1 * r1), bit(l1 + r1 - l1 * r1), bit(l1 + r1 - 2 * l1 * r1), bit(1 - l1),
                d_mul, d_neg, d_inc, d_dec, bit(p_eq), bit(p_in),
                d_copy,
            ),
            1,
        )
        hard = F.one_hot(dists.argmax(-1), 320).float()
        vecs = (hard + dists - dists.detach()) @ W
        vecs = torch.cat((vec_copy.unsqueeze(1), vecs[:, 1:_CW - 1], vec_copy.unsqueeze(1)), 1)
        vecs = vecs * (1.0 / vecs.detach().square().mean(-1, keepdim=True).sqrt().clamp_min(1.0))
        sel = F.one_hot(op_probs.argmax(-1), C).float()
        wsel = sel + op_probs - op_probs.detach()
        gate = (r1 > 0.5).float() + r1 - r1.detach()
        gates = torch.cat((torch.ones(B, _CW - 1, device=dev), gate.unsqueeze(1)), 1)
        out_vec = (wsel[:, 1 : C - 1].unsqueeze(-1) * vecs).sum(1)
        return dists, r1, out_vec, sel, wsel, gates

    def _parse(self, tok):
        """Input-only event parse: distinct named entities in order, literal, reference set."""
        A = self.address_dim
        B, L = tok.shape
        idx = torch.arange(L, device=tok.device).unsqueeze(0).expand(B, L)
        marker = (tok == 4) | ((tok >= 8) & (tok <= 23))
        last = torch.where(marker, idx, torch.full_like(idx, -1)).amax(-1, keepdim=True)
        after = idx > last
        ent = tok - 32
        named = after & (ent >= 0) & (ent < A)
        dup = (
            (ent.unsqueeze(2) == ent.unsqueeze(1))
            & named.unsqueeze(1)
            & (idx.unsqueeze(1) < idx.unsqueeze(2))
        ).any(-1)
        distinct = named & ~dup
        order = distinct.long().cumsum(-1) - 1
        a = [((ent + 1) * (distinct & order.eq(j)).long()).sum(-1) - 1 for j in range(4)]
        a0p = torch.where((a[0] >= 0) & (a[0] + 1 < A), a[0] + 1, torch.full_like(a[0], -1))
        tok_m = tok.gather(1, last.clamp(min=0)).squeeze(1)
        tok_n = tok.gather(1, (last + 1).clamp(0, L - 1)).squeeze(1)
        is_q = (last.squeeze(1) >= 0) & ((tok_m == 4) | ((tok_m >= 20) & (tok_m <= 22)))
        ref_all = (
            torch.zeros(B, A, device=tok.device).scatter_add(1, ent.clamp(0, A - 1), named.float()) > 0
        )
        first = F.one_hot(a[0].clamp(min=0), A).bool() & (a[0] >= 0).unsqueeze(-1)
        ref = torch.where(is_q.unsqueeze(-1), ref_all, ref_all & ~first).float()
        sc = after & (tok >= 136) & (tok < 152)
        lp = torch.where(sc, idx, torch.full_like(idx, -1)).amax(-1)
        lit = tok.gather(1, lp.clamp(min=0).unsqueeze(1)).squeeze(1)
        return dict(a=a, a0p=a0p, ref=ref, has_lit=lp >= 0, lit_tok=lit, tok_m=tok_m, tok_n=tok_n)

    def step(
        self, *, state, context, backbone_context, event_tokens, semantic_value_decoder,
        active_mask, execution_mask, teacher_targets, teacher_weight, step_index, parsed=None,
    ):
        if self.eval_bypass:
            return state
        with torch.autocast(device_type=state.device.type, enabled=False):
            out = self._step(
                state, context, event_tokens, semantic_value_decoder, active_mask,
                execution_mask, teacher_targets, teacher_weight, step_index, parsed,
            )
        return out.to(dtype=state.dtype)

    def _step(self, state, context, event_tokens, decoder, active_mask, execution_mask,
              teacher_targets, teacher_weight, step_index, parsed=None):
        A, D, K, C, s = self.address_dim, self.d_model, self.max_ops, len(_NAMES), self.scale
        state = state.float()
        B, S, _ = state.shape
        dev = state.device
        address, content = torch.split(state, [A, D], dim=-1)
        occupied = address.square().sum(-1).gt(1e-12)
        aid = address.argmax(-1)
        W = decoder.weight.float()
        tok = event_tokens.long()
        ps = parsed if parsed is not None else self._call("parse", self._parse, tok)
        a, has_lit, lit_tok = ps["a"], ps["has_lit"], ps["lit_tok"]
        a_t = torch.stack(a, -1)  # (B,4)
        lit_f = has_lit.float()
        P_lit = F.one_hot((lit_tok - 136).clamp(0, 15), 16).float().mul(lit_f.unsqueeze(-1)).unsqueeze(1)
        V_lit = (W[lit_tok.clamp(0, 319)] * lit_f.unsqueeze(-1)).unsqueeze(1)
        P_snap = decoder(content)[..., 136:152].softmax(-1)

        def oh(x):
            return F.one_hot(x.clamp(min=0), A).float() * (x >= 0).float().unsqueeze(-1)

        def match(ids, occ):
            return ((ids.unsqueeze(-1) == a_t.unsqueeze(1)) & occ.unsqueeze(-1) & a_t.ge(0).unsqueeze(1)).float()

        ev_ids = torch.cat((tok[:, -6:], ps["tok_m"].unsqueeze(1), ps["tok_n"].unsqueeze(1)), 1)
        ev = self.event_proj(self.event_embedding(ev_ids).flatten(1))
        base = self.ctx_norm(torch.cat((context.float(), ev), -1))
        active = active_mask.bool() & execution_mask.bool()
        run_addr, run_content = address, content
        alive = torch.ones(B, device=dev)
        prev_slot = torch.zeros(B, S, device=dev)
        prev_dest = torch.zeros(B, A, device=dev)
        set_bias = torch.cat(
            (self.set_blk[0].expand(S), self.set_blk[1].expand(S), self.set_blk[2:3])
        )
        losses, route_ops = [], []
        use_teacher = teacher_targets is not None and teacher_weight > 0
        mean = lambda x, mk: (x * mk).sum() / mk.sum().clamp_min(1)  # noqa: E731
        acc = lambda hit, mk: (hit & mk).sum().float() / mk.sum().clamp_min(1)  # noqa: E731

        for k in range(K):
            run_occ = run_addr.square().sum(-1).gt(1e-12)
            aid_r = run_addr.argmax(-1)

            ctx = base + self.sub_embed.weight[k]
            op_logits = self.router(ctx)
            op_probs = op_logits.softmax(-1)
            route_ops.append(op_probs)

            if k == 0:
                _al = self.router(base.unsqueeze(1) + self.sub_embed.weight)
                need_t = (_al.argmax(-1).gt(0) & active.unsqueeze(1)).any()
                if use_teacher:
                    need_t = need_t | teacher_targets[:, 0 : K * 11 : 11].gt(0).any()
                need = bool(need_t)
            occ_c = torch.cat((occupied, run_occ, has_lit.unsqueeze(1)), 1)
            addr_c = F.normalize(
                torch.cat((address, run_addr, self.literal_key.view(1, 1, A).expand(B, 1, A)), 1), dim=-1
            )
            feat = torch.zeros(B, 2 * S + 1, _NF, device=dev)
            feat[:, :S, :4] = match(aid, occupied)
            feat[:, S : 2 * S, 4:8] = match(aid_r, run_occ)
            feat[:, S : 2 * S, 9] = prev_slot
            feat[:, 2 * S, 8] = lit_f
            ref_c = torch.cat(
                (
                    ps["ref"].gather(1, aid) * occupied.float(),
                    ps["ref"].gather(1, aid_r) * run_occ.float(),
                    lit_f.unsqueeze(1),
                ),
                1,
            )

            q = F.normalize(self.bind(ctx).view(B, 3, A), dim=-1)
            cs = torch.einsum("bqa,bsa->bqs", q, addr_c)
            left_logits = (s * cs[:, 0] + feat @ self.left_gain[k]).masked_fill(~occ_c, -40.0)
            right_logits = (s * cs[:, 1] + feat @ self.right_gain[k]).masked_fill(~occ_c, -40.0)
            set_logits = self.set_gain[0] * ref_c + self.set_gain[1] * s * cs[:, 2] + set_bias
            wl, wr = left_logits.softmax(-1), right_logits.softmax(-1)
            ms = torch.sigmoid(set_logits) * occ_c.float()
            m = ((ms > 0.5).float() + ms - ms.detach()) * occ_c.float()

            if need:
                if k == 0:
                    P_run, V_run = P_snap, content
                else:
                    P_run, V_run = decoder(run_content)[..., 136:152].softmax(-1), run_content
                P = torch.cat((P_snap, P_run, P_lit), 1)
                V = torch.cat((content, V_run, V_lit), 1)
                dists, r1, out_vec, sel, wsel, gates = self._call(
                    "dist", self._dist_math, P, V, wl, wr, m, op_probs, W, decoder.bias.float(),
                    self.C16, self.S16, self.C32, self.S32,
                )
                mass = (wsel[:, 1 : C - 1] * gates).sum(-1) * active.float() * alive
            else:
                r1 = left_logits.new_zeros(B)
                dists = left_logits.new_zeros(B, C - 2, 320)
                out_vec = left_logits.new_zeros(B, D)
                mass = left_logits.new_zeros(B)
                sel = F.one_hot(op_probs.argmax(-1), C).float()
                wsel = sel + op_probs - op_probs.detach()

            ent_logits = torch.einsum(
                "j,bja->ba", self.dest_gain[k],
                torch.stack((oh(a[0]), oh(a[1]), oh(a[2]), oh(ps["a0p"]), prev_dest), 1),
            ) + self.dest_head(ctx)
            run_occ_d = run_addr.square().sum(-1).gt(1e-12)
            slot_w, dest_ent, _slot, allocate = self._entity_destination_weights(
                ent_logits, run_addr, run_occ_d, temperature=s, straight_through=True
            )
            dm = (mass.unsqueeze(-1) * slot_w).unsqueeze(-1)
            writes = sel[:, 1 : C - 1].sum(-1).gt(0) & (alive > 0.5)
            new = (allocate & active & writes).view(B, 1, 1) & slot_w.detach().gt(0).unsqueeze(-1)
            run_content = run_content * (1 - dm) + out_vec.unsqueeze(1) * dm
            run_addr = torch.where(new, F.one_hot(dest_ent, A).float().unsqueeze(1), run_addr)
            wrote = (mass.detach() > 0.5).float()
            prev_slot = (slot_w.detach() > 0.5).float() * wrote.unsqueeze(-1)
            prev_dest = F.one_hot(dest_ent, A).float() * wrote.unsqueeze(-1)
            alive = alive * (1 - wsel[:, C - 1] * active.float())

            if not use_teacher:
                continue
            t = teacher_targets[:, k * 11 : (k + 1) * 11].long()
            op_t, dest_t, out_t, keys = t[:, 0], t[:, 1], t[:, 2], t[:, 3:]
            pos, zero, valid = op_t.gt(0), op_t.eq(0), op_t.ge(0)
            ce = F.cross_entropy(op_logits, op_t.clamp(0, C - 1), reduction="none")
            present = [pos & op_t.eq(c) for c in range(1, C)]
            cls = torch.stack([mean(ce, mk) for mk in present])
            has = torch.stack([mk.any().float() for mk in present])
            op_loss = 0.5 * (cls * has).sum() / has.sum().clamp_min(1) + 0.5 * mean(ce, zero)

            def slot_tgt(key):
                run = key.ge(_RUN)
                kk = torch.where(run, key - _RUN, key).clamp(0, A - 1)
                t1, f1 = self._target_slot_indices(address, occupied, kk)
                t2, f2 = self._target_slot_indices(run_addr, run_occ, kk)
                lit = key.eq(_LIT)
                tgt = torch.where(lit, torch.full_like(t1, 2 * S), torch.where(run, t2 + S, t1))
                return tgt, (torch.where(run, f2, f1) | lit) & key.ge(0)

            lt, lf = slot_tgt(keys[:, 0])
            rt, rf = slot_tgt(keys[:, 1])
            lv = torch.isin(op_t, op_t.new_tensor(_LEFT)) & lf
            rv = torch.isin(op_t, op_t.new_tensor(_RIGHT)) & rf
            lce = F.cross_entropy(left_logits, lt.masked_fill(~lv, -100), reduction="none", ignore_index=-100)
            rce = F.cross_entropy(right_logits, rt.masked_fill(~rv, -100), reduction="none", ignore_index=-100)
            kv = keys.ge(0)
            # member(x, S): keys[0] is the probe operand, not a set element.
            keys = torch.cat((torch.where(op_t.eq(_MEM), torch.full_like(keys[:, 0], -1), keys[:, 0]).unsqueeze(1), keys[:, 1:]), 1)
            tset = torch.cat(
                (
                    ((aid.unsqueeze(-1) == keys.unsqueeze(1)) & ((keys >= 0) & (keys < _LIT)).unsqueeze(1)).any(-1) & occupied,
                    ((aid_r.unsqueeze(-1) == (keys - _RUN).unsqueeze(1)) & keys.ge(_RUN).unsqueeze(1)).any(-1) & run_occ,
                    keys.eq(_LIT).any(-1, keepdim=True),
                ),
                1,
            ).float()
            sv = (op_t.ge(_SUM) & op_t.le(_PAR) | op_t.eq(_MEM)) & kv.any(-1)
            bce = (F.binary_cross_entropy_with_logits(set_logits, tset, reduction="none") * occ_c.float()).sum(-1)
            wr_op = pos & op_t.lt(_HALT)
            dv = wr_op & dest_t.ge(0)
            dce = F.cross_entropy(s * ent_logits, dest_t.clamp(0, A - 1), reduction="none")
            d_lab = dists.gather(1, (op_t.clamp(1, C - 2) - 1).view(B, 1, 1).expand(B, 1, 320)).squeeze(1)
            nll = -d_lab.gather(1, out_t.clamp(0, 319).unsqueeze(1)).squeeze(1).clamp_min(1e-6).log()
            ov = wr_op & out_t.ge(0)
            cw = op_t.eq(_CW)
            gl = F.binary_cross_entropy(r1.clamp(1e-4, 1 - 1e-4), out_t.ge(0).float(), reduction="none")
            losses += [op_loss, mean(lce, lv), mean(rce, rv), mean(bce, sv), mean(dce, dv), mean(nll, ov), mean(gl, cw)]
            if self.training:
                pred = op_logits.argmax(-1)
                cnt = lambda mk: mk.sum().detach()  # noqa: E731
                diag = {
                    "teacher_operation_loss": op_loss.detach(), "teacher_operation_loss_count": cnt(valid),
                    "teacher_operation_accuracy": acc(pred.eq(op_t), valid), "teacher_operation_accuracy_count": cnt(valid),
                    "teacher_active_operation_accuracy": acc(pred.eq(op_t), pos), "teacher_active_operation_accuracy_count": cnt(pos),
                    "teacher_noop_accuracy": acc(pred.eq(0), zero), "teacher_noop_accuracy_count": cnt(zero),
                    "teacher_operand0_loss": mean(lce, lv).detach(), "teacher_operand0_loss_count": cnt(lv),
                    "teacher_operand1_loss": mean(rce, rv).detach(), "teacher_operand1_loss_count": cnt(rv),
                    "teacher_output_loss": mean(nll, ov).detach(), "teacher_output_loss_count": cnt(ov),
                    "teacher_destination0_binding_loss": mean(dce, dv).detach(), "teacher_destination0_binding_loss_count": cnt(dv),
                    "teacher_feature_off_active_operation_accuracy": acc(pred.eq(op_t), pos),
                    "teacher_feature_off_active_operation_accuracy_count": cnt(pos),
                    "teacher_feature_off_operand0_accuracy": acc(left_logits.argmax(-1).eq(lt), lv),
                    "teacher_feature_off_operand0_accuracy_count": cnt(lv),
                    "teacher_feature_off_operand1_accuracy": acc(right_logits.argmax(-1).eq(rt), rv),
                    "teacher_feature_off_operand1_accuracy_count": cnt(rv),
                    "teacher_feature_off_output_accuracy": acc(d_lab.argmax(-1).eq(out_t), ov),
                    "teacher_feature_off_output_accuracy_count": cnt(ov),
                    "teacher_feature_off_destination0_binding_accuracy": acc(ent_logits.argmax(-1).eq(dest_t), dv),
                    "teacher_feature_off_destination0_binding_accuracy_count": cnt(dv),
                }
                for c, name in enumerate(_NAMES):
                    mk = valid & op_t.eq(c)
                    diag[f"teacher_op_loss_{name}"] = mean(ce, mk).detach()
                    diag[f"teacher_op_loss_{name}_count"] = cnt(mk)
                self._accumulate_diagnostics(diag)

        if losses:
            loss = torch.stack(losses).mean() * float(teacher_weight)
            self._last_aux_loss = loss if self._last_aux_loss is None else self._last_aux_loss + loss
        if self.training:
            _s = (route_ops[0].detach() * active_mask.float().unsqueeze(-1)).sum(0)
            self._route_sum = _s if self._route_sum is None else self._route_sum + _s
            self._route_n = self._route_n + active_mask.sum()
            self._last_training_diagnostics["operation_probability"] = self._route_sum / torch.as_tensor(self._route_n, device=self._route_sum.device).clamp_min(1.0)
        else:
            r = route_ops[0].argmax(-1)
            for ro in route_ops[1:]:
                r = torch.where(r == 0, ro.argmax(-1), r)
            self._last_route_steps.append((int(step_index), r.detach()))
        return torch.cat((run_addr, run_content), -1)