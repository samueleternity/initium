"""
file: mamba3_controller.py -- v1
Interleaved Mamba-3 (SISO) DNC controller. Same Cell/Block/Wrapper split as
mamba2_controller.py. Math: paper Prop.1 (exp-trapezoidal) + Prop.4/Eq.11 (RoPE trick):
  h_t = a_t h_{t-1} + b_t (k_{t-1} x_{t-1}^T) + g_t (k_t x_t^T),  y_t = C_t^T h_t + D x_t
  a=exp(DT*A), g=lam*DT, b=(1-lam)*DT*a, k/C = RMSNorm+bias then rotated by cumsum(angle*DT).
State per block: (angle_state, ssm_state, k_prev, v_prev), ALL fp32 (angle is a running sum).
No conv_state: Mamba-3 has no short conv. Everything after in_proj runs in real fp32.
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from MoE.moe_layer import MoEBlock
from mamba_controller.mamba2_controller import Mamba2ControllerWrapper

try:
    from mamba_ssm.modules.mamba3 import Mamba3
except ImportError as _e:  # pragma: no cover
    Mamba3 = None
    _MAMBA3_IMPORT_ERROR = ("mamba3_controller.py needs mamba-ssm installed from GitHub main "
                            f"(Mamba3 is not in the v2.3.1 wheel). Original error: {_e}")
else:
    _MAMBA3_IMPORT_ERROR = None


def _require_mamba3_ssm() -> None:
    if Mamba3 is None:
        raise ImportError(_MAMBA3_IMPORT_ERROR)


def _heavy_tail(x):  # copied from mamba_ssm.modules.mamba3.heavy_tail_activation
    return x.clamp_min(0) + torch.reciprocal(1 - x.clamp_max(0))


def _rms(v, w, eps):
    return v * torch.rsqrt(v.pow(2).mean(-1, keepdim=True) + eps) * w.float()


class Mamba3ControllerCell(nn.Module):
    def __init__(self, d_model, d_state=64, expand=2, headdim=64, ngroups=1,
                 rope_fraction=0.5, layer_idx=None, device=None, dtype=None):
        super().__init__()
        _require_mamba3_ssm()
        assert ngroups == 1, "Mamba3ControllerCell supports ngroups==1 only"
        # parameter container only; we never call .forward()/.step() on it
        self.mamba3 = Mamba3(d_model=d_model, d_state=d_state, expand=expand, headdim=headdim,
                             ngroups=ngroups, rope_fraction=rope_fraction, is_mimo=False,
                             is_outproj_norm=False, layer_idx=layer_idx, device=device, dtype=dtype)
        m = self.mamba3
        self.d_model, self.d_inner, self.nheads, self.headdim = d_model, m.d_inner, m.nheads, m.headdim
        self.d_state, self.num_rope_angles = m.d_state, m.num_rope_angles

    def init_state(self, batch_size, device=None, dtype=None):  # dtype ignored: always fp32
        device = device if device is not None else self.mamba3.in_proj.weight.device
        H, P, N, S = self.nheads, self.headdim, self.d_state, self.num_rope_angles
        f = dict(device=device, dtype=torch.float32)
        return (torch.zeros(batch_size, H, S, **f), torch.zeros(batch_size, H, P, N, **f),
                torch.zeros(batch_size, H, N, **f), torch.zeros(batch_size, H, P, **f))

    @staticmethod
    def _rotate(v, cos, sin):  # v: (b,H,N); rotate first 2S dims, adjacent pairs (SISO convention)
        S = cos.shape[-1]
        head, tail = v[..., :2 * S], v[..., 2 * S:]
        h0, h1 = head[..., 0::2], head[..., 1::2]
        rot = torch.stack([h0 * cos - h1 * sin, h0 * sin + h1 * cos], dim=-1).flatten(-2)
        return torch.cat([rot, tail], dim=-1)

    def step(self, u, state):
        angle_state, ssm_state, k_prev, v_prev = state
        m = self.mamba3
        H, P, N, S = self.nheads, self.headdim, self.d_state, self.num_rope_angles
        dtype = u.dtype
        with torch.autocast(device_type=u.device.type, enabled=False):
            zxb = F.linear(u.float(), m.in_proj.weight.float())
            z, x, B, C, dd_dt, dd_A, trap_proj, ang_proj = torch.split(
                zxb, [self.d_inner, self.d_inner, N, N, H, H, H, S], dim=-1)
            A = (-_heavy_tail(dd_A)).clamp(max=-m.A_floor)                       # (b,H)
            DT = F.softplus(dd_dt + m.dt_bias.float()).clamp(min=1e-6, max=100.0)
            lam = torch.sigmoid(trap_proj)
            alpha = torch.exp(DT * A)
            gamma = lam * DT
            beta = (1.0 - lam) * DT * alpha
            B = _rms(B, m.B_norm.weight, m.B_norm.eps).unsqueeze(1) + m.B_bias.float().squeeze(1)  # (b,H,N)
            C = _rms(C, m.C_norm.weight, m.C_norm.eps).unsqueeze(1) + m.C_bias.float().squeeze(1)
            angle_new = angle_state + ang_proj.unsqueeze(1) * DT.unsqueeze(-1)   # (b,H,S) cumsum(angle*DT)
            cos, sin = angle_new.cos(), angle_new.sin()
            B, C = self._rotate(B, cos, sin), self._rotate(C, cos, sin)
            x = x.reshape(-1, H, P)
            z = z.reshape(-1, H, P)
            new_ssm = (ssm_state * alpha[..., None, None]
                       + beta[..., None, None] * (v_prev.unsqueeze(-1) * k_prev.unsqueeze(-2))
                       + gamma[..., None, None] * (x.unsqueeze(-1) * B.unsqueeze(-2)))
            new_ssm = new_ssm.clamp(min=-1e4, max=1e4)                            # same hard stop as Mamba-1/2 cells
            y = torch.einsum("bhpn,bhn->bhp", new_ssm, C) + m.D.float().view(1, H, 1) * x
            y = (y * F.silu(z)).reshape(-1, self.d_inner).clamp(min=-1e4, max=1e4)
        out = m.out_proj(y.to(dtype))                                             # back under ambient autocast
        return out, (angle_new, new_ssm, B, x)


class Mamba3ControllerBlock(nn.Module):
    def __init__(self, d_model, d_state=64, expand=2, headdim=64, ngroups=1, rope_fraction=0.5,
                 layer_idx=None, device=None, dtype=None):
        super().__init__()
        self.norm = nn.LayerNorm(d_model, device=device, dtype=dtype)
        self.cell = Mamba3ControllerCell(d_model, d_state=d_state, expand=expand, headdim=headdim,
                                         ngroups=ngroups, rope_fraction=rope_fraction,
                                         layer_idx=layer_idx, device=device, dtype=dtype)

    def init_state(self, batch_size, device=None, dtype=None):
        return self.cell.init_state(batch_size, device=device, dtype=dtype)

    def step(self, x, state):
        out, new_state = self.cell.step(self.norm(x), state)
        return x + out, new_state


class Mamba3ControllerWrapper(Mamba2ControllerWrapper):
    """Reuses Mamba2ControllerWrapper.forward()/init_state() unchanged (they only touch
    in_adapter / blocks / moe_blocks / moe_enabled); only __init__ differs."""

    def __init__(self, in_dim, d_model, num_blocks=2, moe_enabled=False, moe_num_experts=8,
                 moe_expert_dim=None, moe_capacity_factor=1.5, moe_load_balance_alpha=0.01,
                 moe_top_k=1, d_state=64, expand=2, headdim=64, ngroups=1, rope_fraction=0.5,
                 device=None, dtype=None):
        nn.Module.__init__(self)
        self.d_model, self.num_blocks = d_model, num_blocks
        self.in_adapter = (nn.Identity() if in_dim == d_model
                           else nn.Linear(in_dim, d_model, device=device, dtype=dtype))
        self.blocks = nn.ModuleList([
            Mamba3ControllerBlock(d_model, d_state=d_state, expand=expand, headdim=headdim,
                                  ngroups=ngroups, rope_fraction=rope_fraction, layer_idx=i,
                                  device=device, dtype=dtype) for i in range(num_blocks)])
        self.moe_enabled = moe_enabled
        self.moe_blocks = None
        if moe_enabled:
            self.moe_blocks = nn.ModuleList([
                MoEBlock(d_model, num_experts=moe_num_experts, expert_dim=moe_expert_dim,
                         capacity_factor=moe_capacity_factor, top_k=moe_top_k,
                         load_balance_alpha=moe_load_balance_alpha, device=device, dtype=dtype)
                for _ in range(num_blocks)])