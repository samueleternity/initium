"""Workspace-conditioned parameter modulation for parallel Mamba-1 blocks (Phase B)."""

from __future__ import annotations

import torch
import torch.nn as nn

_TARGETS = ("dt", "a", "B", "C", "D")


class ModulationHead(nn.Module):
    """Per-token bounded modulation of Mamba-1 parameters.

    Returns, per modulated block, a dict of already-scaled values:
    dt/a -> log-multipliers (apply exp), B/C/D -> additive offsets (apply 1+v).
    All outputs are zero at init and zero on padded tokens.
    """

    def __init__(
        self,
        d_model,
        d_inner,
        d_state,
        mod_blocks,
        targets,
        source="workspace",
        rank=32,
        alphas=None,
        lam=0.0,
        temporal_lam=0.0,
        route_dim=0,
        meta_responsiveness=False,
    ):
        super().__init__()
        if source not in ("workspace", "backbone", "static"):
            raise ValueError("modulation source must be workspace, backbone or static")
        if not targets or set(targets) - set(_TARGETS):
            raise ValueError(f"modulation targets must be a non-empty subset of {_TARGETS}")
        a = {"dt": 0.5, "a": 0.5, "B": 0.3, "C": 0.3, "D": 0.3}
        a.update(alphas or {})
        if a["a"] > 0.5:
            raise ValueError("alpha_A must be <= 0.5 (keeps A-scale s in ~[0.6, 1.65])")
        self.mod_blocks = tuple(int(b) for b in mod_blocks)
        self.targets = tuple(targets)
        self.source = source
        self.route_dim = int(route_dim) if source == "workspace" else 0
        self.alphas = a
        self.lam, self.temporal_lam = float(lam), float(temporal_lam)
        self.meta_responsiveness = bool(meta_responsiveness)
        self.alpha_scale = 1.0  # warmup multiplier, set by the trainer
        self.dims = {"dt": d_inner, "a": 1, "B": d_state, "C": d_state, "D": d_inner}
        keys = [f"{b}_{t}" for b in self.mod_blocks for t in self.targets]
        if source == "static":
            self.static = nn.ParameterDict(
                {k: nn.Parameter(torch.zeros(self.dims[k.split("_")[1]])) for k in keys}
            )
        else:
            self.norm_h = nn.LayerNorm(d_model)
            self.norm_r = nn.LayerNorm(d_model) if source == "workspace" else None
            self.trunk = nn.Linear(
                d_model * (2 if source == "workspace" else 1) + self.route_dim, rank
            )
            self.out = nn.ModuleDict(
                {k: nn.Linear(rank, self.dims[k.split("_")[1]], bias=False) for k in keys}
            )
            for m in self.out.values():
                nn.init.zeros_(m.weight)
        if self.meta_responsiveness:
            self.responsiveness = nn.Linear(d_model, 1, bias=False)
            nn.init.zeros_(self.responsiveness.weight)
        else:
            self.responsiveness = None
        self._aux = None
        self.last_diagnostics: dict[str, torch.Tensor] = {}

    def forward(self, h, r, valid_mask=None, c=None):
        with torch.autocast(device_type=h.device.type, enabled=False):
            z = None
            if self.source != "static":
                z = self.norm_h(h.float())
                if self.source == "workspace":
                    if r is None:
                        raise ValueError("workspace-sourced modulation needs the workspace readout")
                    z = torch.cat((self.norm_r(r.float()), z), dim=-1)
                    if self.route_dim:
                        if c is None:
                            raise ValueError("route-code modulation needs aux['route_code']")
                        z = torch.cat((z, c.float()), dim=-1)
                z = torch.tanh(self.trunk(z))
            rho = None
            if self.responsiveness is not None:
                context = r.float() if self.source == "workspace" else h.float()
                rho = torch.exp(0.5 * torch.tanh(self.responsiveness(context)))
            m2 = (
                torch.ones(h.shape[:2], device=h.device)
                if valid_mask is None
                else valid_mask.to(device=h.device, dtype=torch.float32)
            )
            m3 = m2.unsqueeze(-1)
            denom = m2.sum().clamp_min(1.0)
            reg = h.new_zeros((), dtype=torch.float32)
            diag, result = {}, {}
            for b in self.mod_blocks:
                d = {}
                for t in self.targets:
                    k = f"{b}_{t}"
                    if z is None:
                        u = (
                            torch.tanh(self.static[k])
                            .view(1, 1, -1)
                            .expand(h.shape[0], h.shape[1], -1)
                        )
                    else:
                        u = torch.tanh(self.out[k](z))
                    u = u * m3
                    if self.training:
                        if self.lam > 0:
                            reg = reg + self.lam * ((u * u).mean(-1) * m2).sum() / denom
                        if self.temporal_lam > 0 and u.shape[1] > 1:
                            dm = m2[:, 1:] * m2[:, :-1]
                            reg = reg + self.temporal_lam * (
                                ((u[:, 1:] - u[:, :-1]) ** 2).mean(-1) * dm
                            ).sum() / dm.sum().clamp_min(1.0)
                        with torch.no_grad():
                            ud = u.detach()
                            diag[f"b{b}_{t}_rms"] = (
                                ud.square().sum() / (denom * ud.shape[-1])
                            ).sqrt()
                            diag[f"b{b}_{t}_sat"] = (
                                (ud.abs() > 0.95).float().mean(-1) * m2
                            ).sum() / denom
                    v = u * (self.alphas[t] * self.alpha_scale)
                    if rho is not None:
                        v = v * rho
                    d[t] = v.squeeze(-1) if t == "a" else v
                result[b] = d
            if rho is not None and self.training:
                with torch.no_grad():
                    diag["responsiveness_rms"] = rho.detach().square().mean().sqrt()
                    diag["responsiveness_saturation"] = (
                        (rho.detach().log().abs() > 0.45).float().mean()
                    )
            if self.training:
                self._aux = reg if (self.lam > 0 or self.temporal_lam > 0) else None
                self.last_diagnostics = diag
            return result

    def pop_aux_loss(self):
        loss, self._aux = self._aux, None
        return loss
