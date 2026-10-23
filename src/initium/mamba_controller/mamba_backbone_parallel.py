"""
file: mamba_backbone_parallel.py

Alternate Phase 3, Step 2, Option 5 ("split the compute graph, don't try to
parallelize the addressing itself") -- see Experiment-Roadmap.md.

This module implements ONLY the "(a) Mamba backbone" half of the Option 5
split: a stack of Mamba (v1) or Mamba-2 blocks driven over the WHOLE
sequence at once via each block's own native, library-provided forward()
(the fast, parallel-scan / SSD path), with NO per-timestep interleaving of
memory reads.

This is a deliberate, much simpler sibling to mamba_controller.py's
MambaControllerCell/Block/Wrapper (Alternative Phase 3, Step 1). That file
had to hand-roll a BPTT-safe manual step() specifically because the DNC's
per-timestep interleaving loop needed to call the controller one token at a
time while chaining (conv_state, ssm_state) through a Python loop under
full backprop, and mamba_ssm's own .step() mutates its state tensors
in-place (fatal for BPTT -- see that file's module docstring for the full
diagnosis).

Option 5's entire point is to remove that constraint for the backbone: since
this backbone's per-token computation has NO dependency on M_{t-1} (the
memory's state at the previous timestep) -- it only ever consumes the raw
input sequence X, never a read vector -- there is nothing forcing it into a
sequential Python loop. It can therefore call mamba_ssm's OWN whole-sequence
forward() directly, which already uses the fast parallel selective-scan
(Mamba-1) or the chunked, matmul-based SSD algorithm (Mamba-2, Dao & Gu
2024) -- no custom step function, no in-place-op workaround needed, because
nothing here is chained through a live memory-state Python loop.

The other half of the Option 5 split -- the "genuinely sequential, but much
smaller and cheaper, memory read/write/addressing step" that consumes this
backbone's parallel output as an input stream -- lives in
split_graph_dnc.py, not here. This file only produces H = backbone(X); it
has no notion of DNC, Memory, or read vectors at all, by design (see
Concept 6 / LB-9's caution about structural-vs-functional separation --
keeping this file's API surface structurally incapable of touching memory
state is a guarantee, not just a convention).

Supports two variants, both from the mamba-ssm package (per the roadmap's
"add Mamba-2 first if Option 5 needs it, otherwise Mamba-1 is already
implemented" instruction -- Option 5 benefits directly from Mamba-2's own
SSD chunked-scan algorithm, since that algorithm is specifically optimized
for exactly the "process the whole sequence at once" call pattern Option 5
newly unlocks for the backbone -- see the SSD paper, Sections 6 and 9.3,
already in corpus):

  - "mamba1": mamba_ssm.modules.mamba_simple.Mamba
  - "mamba2": mamba_ssm.modules.mamba2.Mamba2

Both are used exactly as their own package intends (whole-sequence
forward(hidden_states) call, batch_first (B, L, D) tensors) -- this file
does not reimplement or bypass anything of either class, unlike
mamba_controller.py's cell.

NOTE on Mamba2 kwargs: mamba_ssm's Mamba2 additionally requires `headdim`
(d_model must be divisible by it -- SSD paper's own convention is
head_dim in {64, 128}). This has no analogue in Mamba-1 and is therefore
a new, Mamba2-only constructor argument here, same pattern as
mamba_d_state/mamba_d_conv/mamba_expand having no LSTM analogue in
mamba_controller.py.
"""

from __future__ import annotations

import os as _os

import torch
import torch.nn as nn

from initium.MoE.moe_layer import MoEBlock

try:
    from einops import rearrange
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
except ImportError:  # pragma: no cover - optional CUDA extension
    rearrange = None
    selective_scan_fn = None

_MAMBA1_IMPORT_ERROR: str | None
_MAMBA2_IMPORT_ERROR: str | None
_MAMBA3_IMPORT_ERROR: str | None

try:
    from mamba_ssm.modules.mamba_simple import Mamba as Mamba1
except ImportError as _mamba1_err:  # pragma: no cover - environment-dependent
    Mamba1 = None
    _MAMBA1_IMPORT_ERROR = (
        "mamba_backbone_parallel.py requires the `mamba-ssm` package for "
        f"variant='mamba1'. Original import error: {_mamba1_err}"
    )
else:
    _MAMBA1_IMPORT_ERROR = None

try:
    from mamba_ssm.modules.mamba2 import Mamba2
except ImportError as _mamba2_err:  # pragma: no cover - environment-dependent
    Mamba2 = None
    _MAMBA2_IMPORT_ERROR = (
        "mamba_backbone_parallel.py requires the `mamba-ssm` package "
        "(with Mamba-2 support -- mamba_ssm.modules.mamba2.Mamba2) for "
        f"variant='mamba2'. Original import error: {_mamba2_err}"
    )
else:
    _MAMBA2_IMPORT_ERROR = None

try:
    from mamba_ssm.modules.mamba3 import Mamba3
except ImportError as _mamba3_err:  # pragma: no cover - environment-dependent
    Mamba3 = None
    _MAMBA3_IMPORT_ERROR = (
        "mamba_backbone_parallel.py: variant='mamba3' needs mamba-ssm installed from GitHub main "
        f"(Mamba3 is not in the v2.3.1 wheel). Original import error: {_mamba3_err}"
    )
else:
    _MAMBA3_IMPORT_ERROR = None

# bf16 (the repo's own precision) when the GPU supports it; otherwise the fp32-under-fp16-autocast trick.
_MAMBA3_BF16_OK = (
    torch.cuda.is_available()
    and torch.cuda.get_device_capability()[0] >= 8
    and _os.environ.get("MAMBA3_FORCE_FP32", "0") != "1"
)


def _require_variant(variant: str) -> None:
    if variant == "mamba1" and Mamba1 is None:
        raise ImportError(_MAMBA1_IMPORT_ERROR)
    if variant == "mamba2" and Mamba2 is None:
        raise ImportError(_MAMBA2_IMPORT_ERROR)
    if variant == "mamba3" and Mamba3 is None:
        raise ImportError(_MAMBA3_IMPORT_ERROR)
    if variant not in ("mamba1", "mamba2", "mamba3"):
        raise ValueError(
            f"mamba_backbone_parallel: unknown variant {variant!r}, expected 'mamba1' or 'mamba2'."
        )


class _ParallelBlock(nn.Module):
    """Pre-norm residual wrapper around one whole-sequence Mamba block,
    mirroring mamba_ssm's own Block (Add -> LN -> Mixer) and
    mamba_controller.py's MambaControllerBlock, but calling the mixer's own
    forward() over the full (B, L, D) sequence at once instead of a
    per-timestep step(). No custom state threading of any kind -- the
    mixer manages its own internal scan/chunking entirely within this one
    call.
    """

    def __init__(
        self,
        d_model: int,
        variant: str,
        mamba_kwargs: dict,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
        sgr_delta_error_gating: bool = False,
    ):
        super().__init__()
        _require_variant(variant)
        self.norm = nn.LayerNorm(d_model, device=device, dtype=dtype)
        self.variant = variant
        self.sgr_delta_error_gating = bool(sgr_delta_error_gating)
        self.error_to_delta = None
        if self.sgr_delta_error_gating:
            if variant != "mamba1":
                raise ValueError("SGR delta-error gating currently supports only Mamba-1")
            inner_size = int(d_model * mamba_kwargs.get("expand", 2))
            self.error_to_delta = nn.Linear(1, inner_size, bias=False, device=device, dtype=dtype)
            nn.init.zeros_(self.error_to_delta.weight)
        if variant == "mamba3":
            self.mixer = Mamba3(d_model=d_model, device=device, dtype=dtype, **mamba_kwargs)
        elif variant == "mamba1":
            self.mixer = Mamba1(d_model=d_model, device=device, dtype=dtype, **mamba_kwargs)
        else:
            self.mixer = Mamba2(d_model=d_model, device=device, dtype=dtype, **mamba_kwargs)

    def forward(
        self, x: torch.Tensor, *, prediction_error: torch.Tensor | None = None
    ) -> torch.Tensor:
        # x: (B, L, D). mamba_ssm's own forward() handles the whole
        # sequence's parallel scan / SSD chunked-scan internally -- this is
        # the entire mechanism that makes Option 5 different from
        # mamba_controller.py: one call per block, not one call per DNC
        # timestep.
        #
        # FIX (chronic grad_norm nan under fp16 AMP, Mamba-2 variant only in
        # practice but applied to both for symmetry): same wrapper trick
        # already used by mamba_controller.py's MambaControllerCell.step()
        # and mamba2_controller.py's Mamba2ControllerCell.step() -- force
        # this call to run in real fp32, outside ambient fp16 autocast,
        # rather than reimplementing the mixer's internal math by hand the
        # way those cells do for their own per-timestep step(). Unlike those
        # cells, self.mixer here is the library's own whole-sequence
        # forward() (that's the entire point of the parallel backbone), so
        # we can't insert clamps mid-computation the way step() does --
        # instead we just deny it fp16 entirely. Mamba-2's SSD algorithm
        # computes exp() of a cumulative sum of (dt * A) over a whole chunk
        # (paper Section 6, Listing 1's segsum), which is far more prone to
        # fp16 underflow/overflow than Mamba-1's fused selective-scan
        # kernel -- this is the direct cause of the amp_scale-collapse-to-
        # 0.0 pattern observed in mamba2-backbone split-graph runs that
        # mamba1-backbone runs didn't show. Costs some speed relative to
        # running under fp16 (this block no longer benefits from tensor-core
        # fp16 matmuls), but matches every other Mamba SSM computation path
        # in this project, all of which already force fp32 for this reason.
        if prediction_error is not None:
            return self.forward_with_prediction_error(x, prediction_error)
        numeric_observer = getattr(self, "_initium_forward_numeric_observer", None)
        if self.variant == "mamba3" and _MAMBA3_BF16_OK:
            with torch.autocast(device_type=x.device.type, dtype=torch.bfloat16, enabled=True):
                normed = self.norm(x.float()).to(torch.bfloat16)
                if numeric_observer is not None:
                    numeric_observer("normalized_input", normed)
                out = self.mixer(normed)
        else:
            with torch.autocast(device_type=x.device.type, enabled=False):
                normed = self.norm(x.float())
                if numeric_observer is not None:
                    numeric_observer("normalized_input", normed)
                out = self.mixer(normed)
        if numeric_observer is not None:
            numeric_observer("mixer_output", out)
        residual = out.to(x.dtype)
        if numeric_observer is not None:
            numeric_observer("residual_branch_cast", residual)
        result = x + residual
        if numeric_observer is not None:
            numeric_observer("residual_output", result)
        return result

    def forward_with_prediction_error(
        self, x: torch.Tensor, prediction_error: torch.Tensor
    ) -> torch.Tensor:
        """Run Mamba-1's selective scan with an inter-layer error term on Δ.

        The injected scalar is transformed into a per-channel additive bias
        on the pre-softplus Δ projection. The zero-initialized projection
        makes the initial forward function equal to the ordinary block. This
        intentionally uses the unfused selective-scan path because the fused
        Mamba-1 kernel does not expose a per-position Δ override.
        """
        if self.variant != "mamba1" or self.error_to_delta is None:
            raise RuntimeError("prediction-error injection requires a configured Mamba-1 block")
        if rearrange is None or selective_scan_fn is None:
            raise RuntimeError(
                "SGR delta-error gating requires einops and mamba_ssm's selective_scan_fn"
            )
        if prediction_error.ndim != 2 or prediction_error.shape != x.shape[:2]:
            raise ValueError("prediction_error must have shape (batch, sequence)")

        numeric_observer = getattr(self, "_initium_forward_numeric_observer", None)
        with torch.autocast(device_type=x.device.type, enabled=False):
            normed = self.norm(x.float())
            mixer = self.mixer
            batch, sequence, _ = normed.shape
            xz = mixer.in_proj(normed).transpose(1, 2)
            x_branch, z = xz.chunk(2, dim=1)
            x_branch = mixer.act(mixer.conv1d(x_branch)[..., :sequence])
            projected = mixer.x_proj(rearrange(x_branch, "b d l -> (b l) d"))
            dt_low_rank, B, C = torch.split(
                projected, [mixer.dt_rank, mixer.d_state, mixer.d_state], dim=-1
            )
            dt = mixer.dt_proj.weight @ dt_low_rank.transpose(0, 1)
            dt = rearrange(dt, "d (b l) -> b d l", b=batch, l=sequence)
            error_bias = self.error_to_delta(prediction_error.float().unsqueeze(-1))
            dt = dt + error_bias.transpose(1, 2).to(dtype=dt.dtype)
            B = rearrange(B, "(b l) n -> b n l", b=batch, l=sequence).contiguous()
            C = rearrange(C, "(b l) n -> b n l", b=batch, l=sequence).contiguous()
            A = -torch.exp(mixer.A_log.float())
            if numeric_observer is not None:
                numeric_observer("delta_pre_softplus", dt)
                numeric_observer("prediction_error", prediction_error)
            y = selective_scan_fn(
                x_branch,
                dt,
                A,
                B,
                C,
                mixer.D.float(),
                z=z,
                delta_bias=mixer.dt_proj.bias.float(),
                delta_softplus=True,
            )
            y = rearrange(y, "b d l -> b l d")
            mixer_output = mixer.out_proj(y)
        residual = mixer_output.to(dtype=x.dtype)
        return x + residual


class MambaBackboneParallel(nn.Module):
    """Stack of `num_blocks` whole-sequence Mamba blocks. Pure sequence-to-
    sequence map (B, L, in_dim) -> (B, L, d_model), with NO awareness of
    DNC, Memory, or read vectors -- see module docstring.

    variant: "mamba1" (mamba_ssm.modules.mamba_simple.Mamba, default
        Mamba-1 hyperparameters d_state=16/d_conv=4/expand=2, this
        project's original Alternate-Phase-3 choice) or "mamba2"
        (mamba_ssm.modules.mamba2.Mamba2, needs `headdim` in addition;
        default 64, matching the SSD paper's own convention of head_dim in
        {64, 128}).
    """

    def __init__(
        self,
        in_dim: int,
        d_model: int,
        num_blocks: int = 2,
        variant: str = "mamba1",
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        headdim: int = 64,  # mamba2-only
        moe_enabled: bool = False,
        moe_num_experts: int = 8,
        moe_expert_dim: int | None = None,
        moe_top_k: int = 1,
        moe_capacity_factor: float = 1.5,
        moe_load_balance_alpha: float = 0.01,
        moe_residual_scale: float = 1.0,
        moe_block_isolation: int | None = None,
        sgr_delta_error_gating: bool = False,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        _require_variant(variant)
        self.variant = variant
        self.d_model = d_model
        self.moe_enabled = moe_enabled
        self.moe_blocks: nn.ModuleList | None = None
        if moe_block_isolation is not None and not 0 <= moe_block_isolation < num_blocks:
            raise ValueError(
                f"moe_block_isolation={moe_block_isolation} is outside "
                f"the configured backbone block range [0, {num_blocks - 1}]"
            )

        # Same role as MambaControllerWrapper's in_adapter: DNC's raw
        # per-timestep input (input_size) generally differs from
        # hidden_size/d_model. Unlike the interleaved wrapper, this is
        # applied to the WHOLE sequence at once (a single
        # (B, L, in_dim) -> (B, L, d_model) matmul), not per-timestep.
        self.in_adapter: nn.Module = (
            nn.Identity()
            if in_dim == d_model
            else nn.Linear(in_dim, d_model, device=device, dtype=dtype)
        )

        mamba_kwargs = dict(d_state=d_state, d_conv=d_conv, expand=expand)
        if variant == "mamba2":
            mamba_kwargs["headdim"] = headdim
        if variant == "mamba3":
            mamba_kwargs["headdim"] = headdim
            mamba_kwargs["chunk_size"] = (
                64 if _MAMBA3_BF16_OK else 32
            )  # repo: 64 bf16 / 32 otherwise

        self.blocks = nn.ModuleList(
            [
                _ParallelBlock(
                    d_model,
                    variant,
                    mamba_kwargs,
                    device=device,
                    dtype=dtype,
                    sgr_delta_error_gating=sgr_delta_error_gating and i == 1,
                )
                for i in range(num_blocks)
            ]
        )
        # Match the standard pre-norm SSM stack contract: normalize the
        # residual stream after the final mixer before exposing features to
        # downstream modules. This is deliberately non-affine, so activation
        # scale cannot be relearned to an unsafe range and existing checkpoint
        # state dicts remain load-compatible (no parameters or buffers).
        self.output_norm = nn.LayerNorm(d_model, elementwise_affine=False)
        if moe_enabled:
            self.moe_blocks = nn.ModuleList(
                [
                    (
                        MoEBlock(
                            d_model,
                            num_experts=moe_num_experts,
                            expert_dim=moe_expert_dim,
                            top_k=moe_top_k,
                            capacity_factor=moe_capacity_factor,
                            load_balance_alpha=moe_load_balance_alpha,
                            residual_scale=moe_residual_scale,
                            device=device,
                            dtype=dtype,
                        )
                        if moe_block_isolation is None or i == moe_block_isolation
                        else nn.Identity()
                    )
                    for i in range(num_blocks)
                ]
            )

    def forward(self, x: torch.Tensor, *, token_mask: torch.Tensor | None = None) -> torch.Tensor:
        # x: (B, L, in_dim) -> (B, L, d_model). Called EXACTLY ONCE per
        # training step (not once per DNC timestep, unlike
        # MambaControllerWrapper) -- this single call is the entire
        # "backbone half" of the Option 5 split.
        h = self.in_adapter(x)
        numeric_observer = getattr(self, "_initium_forward_numeric_observer", None)
        if numeric_observer is not None:
            numeric_observer("input_adapter.output", h)
        moe_blocks = self.moe_blocks
        for i, block in enumerate(self.blocks):
            h = block(h)
            if numeric_observer is not None:
                numeric_observer(f"block.{i}.output", h)
            if moe_blocks is not None and isinstance(moe_blocks[i], MoEBlock):
                h = moe_blocks[i](h, token_mask=token_mask)
                if numeric_observer is not None:
                    numeric_observer(f"block.{i}.moe_output", h)
        return self.normalize_output(h)

    def forward_with_intermediate(
        self,
        x: torch.Tensor,
        *,
        token_mask: torch.Tensor | None = None,
        intermediate_observer=None,
    ) -> torch.Tensor:
        """Run the normal stack and expose block-0 features without a second pass."""
        h = self.in_adapter(x)
        numeric_observer = getattr(self, "_initium_forward_numeric_observer", None)
        if numeric_observer is not None:
            numeric_observer("input_adapter.output", h)
        for index, block in enumerate(self.blocks):
            h = block(h)
            if numeric_observer is not None:
                numeric_observer(f"block.{index}.output", h)
            if index == 0 and intermediate_observer is not None:
                intermediate_observer(h)
            if self.moe_blocks is not None and isinstance(self.moe_blocks[index], MoEBlock):
                h = self.moe_blocks[index](h, token_mask=token_mask)
                if numeric_observer is not None:
                    numeric_observer(f"block.{index}.moe_output", h)
        return self.normalize_output(h)

    def forward_sgr(
        self,
        x: torch.Tensor,
        *,
        arm: str,
        decoder: nn.Module | None = None,
        signal_mode: str = "excess-surprise",
        stream_lengths: list[int] | None = None,
        oracle_flags: torch.Tensor | None = None,
        gate_override: str = "none",
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, dict[str, torch.Tensor]]:
        """Forward for the isolated SGR Phase 2 experiment arms."""
        valid_arms = {"content", "aux-only", "error", "blend", "shuffled", "oracle"}
        if arm not in valid_arms:
            raise ValueError(f"unknown SGR Phase 2 arm {arm!r}")
        if self.variant != "mamba1" or len(self.blocks) != 2:
            raise ValueError("SGR Phase 2 requires exactly two Mamba-1 blocks")
        if gate_override not in {"none", "zero", "shuffle"}:
            raise ValueError("gate_override must be none, zero, or shuffle")
        if arm in {"aux-only", "error", "blend", "shuffled"} and decoder is None:
            raise ValueError(f"Phase 2 arm {arm!r} requires the prediction decoder")

        # Phase 2's Mamba mixers run in fp32, and its second block can produce
        # large residuals while the error-to-delta gate is being learned.
        # Preserve the residual stream in fp32 as well: casting a large mixer
        # residual back to the autocast input dtype (fp16) can overflow near
        # 65,504 even when the fp32 computation itself remains finite.
        h = self.in_adapter(x).float()
        first_features = self.blocks[0](h)
        batch, sequence, _ = first_features.shape
        token_ids = x.argmax(dim=-1)
        if stream_lengths is None:
            positions = torch.arange(sequence, device=x.device).unsqueeze(0)
            lengths = torch.where(token_ids.eq(20), positions, sequence).amin(dim=1)
        else:
            if len(stream_lengths) != batch:
                raise ValueError("stream_lengths must contain one value per batch row")
            if any(not 0 <= int(length) <= sequence for length in stream_lengths):
                raise ValueError("stream_lengths must be valid lengths for each batch row")
            lengths = torch.as_tensor(stream_lengths, device=x.device)
        stream_mask = torch.arange(sequence, device=x.device).unsqueeze(0) < lengths.unsqueeze(1)

        def shuffle_valid_stream(values: torch.Tensor) -> torch.Tensor:
            positions = torch.arange(sequence, device=x.device).unsqueeze(0)
            reversed_positions = (lengths.unsqueeze(1) - 1 - positions).clamp_min(0)
            indices = torch.where(positions < lengths.unsqueeze(1), reversed_positions, positions)
            return values.gather(1, indices.expand_as(values))

        logits = None
        error_signal = first_features.new_zeros((batch, sequence), dtype=torch.float32)
        if decoder is not None:
            logits = decoder(first_features[:, :-1]).float()
            log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
            target = token_ids[:, 1:]
            raw = -log_probs.gather(-1, target.unsqueeze(-1)).squeeze(-1)
            if signal_mode == "surprisal":
                score = raw
            elif signal_mode == "excess-surprise":
                probs = log_probs.exp()
                entropy = -(probs * log_probs).sum(dim=-1)
                score = raw - entropy
            else:
                raise ValueError("Phase 2 signal must be surprisal or excess-surprise")
            error_signal[:, 1:] = torch.tanh(score.detach())
            error_signal = error_signal * stream_mask

        alert_mask = x[..., 16:20].sum(dim=-1).gt(0).to(dtype=error_signal.dtype)
        if arm == "content":
            gate_signal = alert_mask
        elif arm == "oracle":
            if oracle_flags is None or oracle_flags.shape != (batch, sequence):
                raise ValueError("oracle arm requires oracle_flags with shape (batch, sequence)")
            gate_signal = oracle_flags.to(device=x.device, dtype=error_signal.dtype)
        elif arm == "error":
            gate_signal = error_signal
        elif arm == "blend":
            gate_signal = 0.5 * (error_signal + alert_mask)
        elif arm == "shuffled":
            gate_signal = shuffle_valid_stream(error_signal)
        else:
            gate_signal = None  # auxiliary-only preserves the ungated backbone

        if gate_override == "zero":
            gate_signal = None
        elif gate_override == "shuffle" and gate_signal is not None:
            gate_signal = shuffle_valid_stream(gate_signal)
        if gate_signal is not None:
            # Keep every intervention on the observed stream. In particular,
            # temporal-shuffle controls must not move signal into query,
            # answer, or right-padding positions.
            gate_signal = gate_signal * stream_mask
            h = self.blocks[1](first_features, prediction_error=gate_signal)
        else:
            h = self.blocks[1](first_features)
        h = self.normalize_output(h)
        diagnostics = {
            "prediction_error": error_signal.detach(),
            "gate_signal": (
                gate_signal.detach() if gate_signal is not None else error_signal.new_zeros(error_signal.shape)
            ),
        }
        return h, first_features, logits, diagnostics

    def normalize_output(self, h: torch.Tensor) -> torch.Tensor:
        """Normalize the final residual stream before exposing backbone features."""
        numeric_observer = getattr(self, "_initium_forward_numeric_observer", None)
        if numeric_observer is not None:
            numeric_observer("output_norm.input", h)
        # Keep normalization math in fp32 even when the caller is under AMP;
        # return the incoming dtype to preserve the backbone's public dtype
        # contract and mixed-precision memory/output paths.
        with torch.autocast(device_type=h.device.type, enabled=False):
            normalized = self.output_norm(h.float()).to(dtype=h.dtype)
        if numeric_observer is not None:
            numeric_observer("output_norm.output", normalized)
        return normalized

    def forward_with_specialists(
        self,
        x: torch.Tensor,
        *,
        token_mask: torch.Tensor | None = None,
        intermediate_observer=None,
    ):
        """Return final MoE candidates and sparse routes for workspace use.

        Earlier backbone/MoE blocks remain unchanged. The final MoE block
        exposes expert deltas before routing so the workspace can broadcast
        context to every specialist, then reuse the original sparse route.
        The split-graph caller applies this block's configured residual scale
        to the combined routed and broadcast delta.
        """
        if not self.moe_enabled or self.moe_blocks is None:
            raise RuntimeError("workspace broadcast requires an enabled MoE expert pool")
        h = self.in_adapter(x)
        numeric_observer = getattr(self, "_initium_forward_numeric_observer", None)
        if numeric_observer is not None:
            numeric_observer("input_adapter.output", h)
        final_index = len(self.blocks) - 1
        if not isinstance(self.moe_blocks[final_index], MoEBlock):
            raise RuntimeError(
                "workspace broadcast requires the final backbone block to have an active MoE bank"
            )
        for i, block in enumerate(self.blocks):
            h = block(h)
            if i == 0 and intermediate_observer is not None:
                intermediate_observer(h)
            if numeric_observer is not None:
                numeric_observer(f"block.{i}.output", h)
            if i == final_index:
                base = h
                _, experts, indices, weights = self.moe_blocks[i](
                    h, return_details=True, token_mask=token_mask
                )
                if numeric_observer is not None:
                    numeric_observer(f"block.{i}.specialists", experts)
                    numeric_observer(f"block.{i}.route_weights", weights)
                return base, experts, indices, weights
            if isinstance(self.moe_blocks[i], MoEBlock):
                h = self.moe_blocks[i](h, token_mask=token_mask)
                if numeric_observer is not None:
                    numeric_observer(f"block.{i}.moe_output", h)
        raise RuntimeError("parallel backbone has no blocks")
