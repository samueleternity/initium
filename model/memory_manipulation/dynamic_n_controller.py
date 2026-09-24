"""
file: dynamic_n_controller.py

Dynamic-N, macro-scale (between episodes): grow nr_cells when the model's
own usage_vector says it needs more capacity, instead of reading a fixed
per-lesson lookup table (LESSON_NR_CELLS / static Option 2). Per the
strategy doc ("Whether Option 1 can be made dynamic the same way"), the
macro-scale case is mechanically just static Option 2's resize mechanism
(dynamic_memory_resize.resize_memory) with a different *trigger* -- no new
tensor-resizing logic is needed here, only the decision of *when* to call
it. This module owns only that decision.

--- Why this is split into record() / should_grow() / mark_grown() --------
The eventual mid-episode (micro-scale) variant needs the exact same
"sustained saturation, not a single-step spike, with a cooldown after each
growth event" decision logic (see the strategy doc's point 1 and point 3),
just fed from a per-timestep usage reading instead of a per-episode one,
and executing a live in-place pad instead of resize_memory()'s
rebuild-and-swap. Keeping the decision core (this class) ignorant of *when*
it's called and *how* growth is physically applied means that future
variant only has to replace the call site (what feeds record(), and what
apply_growth does), not re-derive the EMA/cooldown/ceiling logic:

  - record(frac_saturated): update the state used by should_grow(). Caller
    decides the cadence (once per episode here; once per timestep for the
    micro-scale variant later).
  - should_grow(): pure decision, no side effects, no knowledge of *how*
    growth is applied. Returns the next nr_cells or None.
  - mark_grown(step, old_n, new_n): call AFTER growth has actually been
    applied (by whatever mechanism), to reset the cooldown/EMA and append
    to growth_history. Decoupled from resize_memory() specifically, so a
    future in-place-pad apply function can call it too.

The actual resize call (dynamic_memory_resize.resize_memory) happens at the
call site in Alter_PHASE3_mamba.py's training loop, not in this file --
same "this module doesn't do the tensor work" split link_matrix_ablation.py
uses relative to dynamic_memory_resize.py.
"""



class DynamicNController:
    """Usage-triggered nr_cells growth, macro-scale (between episodes).

    Trigger: an EMA of "fraction of memory cells with usage_vector above
    usage_high", computed fresh each time record() is called. EMA (not the
    raw per-step value) is the sustaining filter the strategy doc's point 1
    calls for -- a single saturated episode doesn't trigger growth, only a
    persistently saturated one does, which also naturally damps against
    growing too eagerly (the strategy doc's other flagged failure mode:
    re-importing Option 2's original "excess capacity is an active cost"
    problem, Concept 17).

    Growth: multiplicative (nr_cells *= growth_factor each event, clamped
    to ceiling), matching the strategy doc's own 128->256->512 example.
    Floor is enforced only at construction (the starting nr_cells passed to
    the model) -- this controller only ever grows, never shrinks; shrinking
    an active run is a different, harder problem (would need to reconcile
    with live content) and isn't part of this task.

    Cooldown: after every growth event, should_grow() returns None for
    cooldown_steps worth of record() calls, regardless of EMA, so the
    controller doesn't fire again while the model's addressing policy is
    still re-settling from the last resize (strategy doc point 3,
    distribution-shift-at-growth risk). The EMA is also reset to 0 on
    growth so the post-growth cooldown window starts from a clean read
    rather than the already-saturated value that just triggered it.
    """

    def __init__(
        self,
        floor: int,
        ceiling: int,
        growth_factor: float = 2.0,
        trigger_frac: float = 0.75,
        ema_decay: float = 0.98,
        cooldown_steps: int = 2000,
    ):
        if floor > ceiling:
            raise ValueError(f"DynamicNController: floor ({floor}) > ceiling ({ceiling})")
        self.floor = floor
        self.ceiling = ceiling
        self.growth_factor = growth_factor
        self.trigger_frac = trigger_frac
        self.ema_decay = ema_decay
        self.cooldown_steps = cooldown_steps

        self.ema: float = 0.0
        self.cooldown_remaining: int = 0
        # (step, old_n, new_n) for every growth event this run has applied
        # (restored from checkpoint on resume -- see state_dict/load_state_dict).
        self.growth_history: list[tuple[int, int, int]] = []

    def record(self, frac_saturated: float) -> None:
        """Update the sustained-saturation EMA. Call once per decision
        cadence (once per episode/training-step for macro-scale) with the
        current episode's fraction of memory cells above the usage-high
        threshold -- see the training-loop call site for how that fraction
        is computed from usage_vector.
        """
        self.ema = self.ema_decay * self.ema + (1.0 - self.ema_decay) * frac_saturated
        if self.cooldown_remaining > 0:
            self.cooldown_remaining -= 1

    def should_grow(self, current_nr_cells: int) -> int | None:
        """Returns the next nr_cells to resize to, or None if no growth
        should happen right now. Pure decision -- does not itself resize
        anything or mutate controller state (mark_grown does that, and only
        after the caller has actually applied the growth).
        """
        if self.cooldown_remaining > 0:
            return None
        if current_nr_cells >= self.ceiling:
            return None
        if self.ema < self.trigger_frac:
            return None
        proposed = int(min(current_nr_cells * self.growth_factor, self.ceiling))
        if proposed <= current_nr_cells:
            return None
        return proposed

    def mark_grown(self, step: int, old_n: int, new_n: int) -> None:
        """Call immediately after growth has actually been applied (by
        resize_memory() at the macro-scale call site, or by whatever
        mid-episode mechanism the micro-scale variant eventually uses).
        Resets the EMA and starts the post-growth cooldown.
        """
        self.growth_history.append((step, old_n, new_n))
        self.ema = 0.0
        self.cooldown_remaining = self.cooldown_steps

    def state_dict(self) -> dict:
        """For checkpointing -- see save_checkpoint()'s dynamic_n_state key.
        Deliberately does NOT include floor/ceiling/growth_factor/
        trigger_frac/ema_decay/cooldown_steps: those are run configuration
        (re-supplied from CLI args / module constants on resume, same
        convention as link_matrix_mode/topk are re-passed rather than
        read back out of the checkpoint), not run *state*. Current
        nr_cells is also deliberately NOT stored here -- it's already
        self-describing via model_config["nr_cells"] (read live from
        rnn.memories[0].nr_cells), the same single-source-of-truth
        convention static Option 2's checkpoint format already uses.
        """
        return {
            "ema": self.ema,
            "cooldown_remaining": self.cooldown_remaining,
            "growth_history": list(self.growth_history),
        }

    def load_state_dict(self, state: dict) -> None:
        self.ema = state.get("ema", 0.0)
        self.cooldown_remaining = state.get("cooldown_remaining", 0)
        self.growth_history = list(state.get("growth_history", []))
