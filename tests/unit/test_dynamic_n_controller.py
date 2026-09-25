import pytest
from memory_manipulation.dynamic_n_controller import DynamicNController


def test_ema_growth_cooldown_and_state_roundtrip():
    ctl = DynamicNController(
        4, 10, growth_factor=2, trigger_frac=0.5, ema_decay=0.0, cooldown_steps=2
    )
    ctl.record(0.25)
    assert ctl.ema == pytest.approx(0.25)
    assert ctl.should_grow(4) is None
    ctl.record(0.75)
    assert ctl.should_grow(4) == 8
    ctl.mark_grown(7, 4, 8)
    assert ctl.ema == 0
    assert ctl.should_grow(8) is None
    state = ctl.state_dict()
    restored = DynamicNController(2, 6, cooldown_steps=9)
    restored.load_state_dict(state)
    assert restored.growth_history == [(7, 4, 8)]
    assert restored.cooldown_remaining == 2
    assert (restored.floor, restored.ceiling, restored.cooldown_steps) == (2, 6, 9)
    restored.record(1.0)
    restored.record(1.0)
    assert restored.should_grow(8) is None
    assert restored.should_grow(6) is None
