from initium.memory_manipulation.dynamic_memory_resize import resize_memory
from initium.memory_manipulation.dynamic_n_controller import DynamicNController


def test_dynamic_n_decision_applies_live_resize(build_tiny_rnn):
    model, _, _, optimizer = build_tiny_rnn("lstm")
    dynamic_n = DynamicNController(16, 32, trigger_frac=0.1, ema_decay=0.0, cooldown_steps=1)
    dynamic_n.record(1.0)
    target = dynamic_n.should_grow(model.memories[0].nr_cells)
    assert target == 32
    old_n = model.memories[0].nr_cells
    resize_memory(model, target, device="cpu", optimizer=optimizer)
    dynamic_n.mark_grown(1, old_n, target)
    assert model.memories[0].nr_cells == 32
    assert dynamic_n.growth_history == [(1, 16, 32)]
