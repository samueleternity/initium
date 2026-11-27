"""Contract tests for the detached neuromodulation controller."""

import random

import pytest
import torch

from initium.core_training import assign_group
from initium.training.neuromodulator import Neuromodulator

GROUPS = ("backbone", "moe", "workspace", "modulation", "task_head")


def test_parameter_group_assignment_covers_each_trainable_parameter_once():
    named_parameters = [
        ("rnn.backbone.blocks.0.mixer.in_proj.weight", torch.nn.Parameter(torch.ones(1))),
        ("rnn.moe_blocks.0.moe.router.weight", torch.nn.Parameter(torch.ones(1))),
        ("rnn.parallel_workspace.Wk.weight", torch.nn.Parameter(torch.ones(1))),
        ("rnn.modulation_head.out.weight", torch.nn.Parameter(torch.ones(1))),
        ("output_proj.weight", torch.nn.Parameter(torch.ones(1))),
        ("rnn.workspace.adaptive_v3.router.weight", torch.nn.Parameter(torch.ones(1))),
        ("rnn.workspace.adaptive_v4.operator.weight", torch.nn.Parameter(torch.ones(1))),
    ]
    grouped = {}
    for name, parameter in named_parameters:
        grouped.setdefault(assign_group(name), []).append(parameter)

    assigned = [parameter for parameters in grouped.values() for parameter in parameters]
    assert len(assigned) == len(named_parameters)
    assert {id(parameter) for parameter in assigned} == {
        id(parameter) for _, parameter in named_parameters
    }
    assert len({id(parameter) for parameter in assigned}) == len(assigned)
    assert set(grouped) == {*GROUPS, "adaptive_v3"}


def test_off_mode_keeps_optimizer_updates_bit_identical():
    baseline = torch.nn.Linear(3, 2)
    controlled = torch.nn.Linear(3, 2)
    controlled.load_state_dict(baseline.state_dict())
    baseline_opt = torch.optim.Adam(baseline.parameters(), lr=0.01)
    controlled_opt = torch.optim.Adam(controlled.parameters(), lr=0.01)
    controller = Neuromodulator(list(GROUPS), mode="off")
    x = torch.arange(60, dtype=torch.float32).reshape(20, 3) / 37
    target = torch.linspace(-1, 1, 40).reshape(20, 2)

    for step in range(20):
        baseline_opt.zero_grad(set_to_none=True)
        controlled_opt.zero_grad(set_to_none=True)
        baseline_loss = (baseline(x) - target).square().mean()
        controlled_loss = (controlled(x) - target).square().mean()
        baseline_loss.backward()
        controlled_loss.backward()
        scale = controller.multipliers()["backbone"]
        for group in controlled_opt.param_groups:
            group["lr"] = 0.01 * scale
        baseline_opt.step()
        controlled_opt.step()
        controller.observe(float(controlled_loss.detach()), 1.0, False)
        controller.update(step + 1)

    for expected, actual in zip(baseline.parameters(), controlled.parameters()):
        assert torch.equal(expected, actual)
    assert controller.multipliers() == {group: 1.0 for group in GROUPS}


def test_multipliers_remain_bounded_and_each_update_obeys_slew_limit():
    controller = Neuromodulator(
        list(GROUPS), r=1.8, update_every=1, slew=0.07, mode="random", seed=52
    )
    previous = controller.multipliers()

    for step in range(1, 101):
        current = controller.update(step)
        for group in GROUPS:
            assert 1 / 1.8 <= current[group] <= 1.8
            assert abs(current[group] - previous[group]) <= 0.07 + 1e-12
        previous = current


def test_checkpoint_round_trip_restores_controller_and_random_state_exactly():
    controller = Neuromodulator(list(GROUPS), update_every=1, slew=0.2, mode="random", seed=73)
    for step in range(1, 5):
        controller.observe(2.0 - 0.1 * step, 1.0 + 0.1 * step, step % 2 == 0)
        controller.update(step)
    restored = Neuromodulator(list(GROUPS), update_every=1, slew=0.2, mode="random", seed=999)
    restored.load_state_dict(controller.state_dict())

    assert restored.state_dict() == controller.state_dict()
    assert restored.update(5) == controller.update(5)


def test_random_control_is_seeded_and_independent_of_global_rng():
    first = Neuromodulator(list(GROUPS), update_every=1, mode="random", seed=81)
    second = Neuromodulator(list(GROUPS), update_every=1, mode="random", seed=81)
    random.seed(5)
    first_values = []
    second_values = []
    for step in range(1, 12):
        first_values.append(first.update(step))
        for _ in range(3):
            random.random()
        second_values.append(second.update(step))

    assert first_values == second_values


def test_lr_mode_raises_on_plateau_and_lowers_when_clipping_is_frequent():
    controller = Neuromodulator(
        ["backbone"],
        ema_fast=0.0,
        ema_slow=0.0,
        update_every=1,
        slew=0.25,
        mode="lr",
    )
    controller.observe(1.0, 1.0, False)
    assert controller.update(1)["backbone"] == pytest.approx(1.25)

    for _ in range(30):
        controller.observe(1.0, 1.0, True)
    assert controller.update(2)["backbone"] == pytest.approx(1.0)


def test_plasticity_mode_uses_retention_dependency_and_moe_signals_by_group():
    controller = Neuromodulator(list(GROUPS), update_every=1, slew=1.0, mode="plasticity")
    controller.observe(1.0, 1.0, False)
    values = controller.update(
        1,
        lesson_event=True,
        eval_signals={
            "retention_drop": 0.2,
            "workspace_dependency_drop": 0.01,
            "moe_cv_load": 0.8,
            "moe_capacity_drop_frac": 0.05,
        },
    )
    assert values["backbone"] == pytest.approx(0.5)
    assert values["workspace"] == pytest.approx(2.0)
    assert values["modulation"] == pytest.approx(2.0)
    assert values["moe"] == pytest.approx(2.0)

    values = controller.update(
        2,
        eval_signals={
            "workspace_dependency_drop": 0.5,
            "moe_cv_load": 0.1,
            "moe_capacity_drop_frac": 0.4,
        },
    )
    assert values["workspace"] == pytest.approx(1.0)
    assert values["modulation"] == pytest.approx(1.0)
    assert values["moe"] == pytest.approx(1.0)


def test_consolidation_mode_protects_backbone_and_reduces_workspace_and_head_rates():
    controller = Neuromodulator(list(GROUPS), update_every=1, slew=1.0, mode="consolidation")
    values = controller.update(1)

    assert values["backbone"] == pytest.approx(2.0)
    assert values["workspace"] == pytest.approx(0.5)
    assert values["modulation"] == pytest.approx(0.5)
    assert values["task_head"] == pytest.approx(0.5)
