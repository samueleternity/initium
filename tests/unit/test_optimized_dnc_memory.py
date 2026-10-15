import torch
from dnc.memory import Memory

from initium.memory_manipulation.dnc_mechanisms import (
    MechanismMemory,
    OptimizedMechanismMemory,
    OptimizedMemory,
    build_memory,
)


def test_optimized_allocation_matches_base_values_and_gradients():
    base = Memory(input_size=8, nr_cells=19, cell_size=6, read_heads=2)
    optimized = OptimizedMemory(input_size=8, nr_cells=19, cell_size=6, read_heads=2)
    optimized.load_state_dict(base.state_dict())

    base_usage = torch.rand(3, 19, requires_grad=True)
    optimized_usage = base_usage.detach().clone().requires_grad_(True)
    write_gate = torch.ones(3, 1)

    base_weights, base_adjusted_usage = base.allocate(base_usage, write_gate)
    optimized_weights, optimized_adjusted_usage = optimized.allocate(
        optimized_usage, write_gate
    )

    torch.testing.assert_close(optimized_weights, base_weights)
    torch.testing.assert_close(optimized_adjusted_usage, base_adjusted_usage)

    weight = torch.randn_like(base_weights)
    (base_weights * weight).sum().backward()
    (optimized_weights * weight).sum().backward()
    torch.testing.assert_close(optimized_usage.grad, base_usage.grad)


def test_optimized_mechanism_memory_preserves_state_dict_and_factory_choice():
    base = build_memory(
        input_size=8,
        nr_cells=19,
        cell_size=6,
        read_heads=2,
        dnc_implementation="base",
    )
    optimized = build_memory(
        input_size=8,
        nr_cells=19,
        cell_size=6,
        read_heads=2,
        dnc_implementation="optimized",
    )

    assert isinstance(base, MechanismMemory)
    assert not isinstance(base, OptimizedMechanismMemory)
    assert isinstance(optimized, OptimizedMechanismMemory)
    assert base.state_dict().keys() == optimized.state_dict().keys()
    optimized.load_state_dict(base.state_dict(), strict=True)


def test_optimized_memory_forward_and_parameter_gradients_match_base():
    base = build_memory(
        input_size=8,
        nr_cells=11,
        cell_size=5,
        read_heads=2,
        dnc_implementation="base",
    )
    optimized = build_memory(
        input_size=8,
        nr_cells=11,
        cell_size=5,
        read_heads=2,
        dnc_implementation="optimized",
    )
    optimized.load_state_dict(base.state_dict(), strict=True)
    base_input = torch.randn(2, 8, requires_grad=True)
    optimized_input = base_input.detach().clone().requires_grad_(True)

    base_read, base_state = base(base_input, base.new(batch_size=2))
    optimized_read, optimized_state = optimized(
        optimized_input, optimized.new(batch_size=2)
    )

    torch.testing.assert_close(optimized_read, base_read)
    for key in base_state:
        torch.testing.assert_close(optimized_state[key], base_state[key])

    base_loss = base_read.square().sum() + sum(
        value.square().sum() for value in base_state.values()
    )
    optimized_loss = optimized_read.square().sum() + sum(
        value.square().sum() for value in optimized_state.values()
    )
    base_loss.backward()
    optimized_loss.backward()

    torch.testing.assert_close(optimized_input.grad, base_input.grad)
    base_params = dict(base.named_parameters())
    optimized_params = dict(optimized.named_parameters())
    assert base_params.keys() == optimized_params.keys()
    for name, parameter in base_params.items():
        torch.testing.assert_close(optimized_params[name].grad, parameter.grad)
