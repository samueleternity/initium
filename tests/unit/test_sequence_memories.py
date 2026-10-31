import pytest
import torch

from initium.memory_manipulation.sequence_memories import CausalKVMemory, GatedDeltaMemory


@pytest.mark.parametrize("lag", [0, 1])
@pytest.mark.parametrize("chunk", [1, 3, 8])
def test_delta_forward_matches_step_reference_and_gradients(lag, chunk):
    torch.manual_seed(12)
    memory = GatedDeltaMemory(7, heads=2, key_size=4, value_size=3, chunk_size=chunk)
    torch.nn.init.normal_(memory.output.weight)
    x = torch.randn(2, 7, 7, requires_grad=True)
    actual, state, _ = memory(x, lag=lag)
    expected_state = memory.reset_state(2)
    expected = []
    for t in range(x.shape[1]):
        read, expected_state = memory.step(x[:, t], expected_state, lag=lag)
        expected.append(read)
    expected = torch.stack(expected, dim=1)
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(state.matrix, expected_state.matrix)
    (actual.sum() + state.matrix.sum()).backward()
    actual_grads = [p.grad.clone() for p in memory.parameters()]
    actual_x_grad = x.grad.clone()
    memory.zero_grad(set_to_none=True)
    x.grad = None
    ref_state = memory.reset_state(2)
    reference_steps = []
    for t in range(x.shape[1]):
        read, ref_state = memory.step(x[:, t], ref_state, lag=lag)
        reference_steps.append(read)
    (torch.stack(reference_steps, dim=1).sum() + ref_state.matrix.sum()).backward()
    for parameter, expected_grad in zip(memory.parameters(), actual_grads, strict=True):
        torch.testing.assert_close(parameter.grad, expected_grad)
    torch.testing.assert_close(x.grad, actual_x_grad)


@pytest.mark.parametrize("lag", [0, 1])
@pytest.mark.parametrize("position_mode", ["none", "rope"])
def test_kv_forward_matches_step_reference_with_and_without_cached_state(lag, position_mode):
    torch.manual_seed(4)
    memory = CausalKVMemory(
        8, heads=2, key_size=4, value_size=3, position_mode=position_mode
    )
    torch.nn.init.normal_(memory.output.weight)
    prefix = torch.randn(2, 3, 8)
    suffix = torch.randn(2, 5, 8)
    _, prefix_state, _ = memory(prefix, lag=lag)
    actual, state, _ = memory(suffix, state=prefix_state, lag=lag)
    expected_state = prefix_state
    expected = []
    for t in range(suffix.shape[1]):
        read, expected_state = memory.step(suffix[:, t], expected_state, lag=lag)
        expected.append(read)
    torch.testing.assert_close(actual, torch.stack(expected, dim=1))
    torch.testing.assert_close(state.keys, expected_state.keys)
    torch.testing.assert_close(state.values, expected_state.values)


def test_kv_memory_is_causal():
    torch.manual_seed(3)
    memory = CausalKVMemory(8, heads=2, key_size=4, value_size=3)
    torch.nn.init.normal_(memory.output.weight)
    x = torch.randn(1, 6, 8)
    changed = x.clone()
    changed[:, 4:] += 50
    output, _, _ = memory(x, lag=1)
    changed_output, _, _ = memory(changed, lag=1)
    torch.testing.assert_close(output[:, :4], changed_output[:, :4])


def test_kv_lag_one_first_read_is_zero_with_trained_output_projection():
    torch.manual_seed(8)
    memory = CausalKVMemory(8, heads=2, key_size=4, value_size=3)
    torch.nn.init.normal_(memory.output.weight)
    torch.nn.init.normal_(memory.output.bias)
    output, _, _ = memory(torch.randn(2, 4, 8), lag=1)
    torch.testing.assert_close(output[:, :1], torch.zeros_like(output[:, :1]))


@pytest.mark.parametrize(
    "memory", [
        GatedDeltaMemory(8, heads=2, key_size=4, value_size=3, chunk_size=2),
        CausalKVMemory(8, heads=2, key_size=4, value_size=3),
    ],
)
def test_memory_output_is_zero_initialized(memory):
    read, _, _ = memory(torch.randn(2, 5, 8))
    torch.testing.assert_close(read, torch.zeros_like(read))


def test_delta_forward_matches_step_under_cpu_autocast():
    memory = GatedDeltaMemory(8, heads=2, key_size=4, value_size=3, chunk_size=3)
    x = torch.randn(2, 5, 8)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        actual, actual_state, _ = memory(x, lag=1)
        state = memory.reset_state(2)
        expected = []
        for index in range(x.shape[1]):
            read, state = memory.step(x[:, index], state, lag=1)
            expected.append(read)
    torch.testing.assert_close(actual, torch.stack(expected, dim=1), atol=2e-3, rtol=2e-3)
    torch.testing.assert_close(actual_state.matrix, state.matrix, atol=2e-3, rtol=2e-3)


@pytest.mark.gpu
@pytest.mark.parametrize("lag", [0, 1])
def test_fla_delta_matches_step_reference(lag):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for the FLA delta kernel")
    pytest.importorskip("fla")
    torch.manual_seed(19)
    memory = GatedDeltaMemory(
        8, heads=2, key_size=16, value_size=16, chunk_size=16, use_fla=True
    ).cuda()
    torch.nn.init.normal_(memory.output.weight)
    x = torch.randn(2, 19, 8, device="cuda", requires_grad=True)
    actual, actual_state, _ = memory(x, lag=lag)
    expected_state = memory.reset_state(2, device=x.device)
    expected = []
    for index in range(x.shape[1]):
        read, expected_state = memory.step(x[:, index], expected_state, lag=lag)
        expected.append(read)
    expected = torch.stack(expected, dim=1)
    torch.testing.assert_close(actual, expected, atol=3e-4, rtol=3e-4)
    torch.testing.assert_close(actual_state.matrix, expected_state.matrix, atol=3e-4, rtol=3e-4)
    (actual.sum() + actual_state.matrix.sum()).backward()
    fast_grads = [parameter.grad.clone() for parameter in memory.parameters()]
    memory.zero_grad(set_to_none=True)
    x.grad = None
    ref_state = memory.reset_state(2, device=x.device)
    reference = []
    for index in range(x.shape[1]):
        read, ref_state = memory.step(x[:, index], ref_state, lag=lag)
        reference.append(read)
    (torch.stack(reference, dim=1).sum() + ref_state.matrix.sum()).backward()
    for parameter, fast_grad in zip(memory.parameters(), fast_grads, strict=True):
        torch.testing.assert_close(parameter.grad, fast_grad, atol=1e-3, rtol=1e-3)
