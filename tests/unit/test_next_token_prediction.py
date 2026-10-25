import torch
from torch import nn

from initium.data.surprise_recall.next_token_prediction import (
    SurprisePredictionHead,
    evaluate_prediction_signals,
    masked_next_token_loss,
    prediction_error_signal,
)
from initium.data.surprise_recall.surprise_recall import VOCAB_SIZE


def _one_hot(token_ids):
    return torch.nn.functional.one_hot(
        torch.tensor(token_ids, dtype=torch.long), num_classes=VOCAB_SIZE
    ).float()


def test_prediction_error_signal_is_bounded_detached_and_masks_query_and_padding():
    input_one_hot = _one_hot([[1, 2, 3, 20, 21], [4, 5, 6, 7, 20]])
    features = torch.randn(2, 5, 4, requires_grad=True)
    decoder = SurprisePredictionHead(4)

    signal, logits = prediction_error_signal(
        features,
        input_one_hot,
        decoder,
        signal_mode="excess-surprise",
        stream_lengths=[3, 4],
    )

    assert logits.shape == (2, 4, VOCAB_SIZE)
    assert signal.shape == (2, 5)
    assert not signal.requires_grad
    assert torch.all(signal.abs() <= 1.0)
    assert torch.equal(signal[0, 3:], torch.zeros(2))
    assert torch.equal(signal[1, 4:], torch.zeros(1))


def test_prediction_error_signal_rejects_misaligned_inputs():
    decoder = SurprisePredictionHead(4)
    with torch.no_grad():
        bad_features = torch.randn(2, 4, 4)
        tokens = _one_hot([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]])

    try:
        prediction_error_signal(
            bad_features,
            tokens,
            decoder,
            signal_mode="surprisal",
        )
    except ValueError as exc:
        assert "matching batch/time" in str(exc)
    else:
        raise AssertionError("misaligned feature and input time dimensions were accepted")


def test_masked_next_token_loss_ignores_nonstream_nan_logits_and_gradient():
    inputs = _one_hot([[1, 2, 3, 20, 21]])
    logits = torch.randn(1, 4, VOCAB_SIZE, requires_grad=True)
    with torch.no_grad():
        logits[0, 2:] = float("nan")

    loss = masked_next_token_loss(logits, inputs, [3])

    assert torch.isfinite(loss)
    loss.backward()
    assert logits.grad is not None
    assert torch.equal(logits.grad[0, 2:], torch.zeros_like(logits.grad[0, 2:]))
    assert torch.isfinite(logits.grad[0, :2]).all()


def test_masked_next_token_loss_returns_differentiable_zero_without_stream_targets():
    inputs = _one_hot([[20, 21, 0]])
    logits = torch.randn(1, 2, VOCAB_SIZE, requires_grad=True)

    loss = masked_next_token_loss(logits, inputs, [1])

    assert loss.item() == 0.0
    loss.backward()
    assert logits.grad is not None
    assert torch.equal(logits.grad, torch.zeros_like(logits.grad))


def test_evaluation_uses_fixed_seed_and_restores_module_modes():
    backbone = nn.Linear(VOCAB_SIZE, 8)
    decoder = SurprisePredictionHead(8)
    backbone.train()
    decoder.train()

    first = evaluate_prediction_signals(
        backbone,
        decoder,
        torch.device("cpu"),
        episodes_per_lesson=2,
        batch_size=1,
        lesson_indices=(0,),
        seed=123,
    )
    second = evaluate_prediction_signals(
        backbone,
        decoder,
        torch.device("cpu"),
        episodes_per_lesson=2,
        batch_size=1,
        lesson_indices=(0,),
        seed=123,
    )

    assert first == second
    assert backbone.training and decoder.training
    assert first["per_type"]["A2"]["count"] == 0
    assert first["calibration"]["count"] > 0
