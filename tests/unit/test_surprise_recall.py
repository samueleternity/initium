import random

import pytest
import torch

from initium.data.surprise_recall.surprise_recall import (
    LESSONS,
    VOCAB_SIZE,
    SurpriseRecallDataset,
    build_episode,
    oracle_answer,
    verify_labels,
)


@pytest.mark.parametrize("lesson_index", range(len(LESSONS)))
def test_generated_episode_labels_and_answer_match_oracle(lesson_index):
    episode = build_episode(lesson_index, random.Random(10_000 + lesson_index))

    assert verify_labels(episode)
    assert episode.inputs[episode.answer_start - 1] == 20  # query precedes answer
    assert episode.inputs[episode.answer_start :] == oracle_answer(episode)
    assert episode.targets[episode.answer_start :] == oracle_answer(episode)
    assert len(episode.inputs) == len(episode.targets)


def test_dataset_batch_has_one_hot_inputs_and_answer_only_loss_mask():
    dataset = SurpriseRecallDataset()
    curriculum = dataset.make_curriculum()
    inputs, targets, loss_mask = dataset.sample_batch(curriculum, batch_size=3)

    assert inputs.ndim == targets.ndim == loss_mask.ndim == 2
    assert inputs.shape[:2] == targets.shape == loss_mask.shape
    assert inputs.shape[-1] == VOCAB_SIZE
    torch.testing.assert_close(inputs.sum(dim=-1), (inputs.sum(dim=-1) > 0).float())
    assert set(loss_mask.unique().tolist()) <= {0.0, 1.0}
    for row, example in enumerate(dataset.last_batch_examples):
        assert loss_mask[row, : example.answer_start].sum() == 0
        assert loss_mask[row, example.answer_start : len(example.inputs)].sum() > 0


def test_task_loss_ignores_non_answer_positions_including_nonfinite_logits():
    dataset = SurpriseRecallDataset()
    logits = torch.randn(1, 5, VOCAB_SIZE, requires_grad=True)
    targets = torch.tensor([[0, 0, 0, 4, 22]])
    mask = torch.tensor([[0.0, 0.0, 0.0, 1.0, 1.0]])
    with torch.no_grad():
        logits[0, :3] = float("nan")

    loss = dataset.loss(logits, targets, mask)

    assert torch.isfinite(loss)
    loss.backward()
    assert logits.grad is not None
    assert torch.equal(logits.grad[0, :3], torch.zeros_like(logits.grad[0, :3]))
    assert torch.isfinite(logits.grad[0, 3:]).all()


def test_dataset_loss_returns_zero_when_there_are_no_answer_positions():
    dataset = SurpriseRecallDataset()
    logits = torch.randn(2, 4, VOCAB_SIZE)
    targets = torch.zeros(2, 4, dtype=torch.long)
    mask = torch.zeros(2, 4)

    loss = dataset.loss(logits, targets, mask)

    assert loss.item() == pytest.approx(0.0)
