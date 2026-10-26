import csv
import io
import random
from collections import Counter

import pytest
import torch

from initium.data.dataset_registry import get_dataset
from initium.data.workspace_capability import dataset as capability_dataset
from initium.data.workspace_capability.curriculum import (
    WorkspaceCapabilityCurriculum,
    difficulty_for,
)
from initium.data.workspace_capability.episode import (
    BOS,
    BOOL_FALSE,
    BOOL_TRUE,
    COUNT_BASE,
    ENTITY_BASE,
    EOS,
    EV_ASSERT,
    EV_COUNT_QUERY,
    EV_FACT,
    EV_INIT,
    EV_KV,
    EV_MOVE,
    EV_PARITY_QUERY,
    EV_PUT,
    EV_REMOVE,
    EV_RETRACT,
    EV_SUM_QUERY,
    EV_SUPERSEDE,
    EV_SWAP,
    EV_UPDATE,
    EV_VECTOR,
    FACT_BASE,
    LOCATION_BASE,
    NONE_TOKEN,
    QUERY,
    SCALAR_BASE,
    SEP,
    SOURCE_BASE,
    TaskEpisode,
    VOCAB_SIZE,
    validate_episode,
)
from initium.data.workspace_capability.tracks import (
    entity_state,
    nth_farthest,
    permutation,
    relational_join,
    retrieval_aggregation,
)


GENERATORS = {
    "permutation": permutation.generate,
    "entity_state": entity_state.generate,
    "relational_join": relational_join.generate,
    "nth_farthest": nth_farthest.generate,
    "retrieval_aggregation": retrieval_aggregation.generate,
}


def _segments(episode):
    result, current = [], []
    for token in episode.inputs:
        if token == BOS and not current and not result:
            continue
        if token == SEP:
            if current:
                result.append(current)
                current = []
        else:
            current.append(token)
    if current:
        result.append(current)
    return result


@pytest.mark.parametrize("track,generator", GENERATORS.items())
def test_track_generators_are_seed_reproducible_and_emit_valid_masked_episodes(track, generator):
    difficulty = difficulty_for(track, 3)
    first = validate_episode(generator(difficulty, random.Random(177), ood=False))
    second = validate_episode(generator(difficulty, random.Random(177), ood=False))

    assert first == second
    assert first.track == track
    assert first.inputs[0] == BOS
    assert first.answer[-1] == EOS
    assert max(first.inputs + first.answer) < VOCAB_SIZE
    assert len(first.sequence) == len(first.targets) == len(first.answer_mask)
    assert not any(first.answer_mask[: len(first.inputs)])
    assert all(first.answer_mask[len(first.inputs) :])
    assert first.metadata["answer_count"] == len(first.answer) - 1


@pytest.mark.parametrize("track,generator", GENERATORS.items())
def test_ood_track_generation_stays_inside_the_shared_vocabulary(track, generator):
    episode = validate_episode(generator(difficulty_for(track, 5, ood=True), random.Random(901), ood=True))
    assert episode.track == track
    assert min(episode.inputs + episode.answer) >= 0
    assert max(episode.inputs + episode.answer) < VOCAB_SIZE


def test_permutation_oracle_composes_swaps_and_answers_both_directions():
    episode = permutation.generate(
        {"entities": 8, "swaps": 18, "distractors": 4}, random.Random(22)
    )
    segments = _segments(episode)
    state, inverse = {}, {}
    query = None
    for segment in segments:
        event = segment[0]
        if event == EV_INIT:
            entity, position = segment[1] - ENTITY_BASE, segment[2] - LOCATION_BASE
            state[entity] = position
            inverse[position] = entity
        elif event == EV_SWAP:
            left, right = segment[1] - ENTITY_BASE, segment[2] - ENTITY_BASE
            state[left], state[right] = state[right], state[left]
            inverse[state[left]], inverse[state[right]] = left, right
        elif event == QUERY:
            query = segment[1:]
    assert query is not None
    if query[0] >= ENTITY_BASE:
        expected = LOCATION_BASE + state[query[0] - ENTITY_BASE]
    else:
        expected = ENTITY_BASE + inverse[query[0] - LOCATION_BASE]
    assert episode.answer == (expected, EOS)


def test_entity_state_oracle_uses_priority_and_applies_retract_supersede():
    difficulty = {
        "entities": 6,
        "sources": 5,
        "updates": 8,
        "distractors": 3,
        "correction": True,
        "correction_probability": 1.0,
        "novelty": True,
        "novelty_probability": 1.0,
    }
    episode = entity_state.generate(difficulty, random.Random(7))
    facts = {}
    query_entity = query_relation = None
    sequence = 0
    for segment in _segments(episode):
        event = segment[0]
        if event in (EV_PUT, EV_ASSERT):
            source, fact_id, entity, relation, value = segment[1:]
            sequence += 1
            facts[fact_id] = (source - SOURCE_BASE, entity - ENTITY_BASE,
                              relation, value - LOCATION_BASE, sequence)
        elif event == EV_RETRACT:
            facts.pop(segment[2], None)
        elif event == EV_MOVE:
            source, old_id, entity, relation, old_value, new_source, new_id, new_value = segment[1:]
            facts.pop(old_id, None)
            sequence += 1
            facts[new_id] = (new_source - SOURCE_BASE, entity - ENTITY_BASE,
                             relation, new_value - LOCATION_BASE, sequence)
        elif event == EV_SUPERSEDE:
            source, old_id, new_id, entity, relation, value = segment[1:]
            facts.pop(old_id, None)
            sequence += 1
            facts[new_id] = (source - SOURCE_BASE, entity - ENTITY_BASE,
                             relation, value - LOCATION_BASE, sequence)
        elif event == EV_REMOVE:
            facts.pop(segment[2], None)
        elif event == QUERY:
            _policy, entity, query_relation = segment[1:]
            query_entity = entity - ENTITY_BASE

    assert query_entity is not None
    candidates = [
        (source, value, timestamp)
        for source, entity, relation, value, timestamp in facts.values()
        if entity == query_entity and relation == query_relation
    ]
    if candidates:
        winner = max(candidates, key=lambda item: (item[0] % 3, item[2], -item[0]))
        expected = LOCATION_BASE + winner[1]
    else:
        expected = NONE_TOKEN
    assert episode.answer == (expected, EOS)
    assert episode.metadata["correction_relevant"]
    assert any(segment[0] == EV_RETRACT for segment in _segments(episode))
    assert any(segment[0] == EV_SUPERSEDE for segment in _segments(episode))
    fact_reference = next(
        token for segment in _segments(episode) for token in segment if token >= FACT_BASE
    )
    assert FACT_BASE <= fact_reference < VOCAB_SIZE


def test_entity_source_priority_then_timestamp_then_source_id_tie_break():
    facts = [
        (10, 3, 0, 100),  # priority 0
        (11, 2, 1, 1),    # priority 2 wins despite older time
        (12, 5, 2, 2),    # same priority, newer time wins
        (13, 2, 3, 2),    # exact priority/time tie; lower source ID wins
    ]
    assert entity_state._winner(facts) == facts[3]


def test_relational_join_oracle_follows_the_requested_relation_chain():
    episode = relational_join.generate(
        {"hops": 3, "distractors": 11}, random.Random(91)
    )
    facts = []
    query = None
    for segment in _segments(episode):
        if segment[0] == EV_FACT:
            _event, _source, subject, relation, obj = segment
            facts.append((subject, relation, obj))
        elif segment[0] == QUERY:
            query = segment[1:]
    assert query is not None
    current = query[0]
    for relation in query[1:]:
        next_entities = [obj for subject, edge, obj in facts if subject == current and edge == relation]
        assert len(next_entities) == 1
        current = next_entities[0]
    assert episode.answer == (current, EOS)


def test_nth_farthest_oracle_ranks_candidates_by_squared_distance():
    episode = nth_farthest.generate(
        {"candidates": 9, "dimensions": 4, "rank": 5, "distractors": 3},
        random.Random(31),
    )
    vectors, query, rank = {}, None, None
    for segment in _segments(episode):
        if segment[0] == EV_VECTOR:
            entity = segment[1]
            vectors[entity] = [value - SCALAR_BASE for value in segment[2:]]
        elif segment[0] == QUERY:
            rank = segment[1] - SCALAR_BASE + 1
            query = [value - SCALAR_BASE for value in segment[2:]]
    assert query is not None and rank is not None
    ranked = sorted(
        vectors,
        key=lambda entity: (
            -sum((left - right) ** 2 for left, right in zip(vectors[entity], query)),
            entity,
        ),
    )
    assert episode.answer == (ranked[rank - 1], EOS)


@pytest.mark.parametrize(
    "aggregate,op_token",
    [("sum_mod16", EV_SUM_QUERY), ("count_even", EV_COUNT_QUERY), ("parity", EV_PARITY_QUERY)],
)
def test_retrieval_aggregation_oracle_uses_latest_values(aggregate, op_token):
    episode = retrieval_aggregation.generate(
        {"keys": 9, "updates": 12, "distractors": 7, "queries": 5,
         "aggregate": aggregate},
        random.Random(81),
    )
    state, query_keys = {}, None
    for segment in _segments(episode):
        event = segment[0]
        if event in (EV_KV, EV_UPDATE):
            state[segment[1] - ENTITY_BASE] = segment[2] - SCALAR_BASE
        elif event == QUERY:
            assert segment[1] == op_token
            query_keys = [key - ENTITY_BASE for key in segment[2:]]
    assert query_keys is not None
    values = [state[key] for key in query_keys]
    if aggregate == "sum_mod16":
        expected = SCALAR_BASE + sum(values) % 16
    elif aggregate == "count_even":
        expected = COUNT_BASE + sum(value % 2 == 0 for value in values)
    else:
        expected = BOOL_TRUE if sum(values) % 2 else BOOL_FALSE
    assert episode.answer == (expected, EOS)


def test_all_track_curriculum_has_separate_schedules_and_requires_each_track(monkeypatch):
    dataset = capability_dataset.WorkspaceCapabilityDataset("all")
    curriculum = WorkspaceCapabilityCurriculum(dataset)
    assert len(curriculum.table) == 6
    assert set(curriculum.table[0]) == set(GENERATORS)

    def mocked_evaluate(*args, **kwargs):
        dataset.last_eval_diagnostics = {
            "tracks": {track: {"accuracy": 90.0} for track in GENERATORS}
        }
        dataset.last_eval_diagnostics["tracks"]["nth_farthest"]["accuracy"] = 40.0
        return 0, 88.0, 70.0

    monkeypatch.setattr(dataset, "evaluate_id", mocked_evaluate)
    curriculum.maybe_advance(None, torch.device("cpu"), step=10)
    assert curriculum.lesson == 0

    def all_ready(*args, **kwargs):
        dataset.last_eval_diagnostics = {
            "tracks": {track: {"accuracy": 90.0} for track in GENERATORS}
        }
        return 0, 88.0, 70.0

    monkeypatch.setattr(dataset, "evaluate_id", all_ready)
    curriculum.maybe_advance(None, torch.device("cpu"), step=20)
    assert curriculum.lesson == 1


def test_registry_exposes_workspace_track_selection():
    dataset = get_dataset(
        "workspace-capability", workspace_task_track="nth-farthest", batch_size_hint=4
    )
    assert dataset.name == "workspace-capability"
    assert dataset.task_track == "nth_farthest"
    assert dataset.batch_size_hint == 4


def test_batch_collation_is_one_hot_and_loss_uses_only_answer_tokens():
    dataset = capability_dataset.WorkspaceCapabilityDataset("all", batch_size_hint=5)
    inputs, targets, mask = dataset.sample_batch(dataset.make_curriculum(), 5)
    counts = Counter(episode.track for episode in dataset.last_batch_examples)
    assert len(counts) == len(GENERATORS)
    assert max(counts.values()) - min(counts.values()) <= 1
    assert inputs.shape[:2] == targets.shape == mask.shape
    assert inputs.shape[-1] == VOCAB_SIZE
    torch.testing.assert_close(
        inputs.sum(-1), (inputs.sum(-1) > 0).to(inputs.dtype)
    )
    assert set(mask.unique().tolist()) <= {0.0, 1.0}
    for row, episode in enumerate(dataset.last_batch_examples):
        assert mask[row, : len(episode.inputs)].sum() == 0
        assert mask[row, len(episode.inputs) :].sum() == len(episode.answer)

    logits = torch.randn(*targets.shape, VOCAB_SIZE, requires_grad=True)
    with torch.no_grad():
        logits[mask == 0] = float("nan")
    loss = dataset.loss(logits, targets, mask)
    assert torch.isfinite(loss)
    loss.backward()
    assert torch.isfinite(logits.grad[mask.bool()]).all()
    assert torch.equal(logits.grad[~mask.bool()], torch.zeros_like(logits.grad[~mask.bool()]))


def test_task_field_log_rows_match_dataset_owned_header():
    dataset = capability_dataset.WorkspaceCapabilityDataset("entity_state")
    episode = entity_state.generate(
        difficulty_for("entity_state", 3), random.Random(72)
    )
    field_log = {}
    dataset._record_field(
        field_log, episode, correct_count=1, answer_count=1, exact=True
    )
    buffer = io.StringIO()
    writer = csv.writer(buffer)
    dataset.write_field_log(writer, buffer, 500, 3, "ood", field_log)
    row = next(csv.reader(io.StringIO(buffer.getvalue())))

    assert len(row) == 3 + len(dataset.field_log_header())
    assert row[:3] == ["500", "3", "ood"]
    assert row[3] == "entity_state"
    assert row[-4:] == ["1", "1", "1", "1"]


def test_episode_rejects_tokens_outside_typed_vocabulary():
    with pytest.raises(ValueError, match="outside"):
        TaskEpisode("bad", (BOS, VOCAB_SIZE), (EOS,), {"query_type": "invalid"})

