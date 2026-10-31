import csv
import io
import random
from collections import Counter
from types import SimpleNamespace

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
    ENTITY_COUNT,
    EV_COMMIT,
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
from initium.workspace.broadcast import WorkspaceBroadcast
from initium.workspace.nth_farthest import select_nth_farthest


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


def _permutation_initial_entity_ids(episode):
    return {
        segment[1] - ENTITY_BASE
        for segment in _segments(episode)
        if segment[0] == EV_INIT
    }


def test_permutation_id_and_ood_entity_labels_are_disjoint():
    difficulty = difficulty_for("permutation", 5)
    id_episode = permutation.generate(difficulty, random.Random(481))
    ood_episode = permutation.generate(
        difficulty, random.Random(481), ood_axes=("labels",)
    )

    id_labels = _permutation_initial_entity_ids(id_episode)
    ood_labels = _permutation_initial_entity_ids(ood_episode)
    assert len(id_labels) == len(ood_labels) == difficulty["entities"]
    assert max(id_labels) < ENTITY_COUNT - 16
    assert min(ood_labels) >= ENTITY_COUNT - 16
    assert id_labels.isdisjoint(ood_labels)


def test_permutation_label_and_length_shifts_are_independent():
    lesson = 2
    base_difficulty = difficulty_for("permutation", lesson)
    longer_difficulty = difficulty_for(
        "permutation", lesson, ood_axes=("length",)
    )
    assert longer_difficulty["entities"] == base_difficulty["entities"]
    assert longer_difficulty["swaps"] > base_difficulty["swaps"]

    baseline = permutation.generate(base_difficulty, random.Random(902))
    label_only = permutation.generate(
        base_difficulty, random.Random(902), ood_axes=("labels",)
    )
    length_only = permutation.generate(
        longer_difficulty, random.Random(902), ood_axes=("length",)
    )
    assert len(label_only.inputs) == len(baseline.inputs)
    assert label_only.metadata["query_type"] == baseline.metadata["query_type"]
    assert label_only.metadata["relevant_updates"] == baseline.metadata["relevant_updates"]
    assert len(length_only.inputs) > len(baseline.inputs)
    assert max(_permutation_initial_entity_ids(length_only)) < ENTITY_COUNT - 16


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


def test_nth_farthest_exact_readout_uses_descending_distance_and_entity_id_ties():
    vectors = torch.tensor([[[3.0, 0.0], [-3.0, 0.0], [0.0, 1.0], [99.0, 99.0]]])
    query = torch.zeros(1, 2)
    entity_ids = torch.tensor([[9, 2, 5, -1]])
    mask = torch.tensor([[True, True, True, False]])
    selected, distances = select_nth_farthest(
        vectors, query, entity_ids, torch.tensor([0]), mask
    )
    # The two distance-9 candidates tie; entity ID 2 wins the oracle tie-break.
    assert selected.tolist() == [2]
    assert distances.tolist() == [[9.0, 9.0, 1.0, 19602.0]]


def test_workspace_nth_farthest_readout_decodes_addressed_slots_and_tie_breaks():
    workspace = WorkspaceBroadcast(
        d_model=64,
        num_experts=1,
        num_slots=64,
        entity_keyed_writes=True,
        semantic_entity_addressing=True,
        nth_farthest_exact_readout=True,
    )
    with torch.no_grad():
        decoder = workspace.semantic_vector_decoder
        decoder.weight.zero_()
        decoder.bias.zero_()
        for coordinate_value in range(16):
            decoder.weight[coordinate_value, coordinate_value] = 10.0
    state = torch.zeros(1, 64, 128)
    # Entities 7 and 2 are both distance 9, while entity 9 is distance 1.
    for entity, coordinate in ((7, 3), (2, 3), (9, 1)):
        state[0, entity, entity] = 1.0
        state[0, entity, 64 + coordinate] = 1.0
    candidate_ids = torch.tensor([[7, 2, 9] + [-1] * 13])
    candidate_mask = torch.tensor([[True, True, True] + [False] * 13])
    query = torch.tensor([[0, -1, -1, -1, -1, -1]])
    logits, selected, _distances, active = workspace._nth_farthest_query_logits(
        state, candidate_ids, candidate_mask, query, torch.tensor([0])
    )
    assert active.tolist() == [True]
    assert selected.tolist() == [2]
    assert int(logits.argmax(dim=-1).item()) == 2


def test_nth_farthest_workspace_metadata_is_input_only_and_handles_all_track():
    dataset = WorkspaceCapabilityDataset("all", batch_size_hint=2)
    episodes = [
        dataset._make_episode("nth_farthest", 1, random.Random(81)),
        dataset._make_episode("entity_state", 1, random.Random(82)),
    ]
    inputs, _targets, _mask = dataset._collate(episodes)
    metadata = dataset.workspace_nth_farthest_metadata(inputs, episodes)
    assert metadata["nth_farthest_vector_steps"]
    assert metadata["nth_farthest_query_steps"] == (len(episodes[0].inputs),)
    query_step = len(episodes[0].inputs)
    assert int(metadata["nth_farthest_query_candidate_mask"][0, query_step].sum()) == episodes[0].metadata["candidate_count"]
    assert not bool(metadata["nth_farthest_query_candidate_mask"][1].any())
    assert metadata["nth_farthest_query_vectors"][0, query_step].ge(0).any()
    assert metadata["nth_farthest_query_vectors"][1].eq(-1).all()


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


def test_transactional_metadata_defers_swap_pair_and_marks_boundary():
    dataset = capability_dataset.WorkspaceCapabilityDataset("permutation")
    episode = TaskEpisode(
        "permutation",
        (
            EV_INIT, ENTITY_BASE + 2, LOCATION_BASE + 1, SEP,
            EV_INIT, ENTITY_BASE + 5, LOCATION_BASE + 3, SEP,
            EV_SWAP, ENTITY_BASE + 2, ENTITY_BASE + 5,
            EV_COMMIT, ENTITY_BASE + 2, ENTITY_BASE + 5, SEP,
            QUERY, ENTITY_BASE + 2, SEP,
        ),
        (LOCATION_BASE + 3, EOS),
        {
            "query_type": "object_to_position",
            "query_entity_id": 2,
            "query_location_id": 3,
        },
    )
    inputs, _, _ = dataset._collate([episode])
    metadata = dataset.workspace_entity_address_metadata(
        inputs, paired_swap_commits=True
    )
    commit_positions = [12, 13]
    assert not metadata["entity_write_mask"][0, commit_positions].any()
    boundary = 14
    assert metadata["pair_swap_mask"][0, boundary]
    assert metadata["pair_swap_entity_ids"][0, boundary].tolist() == [2, 5]
    assert metadata["pair_swap_steps"] == (boundary,)
    query = dataset.workspace_permutation_query_metadata(inputs, [episode])
    assert query["permutation_query_kind"][0, len(episode.inputs)].item() == 1
    assert query["permutation_query_ids"][0, len(episode.inputs)].item() == 2


def test_all_track_semantic_addresses_write_only_at_supported_event_boundaries():
    dataset = WorkspaceCapabilityDataset("all", batch_size_hint=5)
    episodes = [
        dataset._make_episode(track, 1, random.Random(810 + index))
        for index, track in enumerate(GENERATORS)
    ]
    inputs, _, _ = dataset._collate(episodes)

    metadata = dataset.workspace_entity_address_metadata(
        inputs, paired_swap_commits=True
    )
    state_metadata = dataset.workspace_state_supervision_metadata(inputs, episodes)

    for row, episode in enumerate(episodes):
        write_steps = torch.nonzero(
            metadata["entity_write_mask"][row], as_tuple=False
        ).flatten()
        assert write_steps.numel() > 0, episode.track
        assert metadata["entity_ids"][row, write_steps].ge(0).all()
        query_step = episode.inputs.index(QUERY)
        assert not metadata["entity_write_mask"][row, query_step:].any()
        for step in range(query_step + 1, len(episode.inputs)):
            token = episode.inputs[step]
            if ENTITY_BASE <= token < ENTITY_BASE + ENTITY_COUNT:
                assert metadata["entity_ids"][row, step].item() == token - ENTITY_BASE
                assert not metadata["entity_write_mask"][row, step]
        if episode.track == "permutation":
            assert metadata["pair_swap_mask"][row].any()
            assert state_metadata["state_supervision_mask"][row].any()
        else:
            assert not metadata["pair_swap_mask"][row].any()
            assert not state_metadata["state_supervision_mask"][row].any()


def test_kv_evaluation_microbatches_long_episodes_by_attention_workspace():
    episodes = [
        SimpleNamespace(sequence=tuple(range(length)))
        for length in (4096, 64, 4096, 64)
    ]

    groups = list(
        WorkspaceCapabilityDataset._evaluation_groups(
            episodes, 16, kv_attention_heads=4
        )
    )

    assert [len(group) for group in groups] == [2, 1, 1]
    assert [len(episode.sequence) for group in groups for episode in group] == [
        64, 64, 4096, 4096
    ]


def test_transactional_workspace_swaps_payloads_and_emits_query_readout():
    workspace = WorkspaceBroadcast(
        d_model=64,
        num_experts=1,
        num_slots=16,
        entity_keyed_writes=True,
        semantic_entity_addressing=True,
        permutation_transactional=True,
        diagnostics_enabled=False,
    ).eval()
    pre_commit = workspace.init_state(1, device="cpu", dtype=torch.float32)
    pre_commit[0, 0, 2] = 1.0
    pre_commit[0, 1, 5] = 1.0
    pre_commit[0, 0, 64:] = torch.arange(64, dtype=torch.float32)
    pre_commit[0, 1, 64:] = torch.arange(64, dtype=torch.float32) + 100.0
    exchanged, valid = workspace._apply_semantic_pair_swap(
        pre_commit, torch.tensor([[2, 5]]), torch.tensor([True])
    )
    assert valid.tolist() == [True]
    torch.testing.assert_close(exchanged[0, 0, 64:], pre_commit[0, 1, 64:])
    torch.testing.assert_close(exchanged[0, 1, 64:], pre_commit[0, 0, 64:])
    torch.testing.assert_close(exchanged[0, :, :64], pre_commit[0, :, :64])

    batch, time = 1, 4
    deltas = torch.randn(batch, time, 1, 64, requires_grad=True)
    routes = torch.zeros(batch, time, 1, dtype=torch.long)
    weights = torch.ones(batch, time, 1)
    entity_ids = torch.tensor([[2, 5, 2, 2]])
    write_mask = torch.tensor([[True, True, False, False]])
    pair_ids = torch.tensor([[[-1, -1], [-1, -1], [2, 5], [-1, -1]]])
    pair_mask = torch.tensor([[False, False, True, False]])
    query_kind = torch.tensor([[0, 0, 0, 1]])
    query_ids = torch.tensor([[-1, -1, -1, 2]])
    _, state = workspace(
        deltas,
        routes,
        weights,
        workspace.init_state(batch, device=deltas.device, dtype=deltas.dtype),
        selected_specialists_only=True,
        entity_ids=entity_ids,
        entity_write_mask=write_mask,
        pair_swap_entity_ids=pair_ids,
        pair_swap_mask=pair_mask,
        pair_swap_steps=(2,),
        permutation_query_kind=query_kind,
        permutation_query_ids=query_ids,
        permutation_query_steps=(3,),
    )

    commits = workspace.pop_semantic_pair_commits()
    assert len(commits) == 1
    assert commits[0][0] == 2
    assert commits[0][1].tolist() == [True]
    query_predictions = workspace.pop_semantic_query_predictions()
    assert len(query_predictions) == 1
    assert query_predictions[0][0] == 3
    assert query_predictions[0][1].shape == (1, 16)
    assert query_predictions[0][2].shape == (1, ENTITY_COUNT)
    state_content = state.split(64, dim=-1)[1]
    assert torch.isfinite(state_content).all()


def test_workspace_query_logits_replace_answer_class_and_keep_gradient():
    position_logits = torch.randn(1, 16, requires_grad=True)
    position_logits.data[0, 7] = 10.0
    entity_logits = torch.randn(1, ENTITY_COUNT, requires_grad=True)
    query_predictions = [
        (
            0,
            position_logits,
            entity_logits,
            torch.tensor([1]),
            torch.tensor([2]),
        )
    ]
    logits = torch.zeros(1, 1, VOCAB_SIZE, requires_grad=True)
    mapped = capability_dataset.WorkspaceCapabilityDataset.apply_workspace_permutation_query_logits(
        logits, query_predictions
    )
    assert mapped[0, 0].argmax().item() == LOCATION_BASE + 7
    mapped[0, 0, LOCATION_BASE + 7].backward()
    assert position_logits.grad[0, 7].item() == 1.0

