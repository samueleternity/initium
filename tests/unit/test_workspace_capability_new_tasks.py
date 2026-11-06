"""Essential generator and workspace-readout tests for the six newer tasks."""

import random

import pytest
import torch

from initium.data.workspace_capability.curriculum import difficulty_for
from initium.data.workspace_capability.dataset import WorkspaceCapabilityDataset
from initium.data.workspace_capability.episode import (
    BOOL_FALSE,
    BOOL_TRUE,
    BOS,
    ENTITY_BASE,
    EOS,
    EV_ASSERT,
    EV_INIT,
    EV_KV,
    EV_MOVE,
    EV_NOISE,
    EV_UPDATE,
    EV_VECTOR,
    LOCATION_BASE,
    QUERY,
    SCALAR_BASE,
    SCALAR_COUNT,
    SOURCE_BASE,
    SEP,
    VOCAB_SIZE,
    location_token,
    scalar_token,
    source_token,
    validate_episode,
)
from initium.data.workspace_capability.tracks import (
    attentional_blink,
    belief_tracking,
    capacity_bottleneck,
    mixed_skill,
    rule_switching,
    variable_binding,
)


NEW_GENERATORS = {
    "capacity_bottleneck": capacity_bottleneck.generate,
    "belief_tracking": belief_tracking.generate,
    "mixed_skill": mixed_skill.generate,
    "rule_switching": rule_switching.generate,
    "attentional_blink": attentional_blink.generate,
    "variable_binding": variable_binding.generate,
}


def _segments(tokens):
    """Split a task stream into SEP-delimited events, omitting BOS/padding."""
    segments = []
    current = []
    for token in tokens:
        token = int(token)
        if token == 0:  # PAD in batched input rows.
            continue
        if token == BOS and not current and not segments:
            continue
        if token == SEP:
            if current:
                segments.append(current)
                current = []
        else:
            current.append(token)
    if current:
        segments.append(current)
    return segments


@pytest.mark.parametrize("track,generator", NEW_GENERATORS.items())
def test_new_track_generators_are_reproducible_valid_and_ood_bounded(track, generator):
    difficulty = difficulty_for(track, 3)
    episode = validate_episode(generator(difficulty, random.Random(177)))
    repeated = validate_episode(generator(difficulty, random.Random(177)))
    ood_episode = validate_episode(
        generator(difficulty_for(track, 5, ood=True), random.Random(901), ood=True)
    )

    assert episode == repeated
    assert episode.track == track
    assert episode.inputs[0] == BOS
    assert episode.answer[-1] == EOS
    assert max(episode.inputs + episode.answer) < VOCAB_SIZE
    assert len(episode.sequence) == len(episode.targets) == len(episode.answer_mask)
    assert not any(episode.answer_mask[: len(episode.inputs)])
    assert all(episode.answer_mask[len(episode.inputs) :])
    assert ood_episode.track == track
    assert max(ood_episode.inputs + ood_episode.answer) < VOCAB_SIZE


def test_new_task_curriculum_requires_each_task_specific_challenge_group():
    dataset = WorkspaceCapabilityDataset("new_all")

    assert dataset._curriculum_groups("capacity_bottleneck", 5) == (
        "change_detection",
        "multiple_object_tracking",
        "n_back",
    )
    assert dataset._curriculum_groups("belief_tracking", 5) == (
        "false_belief",
        "true_belief",
        "reality",
    )
    assert dataset._curriculum_groups("mixed_skill", 3) == ("dependent_ops_6",)
    assert dataset._curriculum_groups("rule_switching", 5) == (
        "cued_switch",
        "feedback_inferred_switch",
        "mixed_cue_switch",
    )
    assert dataset._curriculum_groups("attentional_blink", 5) == tuple(
        f"lag_{lag}_{condition}"
        for lag in range(1, 9)
        for condition in ("unmasked", "masked")
    )
    assert dataset._curriculum_groups("variable_binding", 5) == ("computed_chain_long",)


def test_capacity_generator_covers_all_three_paradigms_and_oracles():
    difficulty = difficulty_for("capacity_bottleneck", 5)
    modes = set()
    for seed in range(96):
        episode = capacity_bottleneck.generate(difficulty, random.Random(seed))
        mode = episode.metadata["progression_group"]
        modes.add(mode)
        segments = _segments(episode.inputs)

        if mode == "change_detection":
            initial, current = {}, {}
            for segment in segments:
                if segment[0] == EV_INIT:
                    initial[segment[1] - ENTITY_BASE] = segment[2] - LOCATION_BASE
                    current[segment[1] - ENTITY_BASE] = segment[2] - LOCATION_BASE
                elif segment[0] == EV_UPDATE:
                    current[segment[1] - ENTITY_BASE] = segment[2] - LOCATION_BASE
            probe = episode.metadata["probe_item"]
            assert episode.answer[0] == (
                BOOL_TRUE if initial[probe] != current[probe] else BOOL_FALSE
            )
        elif mode == "multiple_object_tracking":
            positions, tracked = {}, set()
            for segment in segments:
                if segment[0] == EV_INIT:
                    entity = segment[1] - ENTITY_BASE
                    positions[entity] = segment[2] - LOCATION_BASE
                    if segment[3] == BOOL_TRUE:
                        tracked.add(entity)
                elif segment[0] == EV_MOVE:
                    positions[segment[1] - ENTITY_BASE] = segment[2] - LOCATION_BASE
            probe = episode.metadata["probe_item"]
            assert probe in tracked
            assert episode.answer == (location_token(positions[probe]), EOS)
        else:
            symbols = [
                token - LOCATION_BASE
                for token in episode.inputs
                if LOCATION_BASE <= token < LOCATION_BASE + 16
            ]
            n = episode.metadata["n_back"]
            assert len(symbols) == episode.metadata["stream_length"]
            assert episode.answer[0] == (
                BOOL_TRUE if symbols[-1] == symbols[-1 - n] else BOOL_FALSE
            )
            assert episode.metadata["workspace_output_programs"][0][1] == "equal"

    assert modes == {"change_detection", "multiple_object_tracking", "n_back"}


def test_belief_tracking_oracle_keeps_each_agents_partial_state():
    for seed in range(24):
        episode = belief_tracking.generate(
            difficulty_for("belief_tracking", 5), random.Random(seed)
        )
        world, beliefs = {}, {}
        query = None
        for segment in _segments(episode.inputs):
            event = segment[0]
            if event == EV_INIT:
                world[segment[1] - ENTITY_BASE] = segment[2] - LOCATION_BASE
            elif event == EV_MOVE:
                world[segment[1] - ENTITY_BASE] = segment[2] - LOCATION_BASE
            elif event == EV_ASSERT:
                agent = segment[1] - SOURCE_BASE
                entity = segment[2] - ENTITY_BASE
                beliefs[agent, entity] = segment[3] - LOCATION_BASE
            elif event == QUERY:
                query = segment[1:]

        assert query is not None
        if len(query) == 1:
            expected = world[query[0] - ENTITY_BASE]
            assert episode.metadata["progression_group"] == "reality"
        else:
            agent, entity = query[0] - SOURCE_BASE, query[1] - ENTITY_BASE
            expected = beliefs[agent, entity]
            if episode.metadata["progression_group"] == "false_belief":
                assert episode.metadata["false_belief"] == (expected != world[entity])
            elif episode.metadata["progression_group"] == "true_belief":
                assert expected == world[entity]
        assert episode.answer == (location_token(expected), EOS)


def test_mixed_skill_oracle_executes_lookup_copy_add_and_two_output_sort():
    episode = mixed_skill.generate(
        {"operations": 8, "keys": 10}, random.Random(91), ood=True
    )
    table, stage_values, flattened = {}, {}, []
    skill_counts = {"lookup": 0, "arithmetic": 0, "copy": 0, "sort": 0}
    for segment in _segments(episode.inputs):
        if segment[0] == EV_KV:
            table[segment[1] - ENTITY_BASE] = segment[2] - SCALAR_BASE
        elif segment[0] == EV_VECTOR:
            destination = segment[1] - ENTITY_BASE
            skill = segment[2] - SOURCE_BASE
            if skill == 0:
                values = (table[segment[3] - ENTITY_BASE],)
                skill_counts["lookup"] += 1
            elif skill == 1:
                left = stage_values[segment[3] - ENTITY_BASE][0]
                right = segment[4] - SCALAR_BASE
                values = ((left + right) % SCALAR_COUNT,)
                skill_counts["arithmetic"] += 1
            elif skill == 2:
                values = stage_values[segment[3] - ENTITY_BASE]
                skill_counts["copy"] += 1
            else:
                left = stage_values[segment[3] - ENTITY_BASE][0]
                right = segment[4] - SCALAR_BASE
                values = tuple(sorted((left, right)))
                skill_counts["sort"] += 1
            stage_values[destination] = values
            flattened.extend(values)

    assert all(count > 0 for count in skill_counts.values())
    assert episode.metadata["outputs_are_not_in_input"]
    assert episode.metadata["workspace_operation_output_keys"]
    assert episode.answer == tuple(scalar_token(value) for value in flattened) + (EOS,)


def test_rule_switching_cues_feedback_and_final_selector_labels_are_consistent():
    groups = set()
    for seed in range(96):
        episode = rule_switching.generate(
            {"evidence_trials": 12, "switches": 4, "feedback_noise": 0.3},
            random.Random(seed),
        )
        groups.add(episode.metadata["progression_group"])
        assert len(episode.metadata["trial_rules"]) == episode.metadata["evidence_trials"]
        assert len(episode.metadata["switch_boundaries"]) == episode.metadata["switch_count"]
        assert episode.metadata["cue_count"] <= episode.metadata["switch_count"]

        final_card = next(
            segment
            for segment in reversed(_segments(episode.inputs))
            if segment[0] == EV_VECTOR
        )
        final_values = tuple(value - SCALAR_BASE for value in final_card[2:5])
        rule = episode.metadata["final_rule"]
        assert episode.answer == (scalar_token(final_values[rule]), EOS)
        assert episode.metadata["workspace_query_reads"] == (
            (len(episode.inputs) - 1, 63, source_token(rule)),
        )
        assert episode.metadata["workspace_output_programs"][0][1:] == (
            "select",
            (63, 0, 1, 2),
            (source_token(rule), *(scalar_token(value) for value in final_values)),
        )
    assert groups == {"cued_switch", "feedback_inferred_switch", "mixed_cue_switch"}


def test_attentional_blink_has_exactly_two_conjunction_targets_at_declared_lag():
    episode = attentional_blink.generate(
        {"maximum_lag": 8, "distractors": 14, "lure_probability": 0.85},
        random.Random(44),
        ood=True,
    )
    dataset = WorkspaceCapabilityDataset("attentional_blink")
    records = dataset._attentional_item_frame_records(list(episode.inputs))
    inputs, _, _ = dataset._collate([episode])
    cue_metadata = dataset.workspace_attentional_cue_metadata(inputs, episodes=[episode])
    color = SOURCE_BASE + episode.metadata["target_color"]
    shape = SOURCE_BASE + episode.metadata["target_shape"]
    targets = [record for record in records if record[2] == color and record[3] == shape]

    assert len(targets) == 2
    assert targets[1][1] - targets[0][1] == episode.metadata["target_lag"]
    assert tuple(record[4] for record in targets) == (
        SCALAR_BASE + episode.metadata["target_one"],
        SCALAR_BASE + episode.metadata["target_two"],
    )
    assert len(records) == episode.metadata["distractor_count"] + 2
    assert episode.answer == (
        scalar_token(episode.metadata["target_one"]),
        scalar_token(episode.metadata["target_two"]),
        EOS,
    )
    assert cue_metadata["attentional_cue_color_ids"].item() == source_token(
        episode.metadata["target_color"]
    )
    assert cue_metadata["attentional_cue_shape_ids"].item() == source_token(
        episode.metadata["target_shape"]
    )
    assert cue_metadata["attentional_cue_query_step_ids"].item() == len(episode.inputs)
    assert cue_metadata["attentional_frame_mask"].sum().item() == len(records)
    if episode.metadata["mask_condition"]:
        assert episode.inputs[targets[1][0] + 1] == EV_NOISE
        assert episode.metadata["mask_is_post_t2_noise_frame"]


def test_variable_binding_oracle_follows_copies_reassignments_and_modular_adds():
    episode = variable_binding.generate(
        {"variables": 10, "assignments": 21}, random.Random(108), ood=True
    )
    values, literal_values = {}, set()
    query_variable = None
    operation_count = 0
    for segment in _segments(episode.inputs):
        if segment[0] == EV_INIT:
            variable = segment[1] - ENTITY_BASE
            value = segment[2] - SCALAR_BASE
            values[variable] = value
            literal_values.add(value)
        elif segment[0] == EV_KV:
            destination, source = segment[1] - ENTITY_BASE, segment[2] - ENTITY_BASE
            values[destination] = values[source]
            operation_count += 1
        elif segment[0] == EV_UPDATE:
            destination, value = segment[1] - ENTITY_BASE, segment[2] - SCALAR_BASE
            values[destination] = value
            literal_values.add(value)
            operation_count += 1
        elif segment[0] == EV_VECTOR:
            destination = segment[1] - ENTITY_BASE
            left = values[segment[3] - ENTITY_BASE]
            right = values[segment[4] - ENTITY_BASE]
            values[destination] = (left + right) % SCALAR_COUNT
            operation_count += 1
        elif segment[0] == QUERY:
            query_variable = segment[1] - ENTITY_BASE

    assert query_variable is not None
    expected = values[query_variable]
    assert episode.answer == (scalar_token(expected), EOS)
    assert expected not in literal_values
    assert operation_count == episode.metadata["assignment_count"]
    assert episode.metadata["copy_count"] >= 3
    assert episode.metadata["chain_hops"] >= 4
    assert episode.metadata["answer_is_literal"] is False


def test_new_all_collates_all_six_tracks_and_builds_causal_workspace_metadata():
    dataset = WorkspaceCapabilityDataset("new_all", batch_size_hint=len(NEW_GENERATORS))
    assert dataset._tracks() == tuple(NEW_GENERATORS)
    episodes = [
        dataset._make_episode(track, 3, random.Random(500 + index))
        for index, track in enumerate(NEW_GENERATORS)
    ]
    inputs, _targets, _mask = dataset._collate(episodes)
    entity_metadata = dataset.workspace_entity_address_metadata(
        inputs, episodes=episodes, attentional_item_memory=True
    )
    kv_metadata = dataset.workspace_key_value_metadata(
        inputs,
        episodes,
        include_operations=True,
        include_operation_diagnostics=True,
        attentional_item_memory=True,
    )
    cue_metadata = dataset.workspace_attentional_cue_metadata(inputs, episodes=episodes)

    assert inputs.shape[0] == len(NEW_GENERATORS)
    assert entity_metadata["entity_ids"].shape == inputs.shape[:2]
    assert entity_metadata["entity_write_mask"].shape == inputs.shape[:2]
    assert kv_metadata["key_value_targets"].shape[:2] == inputs.shape[:2]
    assert kv_metadata["key_value_mask"].shape == kv_metadata["key_value_targets"].shape
    assert kv_metadata["workspace_operation_metadata"].shape == (*inputs.shape[:2], 7)
    assert kv_metadata["workspace_operation_diagnostic_targets"].shape == (
        *inputs.shape[:2],
        4,
    )

    for row, episode in enumerate(episodes):
        query_step = episode.inputs.index(QUERY)
        assert not entity_metadata["entity_write_mask"][row, query_step:].any()
        written = entity_metadata["entity_ids"][row][
            entity_metadata["entity_write_mask"][row]
        ]
        assert written.numel() > 0, episode.track
        assert written.ge(0).all() and written.lt(64).all()
        active_targets = kv_metadata["key_value_targets"][row][
            kv_metadata["key_value_mask"][row]
        ]
        assert active_targets.ge(0).all() and active_targets.lt(VOCAB_SIZE).all()
        for step, key, _value in episode.metadata.get("workspace_query_reads", ()):
            assert step == len(episode.inputs) - 1
            read_keys = kv_metadata["key_value_entity_ids"][row, step][
                kv_metadata["key_value_mask"][row, step]
            ]
            assert read_keys.eq(key).any()

    assert kv_metadata["attentional_frame_color_targets"].ge(0).any()
    assert kv_metadata["attentional_frame_shape_targets"].ge(0).any()
    blink_row = next(
        row for row, episode in enumerate(episodes) if episode.track == "attentional_blink"
    )
    blink_episode = episodes[blink_row]
    assert cue_metadata["attentional_cue_color_ids"][blink_row].item() == source_token(
        blink_episode.metadata["target_color"]
    )
    assert cue_metadata["attentional_cue_shape_ids"][blink_row].item() == source_token(
        blink_episode.metadata["target_shape"]
    )
    assert kv_metadata["workspace_operation_steps"]
    operation_codes = kv_metadata["workspace_operation_metadata"]
    mixed_row = next(
        row for row, episode in enumerate(episodes) if episode.track == "mixed_skill"
    )
    mixed_steps = kv_metadata["workspace_operation_steps"]
    mixed_codes = operation_codes[mixed_row, list(mixed_steps), 0]
    assert mixed_codes.eq(4).any()  # sort has two separately committed outputs.
    sort_rows = operation_codes[mixed_row, list(mixed_steps)][mixed_codes.eq(4)]
    assert torch.equal(sort_rows[:, 2], sort_rows[:, 1] + 1)
    variable_row = next(
        row for row, episode in enumerate(episodes) if episode.track == "variable_binding"
    )
    assert operation_codes[variable_row, :, 0].eq(3).any()  # modular addition.


@pytest.mark.parametrize(
    "operation,left,right,expected",
    [
        ("equal", LOCATION_BASE + 5, LOCATION_BASE + 5, BOOL_TRUE),
        ("equal", LOCATION_BASE + 5, LOCATION_BASE + 9, BOOL_FALSE),
        ("not_equal", LOCATION_BASE + 5, LOCATION_BASE + 5, BOOL_FALSE),
        ("not_equal", LOCATION_BASE + 5, LOCATION_BASE + 9, BOOL_TRUE),
    ],
)
def test_exact_workspace_comparison_readout_is_correct_and_differentiable(
    operation, left, right, expected
):
    logits = torch.zeros(1, 1, VOCAB_SIZE)
    values = torch.zeros(1, 2, VOCAB_SIZE, requires_grad=True)
    with torch.no_grad():
        values[0, 0, left] = 5.0
        values[0, 1, right] = 5.0
    targets = torch.tensor([[[left, right, -100, -100]]])
    metadata = {
        "key_value_mask": torch.tensor([[[True, True, False, False]]]),
        "key_value_targets": targets,
        "key_value_output_programs": ((0, 0, operation, 2),),
    }
    answer_mask = torch.ones(1, 1, dtype=torch.bool)

    result = WorkspaceCapabilityDataset.apply_workspace_key_value_readout_logits(
        logits,
        [(0, values)],
        metadata,
        answer_mask,
        exact_value_comparison=True,
    )
    assert result[0, 0].argmax().item() == expected
    (-result[0, 0, expected]).backward()
    assert values.grad is not None and torch.isfinite(values.grad).all()
    assert values.grad.abs().sum() > 0


def test_exact_workspace_selection_uses_rule_then_value_and_keeps_soft_gradient():
    logits = torch.zeros(1, 1, VOCAB_SIZE)
    values = torch.zeros(1, 4, VOCAB_SIZE, requires_grad=True)
    expected_value = scalar_token(12)
    with torch.no_grad():
        values[0, 0, source_token(2)] = 6.0
        values[0, 1, scalar_token(3)] = 6.0
        values[0, 2, scalar_token(7)] = 6.0
        values[0, 3, expected_value] = 6.0
    metadata = {
        "key_value_mask": torch.ones(1, 1, 4, dtype=torch.bool),
        "key_value_targets": torch.tensor(
            [[[source_token(2), scalar_token(3), scalar_token(7), expected_value]]]
        ),
        "key_value_output_programs": ((0, 0, "select", 4),),
    }

    result = WorkspaceCapabilityDataset.apply_workspace_key_value_readout_logits(
        logits,
        [(0, values)],
        metadata,
        torch.ones(1, 1, dtype=torch.bool),
        exact_selection=True,
    )
    assert result[0, 0].argmax().item() == expected_value
    (-result[0, 0, expected_value]).backward()
    assert values.grad is not None and torch.isfinite(values.grad).all()
    assert values.grad.abs().sum() > 0


def test_attentional_cue_readout_routes_only_active_rows_to_scalar_classes():
    logits = torch.zeros(2, 2, VOCAB_SIZE)
    logits[1, 0, 17] = 4.0
    first = torch.zeros(2, SCALAR_COUNT)
    second = torch.zeros(2, SCALAR_COUNT)
    first[0, 4] = 3.0
    second[0, 11] = 3.0

    result = WorkspaceCapabilityDataset.apply_workspace_attentional_cue_readout_logits(
        logits,
        [(0, first, second, torch.tensor([True, False]))],
    )

    assert result[0, 0].argmax().item() == scalar_token(4)
    assert result[0, 1].argmax().item() == scalar_token(11)
    assert result[1, 0].argmax().item() == 17
