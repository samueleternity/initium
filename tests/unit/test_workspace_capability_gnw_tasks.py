"""Generator/oracle tests for the GNW probes, conditional_ops, developmental and
production tracks, plus gnw_metrics. Each oracle replays the input stream only."""

import math
from random import Random

import pytest

from initium.data.workspace_capability.curriculum import difficulty_for
from initium.data.workspace_capability.episode import (
    BOOL_FALSE,
    BOOL_TRUE,
    BOS,
    ENTITY_BASE,
    EOS,
    EV_ASSERT,
    EV_INIT,
    EV_MOVE,
    EV_NOISE,
    EV_UPDATE,
    LOCATION_BASE,
    SCALAR_BASE,
    SEP,
    VOCAB_SIZE,
    entity_token,
    scalar_token,
    validate_episode,
)
from initium.data.workspace_capability.gnw_metrics import (
    binary_brier,
    expected_calibration_error,
    psychometric_logistic,
    type2_auc,
    wilson_interval,
)
from initium.data.workspace_capability.tracks import (
    bistable_perception,
    conditional_ops,
    developmental,
    ignition,
    masked_priming,
    no_report_confidence,
    production,
    psychological_refractory,
    sustained_maintenance,
    trace_conditioning,
)

GENERATORS = {
    "ignition": ignition.generate,
    "psychological_refractory": psychological_refractory.generate,
    "masked_priming": masked_priming.generate,
    "sustained_maintenance": sustained_maintenance.generate,
    "no_report_confidence": no_report_confidence.generate,
    "bistable_perception": bistable_perception.generate,
    "trace_conditioning": trace_conditioning.generate,
    "conditional_ops": conditional_ops.generate,
    "developmental": developmental.generate,
    "developmental_production": production.generate,
}


def _segments(tokens):
    out, cur = [], []
    for t in tokens:
        if t == BOS and not cur and not out:
            continue
        if t == SEP:
            if cur:
                out.append(cur)
                cur = []
        else:
            cur.append(t)
    if cur:
        out.append(cur)
    return out


def _values(segs, *events):
    return [s[-1] - SCALAR_BASE for s in segs if s[0] in events]


@pytest.mark.parametrize("track,generator", GENERATORS.items())
def test_generators_are_reproducible_valid_and_ood_bounded(track, generator):
    diff = difficulty_for(track, 3)
    first = validate_episode(generator(diff, Random(177)))
    assert first == generator(diff, Random(177))
    assert first.track == track
    assert first.inputs[0] == BOS and first.answer[-1] == EOS
    assert len(first.sequence) == len(first.targets) == len(first.answer_mask)
    assert not any(first.answer_mask[: len(first.inputs)])
    assert all(first.answer_mask[len(first.inputs) :])
    ood = validate_episode(generator(difficulty_for(track, 5, ood=True), Random(901), ood=True))
    assert ood.track == track
    assert 0 <= min(ood.inputs + ood.answer) and max(ood.inputs + ood.answer) < VOCAB_SIZE


def test_ignition_stream_matches_metadata_and_strength_drives_evidence():
    diff = difficulty_for("ignition", 5)
    hits = n = 0
    for seed in range(600):
        ep = ignition.generate(diff, Random(seed))
        obs = tuple(_values(_segments(ep.inputs), EV_INIT, EV_UPDATE))
        present = ep.metadata["target_present"]
        assert obs == ep.metadata["sensory_observations"]
        assert ep.answer == (BOOL_TRUE if present else BOOL_FALSE, EOS)
        assert ep.metadata["stimulus_strength"] <= 8
        if ep.metadata["stimulus_strength"] == 8:
            n += 1
            hits += (sum(obs) >= 3) == present
    assert n > 0 and hits / n >= 0.9


def test_psychological_refractory_returns_both_items_with_soa_filler():
    diff = difficulty_for("psychological_refractory", 3)
    for seed in range(30):
        ep = psychological_refractory.generate(diff, Random(seed))
        segs = _segments(ep.inputs)
        inits = [s for s in segs if s[0] == EV_INIT]
        updates = [s for s in segs if s[0] == EV_UPDATE]
        assert len(updates) == ep.metadata["soa"] <= 5
        assert ep.answer == (inits[0][-1], inits[1][-1], EOS)


def test_sustained_maintenance_item_survives_non_target_updates():
    diff = difficulty_for("sustained_maintenance", 3)
    for seed in range(30):
        ep = sustained_maintenance.generate(diff, Random(seed))
        segs = _segments(ep.inputs)
        updates = [s for s in segs if s[0] == EV_UPDATE]
        assert len(updates) == ep.metadata["delay_events"]
        assert all(s[1] != entity_token(0) for s in updates)
        assert ep.answer == (segs[0][-1], EOS)


def test_trace_conditioning_answer_is_outcome_bound_to_cue():
    diff = difficulty_for("trace_conditioning", 3)
    for seed in range(40):
        ep = trace_conditioning.generate(diff, Random(seed))
        segs = _segments(ep.inputs)
        cue = segs[0][1]
        updates = [s for s in segs if s[0] == EV_UPDATE]
        final = updates[-1]
        assert final[1] == cue and final[2] in (BOOL_FALSE, BOOL_TRUE)
        assert len(updates) - 1 == ep.metadata["gap_events"]
        assert ep.metadata["trace_condition"] == (ep.metadata["gap_events"] > 0)
        assert ep.answer == (final[2], EOS)


def test_bistable_perception_oracle_replays_hysteresis():
    diff = difficulty_for("bistable_perception", 3)
    for seed in range(60):
        ep = bistable_perception.generate(diff, Random(seed))
        segs = _segments(ep.inputs)
        state = segs[0][-1] - SCALAR_BASE
        cues = [s[-1] - SCALAR_BASE - 1 for s in segs if s[0] == EV_UPDATE]
        assert len(cues) == ep.metadata["frame_count"]
        acc = switches = 0
        for cue in cues:
            acc += cue
            if (acc < 0 if state else acc > 0) and abs(acc) >= 2:
                state, acc, switches = 1 - state, 0, switches + 1
        assert ep.answer == (BOOL_TRUE if state else BOOL_FALSE, EOS)
        assert switches == ep.metadata["dominance_switches"]


def test_masked_priming_streams_are_consistent_with_latent_variables():
    diff = difficulty_for("masked_priming", 3)
    seen = set()
    for seed in range(80):
        ep = masked_priming.generate(diff, Random(seed))
        md, segs = ep.metadata, _segments(ep.inputs)
        masked = segs[0][0] == EV_NOISE
        assert masked == md["mask_condition"]
        assert len(segs[0]) == (2 if masked else 3)
        assert segs[0][-1] - SCALAR_BASE == md["prime_value"]
        assert segs[1] == [EV_NOISE, scalar_token(md["target_strength"])]
        obs = tuple(_values(segs[2:], EV_INIT, EV_UPDATE))
        assert obs == md["target_observations"]
        assert md["prime_congruent"] == (md["latent_prime"] == md["target_value"])
        assert ep.answer == (scalar_token(md["target_value"]), EOS)
        seen.add((masked, md["prime_congruent"]))
    assert len(seen) == 4


def test_no_report_confidence_oracle_uses_majority_and_margin():
    diff = difficulty_for("no_report_confidence", 3)
    reported = set()
    for seed in range(80):
        ep = no_report_confidence.generate(diff, Random(seed))
        segs = _segments(ep.inputs)
        evidence = segs[0][-1] - SCALAR_BASE
        obs = _values(segs, EV_UPDATE)
        report = next(s for s in segs if s[0] == EV_ASSERT)[1] == BOOL_TRUE
        assert len(obs) == evidence
        pos = sum(obs)
        margin = abs(2 * pos - evidence)
        expected = (BOOL_TRUE if pos * 2 >= evidence else BOOL_FALSE,)
        if report:
            expected += (scalar_token(min(3, margin * 4 // evidence)),)
        assert ep.answer == expected + (EOS,)
        assert ep.metadata["vote_margin"] == margin
        assert ep.metadata["explicit_report"] == report
        reported.add(report)
    assert reported == {False, True}


def test_conditional_ops_independent_replay_matches_answer():
    diff = difficulty_for("conditional_ops", 3)
    for seed in range(24):
        ep = conditional_ops.generate(diff, Random(seed))
        records = conditional_ops.parse_records(ep.inputs, len(ep.inputs))
        assert tuple(r[1] for r in records) == ep.metadata["operations"]
        assert records[-1][6][0] == ep.answer[0]


def test_developmental_temporal_family_matches_state_oracle():
    diff = {**difficulty_for("developmental", 3), "families": ("d5_temporal",)}
    for seed in range(20):
        ep = developmental.generate(diff, Random(seed))
        segs = _segments(ep.inputs)
        state = {}
        for s in segs[:-1]:
            if s[0] in (EV_INIT, EV_MOVE):
                state[s[1] - ENTITY_BASE] = s[2] - LOCATION_BASE
        q, kind = segs[-1][1] - ENTITY_BASE, segs[-1][2] - SCALAR_BASE
        assert ep.answer == (LOCATION_BASE + state[(48 if kind else 24) + q], EOS)


def test_production_keyed_outputs_align_with_answer_tokens():
    diff = {**difficulty_for("developmental_production", 2), "groups": ("p_entity",)}
    for seed in range(20):
        ep = production.generate(diff, Random(seed))
        stride = ep.metadata["answer_stride"]
        assert ep.metadata["progression_group"] == "p_entity"
        assert stride >= 1 and (len(ep.answer) - 1) % stride == 0
        assert len(ep.metadata["workspace_output_keys"]) == len(ep.answer)


def test_gnw_metrics_basic_properties():
    lo, hi = wilson_interval(50, 100)
    assert lo < 0.5 < hi and lo + hi == pytest.approx(1.0, abs=1e-9)
    assert all(math.isnan(v) for v in wilson_interval(0, 0))
    assert binary_brier([1.0, 0.0], [1, 0]) == 0.0
    assert expected_calibration_error([1.0, 0.0], [1, 0]) == 0.0
    assert type2_auc([0.9, 0.1], [1, 0]) == 1.0
    assert type2_auc([0.5, 0.5], [1, 0]) == 0.5
    assert math.isnan(type2_auc([0.5, 0.6], [1, 1]))


def test_psychometric_fit_recovers_threshold_and_rejects_bad_criterion():
    counts = {i: (10 + 20 * i, 100) for i in range(5)}
    fit = psychometric_logistic(counts, threshold_probability=0.5)
    assert fit["slope"] > 0
    assert fit["threshold_50"] == pytest.approx(2.0, abs=0.05)
    assert fit["threshold_criterion_in_observed_range"]
    sparse = psychometric_logistic({1: (5, 10)})
    assert math.isnan(sparse["threshold_criterion"]) and sparse["small_sample_warning"]
    with pytest.raises(ValueError):
        psychometric_logistic(counts, threshold_probability=1.0)
