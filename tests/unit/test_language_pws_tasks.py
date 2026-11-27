"""Generator and curriculum contracts for PWS-oriented language induction and logic."""

import random
from types import SimpleNamespace

from initium.data.workspace_capability.curriculum import difficulty_for
from initium.data.workspace_capability.episode import (
    ENTITY_BASE,
    MARK_O,
    MARK_R,
    MARK_S,
    TEXT_BASE,
    validate_episode,
)
from initium.data.workspace_capability.lang_dataset import LanguageDataset
from initium.data.workspace_capability.tracks import _lang_grammar, _lang_text, lang5_tracks


class _TestTokenizer:
    """Small deterministic tokenizer so generator tests need no corpus files."""

    def __init__(self):
        self.ids = {}

    def word(self, word):
        if word not in self.ids:
            self.ids[word] = len(self.ids)
        return (TEXT_BASE + self.ids[word],)


def _use_test_tokenizer(monkeypatch):
    tokenizer = _TestTokenizer()
    monkeypatch.setattr(_lang_text, "tokenizer", lambda: tokenizer)
    # PWS tasks do not require corpus annotation, but pseudo-word sampling
    # checks the grammar surface vocabulary. Keep these unit tests independent
    # of generated Stage 3/4 corpus artifacts.
    monkeypatch.setattr(_lang_grammar, "grammar", lambda: SimpleNamespace(surf={}))
    return tokenizer


def test_induce_pws_curriculum_progresses_from_contrastive_entity_bindings(monkeypatch):
    _use_test_tokenizer(monkeypatch)
    expected_stages = (0, 1, 2, 3, 3, 3)
    observed_entity_targets = set()

    for level, expected_stage in enumerate(expected_stages):
        difficulty = difficulty_for("lang_induce_pws", level)
        assert difficulty["binding_stage"] == expected_stage
        seeds = range(32) if expected_stage == 0 else (1_000 + level,)
        for seed in seeds:
            episode = validate_episode(lang5_tracks._g_induce_pws(difficulty, random.Random(seed)))
            assert episode.track == "lang_induce_pws"
            if expected_stage < 3:
                assert episode.metadata["binding_stage"] == expected_stage
            assert "workspace_output_keys" not in episode.metadata
            assert not episode.metadata.get("workspace_query_reads")

            answer_kinds = episode.metadata["induction_answer_kinds"]
            answer_tokens = episode.answer[:-1]  # EOS is appended by the episode builder.
            if expected_stage <= 1:
                assert answer_kinds == ("entity",)
                assert len(answer_tokens) == 1
            elif expected_stage == 2:
                assert len(answer_kinds) == len(answer_tokens) == 1
                assert answer_kinds[0] in {"entity", "relation"}
            else:
                assert len(answer_kinds) == len(answer_tokens)
                assert set(answer_kinds) <= {"entity", "relation", "other"}

            if expected_stage == 0:
                observed_entity_targets.add(episode.metadata["induction_eval_entity_target"])
                assert (
                    answer_tokens[0]
                    == ENTITY_BASE + episode.metadata["induction_eval_entity_target"]
                )

    assert observed_entity_targets <= {0, 1}
    assert observed_entity_targets == {0, 1}


def test_induce_pws_keeps_demonstrations_in_the_same_sequence(monkeypatch):
    _use_test_tokenizer(monkeypatch)
    episode = lang5_tracks._g_induce_pws(difficulty_for("lang_induce_pws", 0), random.Random(41))

    split = LanguageDataset._split_context_query(episode)

    assert split == episode
    assert not split.metadata.get("prev")


def test_logic_pws_curriculum_expands_proof_groups_and_emits_structured_traces(monkeypatch):
    _use_test_tokenizer(monkeypatch)
    previous_groups = set()
    observed_groups = set()

    for level in range(6):
        difficulty = difficulty_for("lang_logic_pws", level)
        groups = set(difficulty["groups"])
        assert previous_groups <= groups
        previous_groups = groups
        for seed in range(40):
            episode = validate_episode(
                lang5_tracks._g_logic_pws(difficulty, random.Random(level * 100 + seed))
            )
            observed_groups.add(episode.metadata["progression_group"])
            assert episode.track == "lang_logic_pws"
            assert episode.metadata["answer_kind"] == "mr"
            assert episode.metadata["logic_proof_trace"] is True
            assert episode.metadata["proof_start"] == episode.metadata["verdict_len"]
            assert sum(episode.metadata["proof_step_token_lengths"]) == (
                len(episode.answer) - episode.metadata["proof_start"] - 1
            )
            assert all(length > 0 for length in episode.metadata["proof_step_token_lengths"])
            assert {MARK_S, MARK_R, MARK_O} <= set(episode.inputs)
            assert not episode.metadata.get("workspace_output_keys")
            assert "workspace_output_kv_targets" not in episode.metadata
            assert LanguageDataset._split_context_query(episode) == episode

    assert observed_groups == {
        "l_chain",
        "l_neg",
        "l_unknown",
        "l_contra",
        "l_count",
        "l_some",
        "l_cond",
    }
