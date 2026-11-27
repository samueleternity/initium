"""Stage 4 workspace-grounded language dataset (token-ID only, V_total = TEXT_BASE + text pieces).

Read keys / write masks / slot targets come from episode metadata produced by the text-only
annotator in _lang_grammar.Ep; nothing here inspects answer tokens.
"""

from __future__ import annotations

import contextlib
import io
import random
from collections import defaultdict

import torch

from initium.data.workspace_capability.curriculum import difficulty_for
from initium.data.workspace_capability.dataset import (
    KEY_VALUE_OPERAND_CAPACITY,
    WorkspaceCapabilityDataset,
)
from initium.data.workspace_capability.episode import (
    BOOL_FALSE,
    BOOL_TRUE,
    BOS,
    ENTITY_BASE,
    ENTITY_COUNT,
    EOS,
    FACT_BASE,
    FACT_COUNT,
    RELATION_BASE,
    RELATION_COUNT,
    SEP,
    TASK_PARA,
    TASK_QA,
    TASK_RETELL,
    TASK_T2D,
    TEXT_BASE,
    TaskEpisode,
    validate_episode,
)
from initium.data.workspace_capability.tracks import _dev_lexicon
from initium.data.workspace_capability.tracks import _lang_grammar as lg
from initium.data.workspace_capability.tracks import _lang_text as lt
from initium.data.workspace_capability.tracks import lang5_tracks as l5
from initium.data.workspace_capability.tracks import lang_tracks as ltr

_GEN = {
    "lang_template": ltr._g_template,
    "lang_nlg": ltr._g_nlg,
    "lang_story_qa": ltr._g_story,
    "lang_dialogue": ltr._g_dialogue,
    "lang_syntax_probes": ltr._g_probe,
    "lang_open": ltr._g_open,
}
_RETRIES = 64
_GEN.update(l5.GENERATORS)
_SLOT_RANGES5 = ((48, 80, 0), (80, 112, 1), (112, 128, 2))
_DUMMY = TaskEpisode("lang_pad", (BOS, SEP), (EOS,), {"query_type": "pad"})
_EXTRA_FUNC = (
    "ok",
    "hello",
    "goodbye",
    "welcome",
    "mean",
    "carrying",
    "nothing",
    "everything",
    "consistent",
    "zero",
    "one",
    "two",
    "three",
    "four",
    "five",
    "in",
    "is",
    "yes,",
    "no,",
    "ok,",
)
_SLOT_KINDS = ("subject", "object", "relation", "word")
_SLOT_RANGES = ((24, 32, 0), (32, 40, 1), (48, 56, 2))


def _contains(seq, sub):
    n = len(sub)
    return n > 0 and any(seq[i : i + n] == sub for i in range(len(seq) - n + 1))


class LanguageDataset(WorkspaceCapabilityDataset):
    """Language tracks, the Stage 5 subset, and the full lang_all mix."""

    def __init__(
        self,
        task_track: str = "lang_all",
        *,
        batch_size_hint: int = 16,
        length_bucket_microbatches: int = 1,
    ):
        super().__init__(task_track, batch_size_hint=batch_size_hint)
        self.length_bucket_microbatches = max(1, int(length_bucket_microbatches))
        self.input_dim = self.output_dim = lt.tokenizer().total
        self.diag_episodes = 48  # episodes per slot-probe / surgery / paraphrase diagnostic
        # Generation re-forwards the model once per output token. Keep this
        # diagnostic small enough to avoid dominating each curriculum eval.
        self.generation_eval_per_group = 8
        self.generation_max_tokens = 64
        self._lang_ready = True
        self._weighted_sampling = True
        self._seg_queue: list = []
        self.segment_continue = False
        self._track_exact_history: dict[str, list[float]] = {}
        self._track_optimizer_step_counts: dict[str, int] = defaultdict(int)
        self._track_sampled_episode_counts: dict[str, int] = defaultdict(int)
        self._track_sampled_batch_counts: dict[str, int] = defaultdict(int)
        self._track_group_sampling_weights: dict[str, dict[str, float]] = {}
        self._pending_chain_tracks: set[str] = set()
        self.segment_chain_start = True
        self.segment_chain_complete = True
        self.segment_chain_steps = 1
        self.segment_loss_weight = 1.0

    @staticmethod
    def slot_text_columns():
        if lt.settings["stage5"]:
            tk = lt.tokenizer()
            skip = {
                t for w in (*lg.FUNCTION, *lg.PUNCT, *_EXTRA_FUNC, "|", ";") for t in tk.word(w)
            }
            # Stage-5 slot values store a word's first SentencePiece only. Keep
            # continuation pieces on the backbone path so the slot head is not
            # asked to hallucinate pieces that were never written to memory.
            word_starts = {
                TEXT_BASE + piece_id
                for piece_id in range(tk.text_size)
                if tk.sp.id_to_piece(piece_id).startswith("▁")
            }
            return sorted(
                (word_starts - skip)
                | set(range(ENTITY_BASE, ENTITY_BASE + ENTITY_COUNT))
                | set(range(RELATION_BASE, RELATION_BASE + RELATION_COUNT))
                | {BOOL_FALSE, BOOL_TRUE, l5.NEG_TOK, l5.PLU_TOK}
            )
        G, tk = lg.grammar(), lt.tokenizer()
        skip = {tk.word(w)[0] for w in (*lg.FUNCTION, *lg.NAMES, *lg.OBJS, *lg.PUNCT)}
        words = {tk.word(G.word[i])[0] for i in G.ok} | {tk.word(w)[0] for w in lg.PLACES}
        return sorted((words - skip) | {BOOL_FALSE, BOOL_TRUE})

    @property
    def token_id_input(self):
        return True

    @token_id_input.setter
    def token_id_input(self, value):
        if not value and getattr(self, "_lang_ready", False):
            raise ValueError("language tracks require --workspace-token-id-input")

    # ------------------------------------------------------------------ tracks / episodes
    def _tracks(self):
        if self.task_track == "lang_all":
            return lt.available_tracks()
        if self.task_track == lt.LANG_V5_SELECTOR:
            available = set(lt.available_tracks())
            return tuple(track for track in lt.LANG_V5_TRACKS if track in available)
        return (self.task_track,)

    @staticmethod
    def _curriculum_groups(track, lesson):
        return tuple(difficulty_for(track, lesson)["groups"])

    def _make_episode(
        self, track, level, rng, *, ood=False, ood_axes=None, generalization_axis=None
    ):
        track_level = level
        if track in ("lang_all", lt.LANG_V5_SELECTOR):
            pool = self._tracks()
            if not pool:
                raise RuntimeError(f"{track}: no language tracks are available in this corpus")
            track = pool[min(int(rng.random() * len(pool)), len(pool) - 1)]
        if isinstance(level, dict):
            track_level = int(level.get(track, 0))
        cfg = difficulty_for(track, track_level, ood=ood, ood_axes=ood_axes)
        group_weights = self._track_group_sampling_weights.get(track)
        if group_weights:
            cfg["_group_sampling_weights"] = group_weights
        generate = _GEN[track]
        for _ in range(_RETRIES):
            try:
                episode = validate_episode(generate(cfg, rng))
                # Preserve the exact difficulty level used to generate this
                # episode. Curriculum promotion happens after evaluation, so
                # the live lesson map may already differ by the time metrics
                # are written or paired ablations run.
                episode.metadata["curriculum_lesson"] = int(track_level)
                return self._split_context_query(episode)
            except OverflowError:  # generator-signalled resample (length/feasibility)
                continue
        raise RuntimeError(
            f"{track}: no valid episode after {_RETRIES} attempts at level {track_level}"
        )

    @staticmethod
    def _split_context_query(episode):
        """Split selected tasks to train context-only workspace carry."""
        markers = {
            "lang_template": {TASK_QA, TASK_RETELL, TASK_PARA},
            "lang_nlg": {TASK_RETELL, TASK_T2D},
            "lang_parse": {l5.TASK_MR},
            "lang_define": {TASK_QA},
            "lang_induce": {l5.TASK_S2MR, l5.TASK_MR2S},
            "lang_logic": {l5.TASK_LOGIC},
            lt.LANG_PWS_LOGIC: {l5.TASK_LOGIC},
        }.get(episode.track)
        if not markers or episode.metadata.get("prev"):
            return episode
        prefix_len = 3
        cut = next(
            (
                i
                for i, token in enumerate(episode.inputs[prefix_len:], prefix_len)
                if token in markers
            ),
            None,
        )
        if cut is None or cut <= prefix_len or cut >= len(episode.inputs) - 1:
            return episode
        prefix = episode.inputs[:prefix_len]
        context_inputs = episode.inputs[:cut]
        query_inputs = prefix + episode.inputs[cut:]
        context_meta, query_meta = dict(episode.metadata), dict(episode.metadata)
        position_fields = (
            "lang_addr",
            "workspace_event_writes",
            "workspace_query_reads",
            "workspace_match_writes",
            "workspace_binding_aux",
            "language_induction_binding_aux",
        )
        for field in position_fields:
            vals = episode.metadata.get(field, ())
            left, right = [], []
            for row in vals:
                pos = int(row[0])
                if pos < cut:
                    left.append(row)
                else:
                    mapped = (prefix_len + pos - cut, *row[1:])
                    right.append(mapped)
            context_meta[field], query_meta[field] = tuple(left), tuple(right)
        context_meta.update(
            context_only=True,
            answer_count=0,
            workspace_output_keys=(),
            workspace_output_kv_targets=None,
        )
        query_meta["prev"] = (
            TaskEpisode(episode.track, tuple(context_inputs), (EOS,), context_meta),
        )
        query_meta["sequence_length"] = len(query_inputs)
        return validate_episode(
            TaskEpisode(episode.track, tuple(query_inputs), episode.answer, query_meta)
        )

    # ------------------------------------------------------------------ workspace metadata
    def workspace_entity_address_metadata(self, input_vectors, *, episodes=None, **_):
        examples = self.last_batch_examples if episodes is None else episodes
        batch, time = input_vectors.shape[:2]
        if len(examples) != batch:
            raise RuntimeError("language address metadata lost batch/episode alignment")
        rows, pos, keys, writes = [], [], [], []
        match_rows, match_pos, match_keys = [], [], []
        for row, episode in enumerate(examples):
            for p, key, write in episode.metadata.get("lang_addr", ()):
                if p < time:
                    rows.append(row)
                    pos.append(p)
                    keys.append(key)
                    writes.append(write)
            for p, key in episode.metadata.get("workspace_match_writes", ()):
                if p < time:
                    match_rows.append(row)
                    match_pos.append(p)
                    match_keys.append(key)
        ids = torch.full((batch, time), -1, dtype=torch.long)
        mask = torch.zeros((batch, time), dtype=torch.bool)
        if rows:
            r, p = torch.tensor(rows), torch.tensor(pos)
            ids[r, p] = torch.tensor(keys)
            mask[r, p] = torch.tensor(writes)
        match_ids = torch.full((batch, time), -1, dtype=torch.long)
        match_mask = torch.zeros((batch, time), dtype=torch.bool)
        if match_rows:
            mr, mp = torch.tensor(match_rows), torch.tensor(match_pos)
            match_ids[mr, mp] = torch.tensor(match_keys)
            match_mask[mr, mp] = True
        device = input_vectors.device
        return {
            "entity_ids": ids.to(device),
            "entity_write_mask": mask.to(device),
            "workspace_match_entity_ids": match_ids.to(device),
            "workspace_match_write_mask": match_mask.to(device),
        }

    def workspace_key_value_metadata(
        self, input_vectors, episodes=None, *, include_adaptive_v3=False, **_
    ):
        examples = self.last_batch_examples if episodes is None else episodes
        batch, time = input_vectors.shape[:2]
        if len(examples) != batch:
            raise RuntimeError("language key-value metadata lost batch/episode alignment")
        cap = KEY_VALUE_OPERAND_CAPACITY
        copy_tracks = {
            "lang_parse",
            "lang_define",
            "lang_logic",
            "lang_induce",
            "lang_act",
            "lang_discourse",
            "lang_nlg",
        }
        function_pieces = {
            token for word in (*lg.FUNCTION, *lg.PUNCT) for token in lt.tokenizer().word(word)
        }

        def copy_source(episode):
            md = episode.metadata
            source_ids = tuple(md.get("workspace_copy_source_token_ids", ()))
            source_positions = tuple(md.get("workspace_copy_source_positions", ()))
            if episode.track in ("lang_discourse", "lang_act"):
                # These answers may depend on earlier turns carried in
                # `prev`; expose those visible input tokens to the copy arm as
                # well as tokens in the final segment. This is essential for
                # a_ask: its answer location is absent from the final query.
                source_ids = tuple(
                    token
                    for segment in (*tuple(md.get("prev", ())), episode)
                    for token in segment.inputs
                )
                source_positions = ()
            elif episode.track == "lang_nlg" and md.get("progression_group") == "retell_all":
                # Ep (the Stage-4 language builder) predates Ep5's copy
                # metadata. Retell-all is an exact-context reproduction task.
                # The final query segment carries the context in `prev`, so
                # include those tokens as well as the current segment.
                source_ids = tuple(
                    token
                    for segment in (*tuple(md.get("prev", ())), episode)
                    for token in segment.inputs
                )
                source_positions = ()
            if source_ids and len(source_positions) != len(episode.answer):
                source_by_token: dict[int, int] = {}
                for position, token in enumerate(source_ids):
                    source_by_token[int(token)] = position
                source_positions = tuple(
                    source_by_token.get(int(token), -1)
                    if int(token) < TEXT_BASE or int(token) not in function_pieces
                    else -1
                    for token in episode.answer
                )
            return source_ids, source_positions

        copy_sources = [copy_source(episode) for episode in examples]
        used: dict[tuple[int, int], int] = {}
        rr, ss, cc, kk, tt = [], [], [], [], []
        route_mask = torch.ones((batch, time), dtype=torch.bool)
        slot_target_ids = torch.full((batch, time), -1, dtype=torch.long)
        copy_target_positions = torch.full((batch, time), -1, dtype=torch.long)
        copy_source_size = max(
            (len(source_ids) for source_ids, _ in copy_sources),
            default=0,
        )
        copy_source_token_ids = torch.zeros((batch, copy_source_size), dtype=torch.long)
        copy_source_mask = torch.zeros((batch, copy_source_size), dtype=torch.bool)
        binding_aux_targets = torch.full((batch, time), -100, dtype=torch.long)
        copy_target_token_ids = torch.full((batch, time), -100, dtype=torch.long)
        for row, episode in enumerate(examples):
            limit = min(time, len(episode.inputs))
            md = episode.metadata
            source_ids, source_positions = copy_sources[row]
            if episode.track in copy_tracks and source_ids:
                copy_source_token_ids[row, : len(source_ids)] = torch.tensor(source_ids)
                copy_source_mask[row, : len(source_ids)] = True
            for step, target in md.get("workspace_binding_aux", ()):
                step, target = int(step), int(target)
                if 0 <= step < limit and 0 <= target < self.output_dim:
                    binding_aux_targets[row, step] = target
            for step, key, target in (
                *md.get("workspace_event_writes", ()),
                *md.get("workspace_query_reads", ()),
            ):
                col = used.get((row, step), 0)
                if step >= limit or col >= cap:
                    continue
                used[(row, step)] = col + 1
                rr.append(row), ss.append(step), cc.append(col), kk.append(key), tt.append(target)
            output_keys = tuple(md.get("workspace_output_keys", ()))
            output_kv_targets = md.get("workspace_output_kv_targets")
            # Episode builders append EOS in their shared `done` method, while
            # copy metadata is produced from the raw answer tokens. Zip against
            # the canonical episode answer so any legacy/overlong metadata is
            # safely truncated and EOS remains unsupervised by the copy head.
            for offset, (target_token, source_position) in enumerate(
                zip(episode.answer, source_positions)
            ):
                step = len(episode.inputs) + offset
                source_position = int(source_position)
                if 0 <= source_position < len(source_ids) and step < time:
                    copy_target_positions[row, step] = source_position
                    copy_target_token_ids[row, step] = int(target_token)
            for offset, key in enumerate(output_keys):
                key = int(key)
                step = len(episode.inputs) + offset
                if key >= 128:
                    raise ValueError("language workspace output key is outside [0, 128)")
                if key < 0 or step >= time or offset >= len(episode.answer):
                    continue
                slot_target_ids[row, step] = key
                if output_kv_targets is None:
                    continue
                col = used.get((row, step), 0)
                if col >= cap:
                    continue
                if offset >= len(output_kv_targets):
                    continue
                used[(row, step)] = col + 1
                (
                    rr.append(row),
                    ss.append(step),
                    cc.append(col),
                    kk.append(key),
                    tt.append(int(output_kv_targets[offset])),
                )
                route_mask[row, step] = False
        ids = torch.full((batch, time, cap), -1, dtype=torch.long)
        targets = torch.full((batch, time, cap), -100, dtype=torch.long)
        mask = torch.zeros((batch, time, cap), dtype=torch.bool)
        if rr:
            idx = (torch.tensor(rr), torch.tensor(ss), torch.tensor(cc))
            ids[idx], targets[idx], mask[idx] = torch.tensor(kk), torch.tensor(tt), True
        result = {
            "key_value_entity_ids": ids,
            "key_value_targets": targets,
            "key_value_mask": mask,
            "key_value_route_mask": route_mask,
            "workspace_slot_target_ids": slot_target_ids,
            "workspace_copy_source_token_ids": copy_source_token_ids if copy_source_size else None,
            "workspace_copy_source_mask": copy_source_mask if copy_source_size else None,
            "workspace_copy_target_positions": copy_target_positions,
            "workspace_copy_target_token_ids": copy_target_token_ids,
            "workspace_copy_has_targets": any(
                bool(source_positions) and any(int(p) >= 0 for p in source_positions)
                for _source_ids, source_positions in copy_sources
            ),
            "workspace_binding_aux_targets": binding_aux_targets,
            "key_value_steps": tuple(sorted(set(ss))),
            "key_value_output_programs": (),
            "key_value_argmatch_locations": {},
        }
        if include_adaptive_v3:  # language tracks carry no operator programs: V3/V4 stay inert
            result["adaptive_operator_candidate_mask"] = torch.zeros(
                (batch, time), dtype=torch.bool
            )
            result["adaptive_operator_execution_mask"] = torch.ones((batch, time), dtype=torch.bool)
            result["adaptive_operator_steps"] = ()
            result["adaptive_operator_teacher_steps"] = ()
        return result

    # ------------------------------------------------------------------ segment-carry (Stage 5)
    def _emit(self, group, cont):
        group = [g if g is not None else _DUMMY for g in group]
        self.last_batch_examples, self.segment_continue = group, cont
        x, y, m = self._collate(
            group, token_ids_only=True, teacher_forced=self.teacher_forced_answers
        )
        m[torch.tensor([g is _DUMMY for g in group])] = 0.0
        for row, episode in enumerate(group):
            if episode.metadata.get("context_only"):
                m[row].zero_()
        lengths = [len(episode.sequence) for episode in group]
        max_length = max(lengths, default=0)
        self.last_batch_sequence_max = max_length
        self.last_batch_sequence_mean = sum(lengths) / max(1, len(lengths))
        self.last_batch_padding_utilization = sum(lengths) / max(1, len(lengths) * max_length)
        return x, y, m

    def sample_batch(self, curriculum, batch_size):
        if self._seg_queue:
            group, cont, steps, weight, chain_start, chain_complete = self._seg_queue.pop(0)
            self.segment_chain_start = chain_start
            self.segment_chain_complete = chain_complete and not self._seg_queue
            self.segment_chain_steps = steps
            self.segment_loss_weight = weight
            if self.segment_chain_complete:
                for track in self._pending_chain_tracks:
                    self._track_optimizer_step_counts[track] += 1
                self._pending_chain_tracks.clear()
            return self._emit(group, cont)
        sample_count = batch_size * self.length_bucket_microbatches
        super().sample_batch(curriculum, sample_count)
        sampled_tracks = [
            episode.track for episode in self.last_batch_examples if episode is not _DUMMY
        ]
        for track in sampled_tracks:
            self._track_sampled_episode_counts[track] += 1
        for track in set(sampled_tracks):
            self._track_sampled_batch_counts[track] += 1
        self.segment_continue = False
        chains = [tuple(e.metadata.get("prev", ())) + (e,) for e in self.last_batch_examples]
        self._pending_chain_tracks = {
            episode.track for episode in self.last_batch_examples if episode is not _DUMMY
        }
        buckets: dict[int, list] = defaultdict(list)
        for episode, chain in zip(self.last_batch_examples, chains):
            buckets[len(chain)].append((episode, chain))
        ordered = [
            sorted(buckets[n], key=lambda item: len(item[0].sequence)) for n in sorted(buckets)
        ]
        total_episodes = max(1, len(self.last_batch_examples))
        segments = []
        for bucket in ordered:
            n = len(bucket[0][1])
            chunk_count = min(self.length_bucket_microbatches, len(bucket))
            for chunk_index in range(chunk_count):
                start = (len(bucket) * chunk_index) // chunk_count
                end = (len(bucket) * (chunk_index + 1)) // chunk_count
                chunk = bucket[start:end]
                group_weight = len(chunk) / total_episodes
                for segment_index in range(n):
                    group = [chain[segment_index] for _, chain in chunk]
                    segments.append(
                        (
                            group,
                            segment_index > 0,
                            n,
                            group_weight,
                            not segments,
                            False,
                        )
                    )
        if segments:
            final = list(segments[-1])
            final[-1] = True
            segments[-1] = tuple(final)
        self._seg_queue = segments[1:]
        group, cont, steps, weight, chain_start, _complete = segments[0]
        self.segment_chain_start = chain_start
        self.segment_chain_complete = _complete and not self._seg_queue
        self.segment_chain_steps = steps
        self.segment_loss_weight = weight
        if self.segment_chain_complete:
            for track in self._pending_chain_tracks:
                self._track_optimizer_step_counts[track] += 1
            self._pending_chain_tracks.clear()
        return self._emit(group, cont)

    @property
    def has_pending_chain_segments(self):
        return bool(self._seg_queue)

    def loss(self, output, target, mask):
        """Average answer loss per example so long rationales do not dominate mixed batches."""
        if output.shape[:-1] != target.shape or target.shape != mask.shape:
            raise ValueError("output, target, and mask batch/time dimensions must match")
        valid = mask.bool()
        locations = torch.nonzero(valid, as_tuple=False)
        if locations.numel() == 0:
            return output.new_zeros((), requires_grad=output.requires_grad)
        token_loss = torch.nn.functional.cross_entropy(
            output[valid].float(), target[valid], reduction="none"
        )
        counts = valid.sum(dim=1).to(token_loss.dtype)
        active_rows = counts.gt(0)
        per_example_sum = token_loss.new_zeros((target.shape[0],)).scatter_add(
            0, locations[:, 0], token_loss
        )
        per_example = per_example_sum / counts.clamp_min(1)
        return per_example[active_rows].mean()

    @staticmethod
    def loss_sparse(logits, target, mask, positions):
        """Per-example answer CE for logits projected only at selected positions."""
        if logits.ndim != 2 or positions.ndim != 2 or positions.shape[-1] != 2:
            raise ValueError(
                "sparse language loss expects [positions,vocab] logits and [N,2] coordinates"
            )
        if positions.shape[0] != logits.shape[0]:
            raise ValueError("sparse language logits and coordinates differ in length")
        if logits.shape[0] == 0:
            return logits.new_zeros((), requires_grad=logits.requires_grad)
        rows, times = positions[:, 0], positions[:, 1]
        active = mask[rows, times].bool()
        sparse_targets = target[rows, times].masked_fill(~active, -100)
        token_loss = torch.nn.functional.cross_entropy(
            logits.float(), sparse_targets, reduction="none", ignore_index=-100
        )
        valid_weight = active.to(token_loss.dtype)
        counts = token_loss.new_zeros((target.shape[0],)).scatter_add(0, rows, valid_weight)
        sums = token_loss.new_zeros((target.shape[0],)).scatter_add(0, rows, token_loss)
        has_answer = counts.gt(0).to(token_loss.dtype)
        per_example = sums / counts.clamp_min(1.0)
        return (per_example * has_answer).sum() / has_answer.sum().clamp_min(1.0)

    @staticmethod
    def diversity_sparse(logits, mask, positions):
        if logits.shape[0] == 0:
            return logits.new_zeros(())
        rows, times = positions[:, 0], positions[:, 1]
        active = mask[rows, times].bool()
        prediction = logits.argmax(dim=-1)
        histogram = torch.bincount(
            prediction,
            weights=active.to(dtype=torch.float32),
            minlength=logits.shape[-1],
        )
        # Keep this as a device scalar. The training loop transfers it only
        # once when it emits the periodic log, rather than synchronizing each
        # segment to convert a unique-count to Python float.
        return histogram.ne(0).sum().to(dtype=logits.dtype)

    @staticmethod
    def apply_workspace_key_value_readout_sparse(
        logits, positions, predictions, metadata, answer_mask
    ):
        """Route language answer positions without materializing dense vocabulary logits."""
        if not predictions:
            return logits
        if metadata.get("key_value_output_programs"):
            raise ValueError(
                "sparse language readout does not support structured KV output programs"
            )
        rows, times = positions[:, 0], positions[:, 1]
        masks = metadata["key_value_mask"].to(device=logits.device)
        route_mask = metadata.get("key_value_route_mask")
        if route_mask is not None:
            route_mask = route_mask.to(device=logits.device).bool()
        answer_mask = answer_mask.to(device=logits.device).bool()
        result = logits
        for step, value_logits in predictions:
            step = int(step)
            if not 0 <= step < answer_mask.shape[1]:
                continue
            # Work over the already-gathered sparse answer rows. Avoid
            # per-step CUDA nonzero()/Python truth checks, which synchronize
            # the device for each workspace prediction time.
            active = times.eq(step) & masks[rows, times, 0] & answer_mask[rows, times]
            if route_mask is not None:
                active &= route_mask[rows, times]
            routed = (
                value_logits[:, 0]
                .index_select(0, rows)
                .to(device=logits.device, dtype=logits.dtype)
            )
            result = torch.where(active.unsqueeze(-1), routed, result)
        return result

    def _evaluate_episodes(self, model, device, episodes, **kw):
        if not any(e.metadata.get("prev") for e in episodes) or not hasattr(
            model, "_initium_carry"
        ):
            return super()._evaluate_episodes(model, device, episodes, **kw)
        context_teacher_forced = bool(kw.get("teacher_forced", self.teacher_forced_answers))
        hint, carries = min(self.batch_size_hint, max(1, len(episodes))), []
        prior_segment_count = sum(len(tuple(e.metadata.get("prev", ()))) for e in episodes)
        if prior_segment_count:
            print(
                f"  [lang-eval-context] replaying {prior_segment_count} prior segments "
                f"| teacher_forced={'on' if context_teacher_forced else 'off'} "
                f"| workspace_kv={'on' if getattr(model, 'workspace_key_value_readout_enabled', False) else 'off'}"
            )
        for off in range(0, len(episodes), hint):
            chains = [tuple(e.metadata.get("prev", ())) for e in episodes[off : off + hint]]
            n, carry = max(map(len, chains)), None
            for i in range(n):
                group = [c[i - (n - len(c))] if i >= n - len(c) else _DUMMY for c in chains]
                model._initium_carry = [carry]
                self._probe_forward(
                    model,
                    device,
                    group,
                    lambda *a: None,
                    want_kv=bool(getattr(model, "workspace_key_value_readout_enabled", False)),
                    teacher_forced=context_teacher_forced,
                    state_only=True,
                )
                carry = model._carry_out
            carries.append(carry)
        model._initium_carry = carries
        try:
            return super()._evaluate_episodes(model, device, episodes, **kw)
        finally:
            model._initium_carry = []

    def _ungrounded(self, episodes):
        """Content-token support from the visible context plus labelled slot values."""
        tk = lt.tokenizer()
        func = {t for w in (*lg.FUNCTION, *lg.PUNCT, *_EXTRA_FUNC) for t in tk.word(w)}
        bad = tot = n = 0
        for e in episodes:
            p = self._last_eval_preds.get(id(e))
            if p is None or e.metadata.get("answer_kind", "text") not in ("text", "mr"):
                continue
            supported = set(e.inputs)
            metadata = e.metadata
            for field in (
                "workspace_event_writes",
                "workspace_query_reads",
                "workspace_binding_aux",
                "language_induction_binding_aux",
            ):
                for row in metadata.get(field, ()):
                    value = int(row[-1])
                    if value >= TEXT_BASE:
                        supported.add(value)
            n += 1
            for t in p:
                if t >= TEXT_BASE and t not in func:
                    tot += 1
                    bad += t not in supported
        return (
            {"ungrounded_tokens": {"rate_pct": 100.0 * bad / max(tot, 1), "episodes": n}}
            if n
            else {}
        )

    def _lang5_metrics(self, episodes):
        """Coreference-by-distance, derivation-by-depth and clarification P/R (also printed by the caller)."""
        preds, clar = self._last_eval_preds, [*ltr._toks(["who", "do", "you", "mean?"]), EOS]
        acc: dict[str, list[int]] = defaultdict(lambda: [0, 0])
        induce_positions: dict[int, list[int]] = defaultdict(lambda: [0, 0])
        induce_kinds: dict[str, list[int]] = defaultdict(lambda: [0, 0])
        induce_entity_targets: dict[int, list[int]] = defaultdict(lambda: [0, 0])
        induce_entity_predictions: dict[str, int] = defaultdict(int)
        induce_entity_confusions: dict[tuple[int, str], int] = defaultdict(int)
        induce_eos = [0, 0]
        act_positions: dict[tuple[str, int], list[int]] = defaultdict(lambda: [0, 0])
        act_eos: dict[str, list[int]] = defaultdict(lambda: [0, 0])
        output_positions: dict[tuple[str, str, int], list[int]] = defaultdict(lambda: [0, 0])
        output_eos: dict[tuple[str, str], list[int]] = defaultdict(lambda: [0, 0])
        tp = pc = gc = 0
        for e in episodes:
            p = preds.get(id(e))
            if p is None:
                continue
            m, ok = e.metadata, p == list(e.answer)
            if m.get("coref_dist", -1) >= 0:
                a = acc[f"coref_dist_{min(m['coref_dist'], 6)}"]
                a[0] += ok
                a[1] += 1
            if e.track in ("lang_logic", lt.LANG_PWS_LOGIC):
                a = acc[f"derivation_depth_{m.get('depth', 0)}"]
                a[0] += ok
                a[1] += 1
            if e.track in ("lang_induce", lt.LANG_PWS_INDUCE):
                answer_kinds = m.get("induction_answer_kinds", ())
                for position, target in enumerate(e.answer):
                    if target == EOS:
                        induce_eos[0] += int(position < len(p) and p[position] == target)
                        induce_eos[1] += 1
                        continue
                    counts = induce_positions[position]
                    correct = int(position < len(p) and p[position] == target)
                    counts[0] += correct
                    counts[1] += 1
                    if e.track == lt.LANG_PWS_INDUCE and position < len(answer_kinds):
                        answer_kind = str(answer_kinds[position])
                        if answer_kind in ("entity", "relation"):
                            kind_counts = induce_kinds[answer_kind]
                            kind_counts[0] += correct
                            kind_counts[1] += 1
                        if answer_kind == "entity" and ENTITY_BASE <= int(target) < (
                            ENTITY_BASE + ENTITY_COUNT
                        ):
                            target_entity = int(target) - ENTITY_BASE
                            target_counts = induce_entity_targets[target_entity]
                            target_counts[0] += correct
                            target_counts[1] += 1
                            if position < len(p) and ENTITY_BASE <= int(p[position]) < (
                                ENTITY_BASE + ENTITY_COUNT
                            ):
                                predicted_entity = str(int(p[position]) - ENTITY_BASE)
                            else:
                                predicted_entity = "other"
                            induce_entity_predictions[predicted_entity] += 1
                            induce_entity_confusions[(target_entity, predicted_entity)] += 1
            if e.track in ("lang_parse", "lang_define"):
                group = str(m.get("progression_group", "all"))
                for position, target in enumerate(e.answer):
                    if target == EOS:
                        eos = output_eos[(e.track, group)]
                        eos[0] += int(position < len(p) and p[position] == target)
                        eos[1] += 1
                        continue
                    counts = output_positions[(e.track, group, position)]
                    counts[0] += int(position < len(p) and p[position] == target)
                    counts[1] += 1
            if e.track == "lang_act":
                response_act = str(m.get("response_act", "unknown"))
                for position, target in enumerate(e.answer):
                    if target == EOS:
                        eos = act_eos[response_act]
                        eos[0] += int(position < len(p) and p[position] == target)
                        eos[1] += 1
                        continue
                    counts = act_positions[(response_act, position)]
                    counts[0] += int(position < len(p) and p[position] == target)
                    counts[1] += 1
                g, c = m.get("response_act") == "clarify", p == clar
                tp += g and c
                pc += c
                gc += g
        out = {k: {"exact_pct": 100.0 * a / n, "episodes": n} for k, (a, n) in sorted(acc.items())}
        out.update(
            {
                f"induce_answer_position_{position}": {
                    "accuracy_pct": 100.0 * correct / max(count, 1),
                    "episodes": count,
                }
                for position, (correct, count) in sorted(induce_positions.items())
            }
        )
        if induce_eos[1]:
            out["induce_eos_accuracy"] = {
                "accuracy_pct": 100.0 * induce_eos[0] / induce_eos[1],
                "episodes": induce_eos[1],
            }
        out.update(
            {
                f"induce_pws_{kind}_binding_accuracy": {
                    "accuracy_pct": 100.0 * correct / max(count, 1),
                    "episodes": count,
                }
                for kind, (correct, count) in sorted(induce_kinds.items())
            }
        )
        out.update(
            {
                f"induce_pws_entity_target_{target}_accuracy": {
                    "accuracy_pct": 100.0 * correct / max(count, 1),
                    "episodes": count,
                }
                for target, (correct, count) in sorted(induce_entity_targets.items())
            }
        )
        out.update(
            {
                f"induce_pws_entity_prediction_{prediction}_count": {
                    "count": count,
                    "targets": count,
                }
                for prediction, count in sorted(induce_entity_predictions.items())
            }
        )
        out.update(
            {
                f"induce_pws_entity_target_{target}_prediction_{prediction}_count": {
                    "count": count,
                    "targets": count,
                }
                for (target, prediction), count in sorted(induce_entity_confusions.items())
            }
        )
        out.update(
            {
                f"act_answer_position_{act}_{position}": {
                    "accuracy_pct": 100.0 * correct / max(count, 1),
                    "episodes": count,
                }
                for (act, position), (correct, count) in sorted(act_positions.items())
            }
        )
        out.update(
            {
                f"act_eos_accuracy_{act}": {
                    "accuracy_pct": 100.0 * correct / max(count, 1),
                    "episodes": count,
                }
                for act, (correct, count) in sorted(act_eos.items())
            }
        )
        out.update(
            {
                f"{track}_answer_position_{group}_{position}": {
                    "accuracy_pct": 100.0 * correct / max(count, 1),
                    "episodes": count,
                }
                for (track, group, position), (correct, count) in sorted(output_positions.items())
            }
        )
        out.update(
            {
                f"{track}_eos_accuracy_{group}": {
                    "accuracy_pct": 100.0 * correct / max(count, 1),
                    "episodes": count,
                }
                for (track, group), (correct, count) in sorted(output_eos.items())
            }
        )
        if gc or pc:
            out["clarify"] = {
                "precision_pct": 100.0 * tp / max(pc, 1),
                "recall_pct": 100.0 * tp / max(gc, 1),
                "episodes": gc,
            }
        return out

    # ------------------------------------------------------------------ display
    @staticmethod
    def _fmt_answer(e, pred):
        kind = e.metadata.get("answer_kind", "text")

        def fmt(tokens):
            toks = [int(t) for t in tokens]
            toks = toks[: toks.index(EOS)] if EOS in toks else toks
            if kind == "bool":
                return " ".join(
                    "yes" if t == BOOL_TRUE else "no" if t == BOOL_FALSE else "?" for t in toks
                )
            if kind == "mr":
                return " ".join(
                    lt.tokenizer().decode([t]) if t >= TEXT_BASE else f"<{t}>" for t in toks
                )
            if kind == "typed":
                return " ".join(
                    _dev_lexicon.word_name(t - FACT_BASE)
                    if FACT_BASE <= t < FACT_BASE + FACT_COUNT
                    else _dev_lexicon.word_name(t - RELATION_BASE)
                    if RELATION_BASE <= t < RELATION_BASE + RELATION_COUNT
                    else str(t)
                    for t in toks
                )
            return lt.tokenizer().decode(toks)

        return f"guess: {fmt(pred)} | gold: {fmt(e.answer)}"

    # ------------------------------------------------------------------ evaluation
    def evaluate_ood(self, model, device, num_episodes, rng, verbose_n=0, field_log=None):
        # label / length / combo / combined splits; language selectors resolve
        # one concrete track per paired seed in _make_episode.
        return self.evaluate_dev_axes(model, device, num_episodes, rng, verbose_n, field_log)

    def _generation_metrics(self, model, device, episodes, cap=None):
        cap = cap or self.generation_max_tokens
        seen: dict[tuple, int] = defaultdict(int)
        picked = []
        for e in episodes:
            key = (e.track, e.metadata.get("progression_group"))
            if len(e.answer) <= cap and seen[key] < self.generation_eval_per_group:
                seen[key] += 1
                picked.append(e)
        if not picked:
            return {}
        generate = super()._generation_metrics
        by_track = {
            track: generate(model, device, [e for e in picked if e.track == track], cap=cap)
            for track in sorted({e.track for e in picked})
        }
        total = sum(int(values.get("episodes", 0)) for values in by_track.values())
        overall = {}
        for metric in (
            "exact",
            "token_accuracy",
            "length_accuracy",
            "repetition_rate",
            "fact_set_match",
            "teacher_forced_exact",
        ):
            overall[metric] = sum(
                float(values.get(metric, 0.0)) * int(values.get("episodes", 0))
                for values in by_track.values()
            ) / max(total, 1)
        overall["gap_pp"] = overall["teacher_forced_exact"] - overall["exact"]
        overall["free_over_tf"] = overall["exact"] / max(overall["teacher_forced_exact"], 1e-9)
        overall["episodes"] = total
        return {"all": overall, "by_track": by_track}

    def evaluate_curriculum_id(self, model, device, lesson_idx, *, field_log=None):
        self._curriculum_track_levels = (
            dict(lesson_idx)
            if isinstance(lesson_idx, dict)
            else {track: int(lesson_idx) for track in self._tracks()}
        )
        result = super().evaluate_curriculum_id(model, device, lesson_idx, field_log=field_log)
        diagnostics = self.last_eval_diagnostics
        evaluation_track_lessons = diagnostics.get("evaluation_track_lessons", {})
        progression_groups = diagnostics.get("progression_groups", {})
        weight_basis_exact = {}
        for track, values in diagnostics.get("tracks", {}).items():
            # Track-level averages can hide the one group blocking promotion
            # (for example, a_ask behind easy a_inform). Weight from the
            # weakest observed group so solved strata do not starve weak ones.
            group_scores = [
                float(group_values["exact_match"])
                for group_values in progression_groups.get(track, {}).values()
                if group_values.get("episodes", 0) > 0 and "exact_match" in group_values
            ]
            weight_basis_exact[track] = (
                min(group_scores) if group_scores else float(values["exact_match"])
            )
        self._track_group_sampling_weights = {
            track: {
                group: max(
                    0.05,
                    0.05 + max(0.0, 0.9 - float(group_values.get("exact_match", 0.0)) / 100.0),
                )
                for group, group_values in progression_groups.get(track, {}).items()
                if group_values.get("episodes", 0) > 0
            }
            for track in diagnostics.get("tracks", {})
        }
        raw_weights = {
            # Keep solved Stage-4 tracks in the replay mixture so they retain
            # a meaningful minimum share while weak Stage-5 tracks are favored.
            track: 0.05
            if track == "lang_open"
            else 0.05 + max(0.0, 0.9 - weight_basis_exact[track] / 100.0)
            for track in diagnostics.get("tracks", {})
        }
        stalled = set()
        for track, values in self.last_eval_diagnostics.get("tracks", {}).items():
            history = self._track_exact_history.setdefault(track, [])
            history.append(float(values["exact_match"]))
            del history[:-3]
            if len(history) == 3 and max(history) - min(history) < 2.0:
                stalled.add(track)
        active = sorted(value for track, value in raw_weights.items() if track != "lang_open")
        median = active[len(active) // 2] if active else 1.0
        self._track_weights = {}
        for track, weight in raw_weights.items():
            if track == "lang_open":
                self._track_weights[track] = weight
                continue
            # The hardest track also needs enough updates to learn its binding
            # operation. Capping it below the ordinary ceiling caused its low
            # exact score to persist while it received less training.
            cap_scale = (
                3.0
                if track in ("lang_induce", lt.LANG_PWS_INDUCE, lt.LANG_PWS_LOGIC)
                else (1.5 if track in stalled else 3.0)
            )
            self._track_weights[track] = min(weight, max(0.05, cap_scale * median))
        sampled_steps = dict(self._track_optimizer_step_counts)
        sampled_episodes = dict(self._track_sampled_episode_counts)
        sampled_batches = dict(self._track_sampled_batch_counts)
        self._track_optimizer_step_counts.clear()
        self._track_sampled_episode_counts.clear()
        self._track_sampled_batch_counts.clear()
        step_total = sum(sampled_steps.values())
        episode_total = sum(sampled_episodes.values())
        batch_total = sum(sampled_batches.values())
        weight_total = sum(self._track_weights.values())
        diagnostics["track_sampling"] = {
            track: {
                "sampled_episodes": int(sampled_episodes.get(track, 0)),
                "sampled_episode_share_pct": 100.0
                * sampled_episodes.get(track, 0)
                / max(episode_total, 1),
                "sampled_batches": int(sampled_batches.get(track, 0)),
                "sampled_batch_share_pct": 100.0
                * sampled_batches.get(track, 0)
                / max(batch_total, 1),
                "optimizer_steps": int(sampled_steps.get(track, 0)),
                "observed_step_share_pct": 100.0 * sampled_steps.get(track, 0) / max(step_total, 1),
                "weight_basis_exact_pct": float(weight_basis_exact.get(track, 0.0)),
                "next_sampling_share_pct": 100.0 * weight / max(weight_total, 1e-12),
            }
            for track, weight in sorted(self._track_weights.items())
        }
        diagnostics["group_sampling_weights"] = {
            track: dict(sorted(weights.items()))
            for track, weights in sorted(self._track_group_sampling_weights.items())
        }
        print(
            "  [track-sampling] "
            + " ".join(
                f"{track}=episodes:{sampled_episodes.get(track, 0)} "
                f"batches:{sampled_batches.get(track, 0)} "
                f"steps:{sampled_steps.get(track, 0)} "
                f"basis_exact:{weight_basis_exact.get(track, 0.0):.1f}% "
                f"episode_share:{sampled_episodes.get(track, 0) / max(episode_total, 1):.1%} "
                f"observed:{sampled_steps.get(track, 0) / max(step_total, 1):.1%} "
                f"next:{weight / max(weight_total, 1e-12):.1%}"
                for track, weight in sorted(self._track_weights.items())
            )
        )
        diagnostics, episodes, preds = (
            self.last_eval_diagnostics,
            self._last_id_eval_episodes,
            self._last_eval_preds,
        )
        if getattr(self, "progression_eval_only", False):
            # Keep the cheap, prediction-only breakdowns that explain why a
            # progression group misses its gate.  These reuse the just-finished
            # ID evaluation and do not add model forwards (unlike generation,
            # slot surgery, paraphrase, or grounding probes).
            try:
                position_metrics = self._lang5_metrics(episodes)
                diagnostics["progression_position_metrics"] = position_metrics
                for key, values in sorted(position_metrics.items()):
                    if key.startswith(
                        (
                            "induce_answer_position_",
                            "induce_eos_accuracy",
                            "induce_pws_",
                            "act_answer_position_",
                            "act_eos_accuracy_",
                        )
                    ):
                        print(
                            f"  [lang/progression/{key}] count={values['count']} "
                            f"n={values.get('targets', 0)}"
                            if "count" in values
                            else f"  [lang/progression/{key}] "
                            f"accuracy={values.get('accuracy_pct', values.get('exact_pct', 0.0)):.1f}% "
                            f"n={values.get('episodes', 0)}"
                        )
            except (RuntimeError, ValueError, KeyError) as exc:
                print(f"[lang-progression-diagnostics] skipped: {exc!r}")
            self.last_eval_diagnostics, self._last_eval_preds = diagnostics, preds
            return result
        self._print_samples(episodes, "id")
        diagnostics["stage3_counterfactual"] = self._stage3_metrics(episodes)
        extra: dict = {}
        try:
            if self.generation_eval:
                generation = self._generation_metrics(model, device, episodes)
                diagnostics["production_generation"] = generation
                for track, values in generation.get("by_track", {}).items():
                    print(
                        f"  [lang/free-generation/{track} lesson="
                        f"{evaluation_track_lessons.get(track, 0) + 1}] "
                        f"exact={values.get('exact', 0.0):.1f}% "
                        f"teacher_forced={values.get('teacher_forced_exact', 0.0):.1f}% "
                        f"gap={values.get('gap_pp', 0.0):+.1f}pp "
                        f"n={int(values.get('episodes', 0))}"
                    )
            workspace = getattr(model, "workspace", None)
            if workspace is not None and getattr(
                model, "workspace_semantic_entity_addressing", False
            ):
                if getattr(model, "workspace_key_value_readout_enabled", False):
                    extra.update(self._slot_probe(model, device, episodes))
                probe_level = (
                    max(lesson_idx.values(), default=0)
                    if isinstance(lesson_idx, dict)
                    else int(lesson_idx)
                )
                extra.update(self._slot_surgery(model, device, probe_level))
            probe_level = (
                max(lesson_idx.values(), default=0)
                if isinstance(lesson_idx, dict)
                else int(lesson_idx)
            )
            extra.update(self._paraphrase_probe(model, device, probe_level))
            self._last_eval_preds = (
                preds  # the probes above overwrote it; the metrics below need ID predictions
            )
            extra.update(self._ungrounded(episodes))
            extra.update(self._lang5_metrics(episodes))
        except (RuntimeError, ValueError, KeyError) as exc:  # diagnostics must not kill a long run
            print(f"[lang-diagnostics] skipped: {exc!r}")
        diagnostics["production"] = extra
        for k, v in sorted(extra.items()):
            print(
                f"  [lang/{k}] "
                + " ".join(f"{a}={b:.1f}" for a, b in v.items() if a != "episodes")
                + f" n={v.get('episodes', 0)}"
            )
        self.last_eval_diagnostics, self._last_eval_preds = diagnostics, preds
        return result

    # ------------------------------------------------------------------ section-8 diagnostics
    def _probe_forward(
        self,
        model,
        device,
        episodes,
        on_batch,
        make_observer=None,
        want_kv=False,
        teacher_forced=False,
        state_only=False,
    ):
        """Microbatched eval forward; calls on_batch(group, answer_preds, kv_meta, kv_preds)."""
        workspace = getattr(model, "workspace", None)
        prior, was_training = (
            getattr(workspace, "_initium_recurrent_state_observer", None),
            model.training,
        )
        model.eval()
        try:
            with torch.no_grad():
                for off in range(0, len(episodes), self.batch_size_hint):
                    group = episodes[off : off + self.batch_size_hint]
                    x, _targets, answer_mask = self._collate(
                        group,
                        token_ids_only=True,
                        teacher_forced=teacher_forced,
                    )
                    kwargs = {
                        k: v.to(device)
                        for k, v in self.workspace_entity_address_metadata(
                            x, episodes=group
                        ).items()
                    }
                    kv = self.workspace_key_value_metadata(x, group) if want_kv else None
                    if kv is not None:
                        kwargs.update(
                            key_value_entity_ids=kv["key_value_entity_ids"].to(device),
                            key_value_mask=kv["key_value_mask"].to(device),
                            key_value_steps=kv["key_value_steps"],
                        )
                    if workspace is not None:
                        workspace._initium_recurrent_state_observer = (
                            make_observer(group) if make_observer else None
                        )
                    output, state = model(
                        x.to(device), (None, None, None), reset_experience=True, **kwargs
                    )
                    if state_only:
                        if want_kv:
                            workspace.pop_key_value_predictions()
                        continue
                    features = output.transpose(0, 1)
                    logits = self._output_proj(features)
                    if getattr(model, "workspace_slot_readout_enabled", False):
                        logits = model.apply_slot_readout(
                            logits,
                            features,
                            state,
                            answer_mask.to(device),
                            copy_source_token_ids=(kv or {}).get("workspace_copy_source_token_ids"),
                            copy_source_mask=(kv or {}).get("workspace_copy_source_mask"),
                        )
                    tokens = logits.argmax(-1).cpu().tolist()
                    preds = [
                        t[len(e.inputs) : len(e.inputs) + len(e.answer)]
                        for e, t in zip(group, tokens)
                    ]
                    on_batch(
                        group, preds, kv, workspace.pop_key_value_predictions() if want_kv else None
                    )
        finally:
            if workspace is not None:
                workspace._initium_recurrent_state_observer = prior
            model.train(was_training)

    def _slot_probe(self, model, device, episodes):
        """Slot-content decode accuracy at write time and at the final query (retention)."""
        eps, selected_per_track = [], defaultdict(int)
        for episode in episodes:
            if not episode.metadata.get("workspace_event_writes"):
                continue
            if selected_per_track[episode.track] >= self.diag_episodes:
                continue
            eps.append(episode)
            selected_per_track[episode.track] += 1
        if not eps:
            return {}
        stats: dict[tuple[str, str], list[int]] = defaultdict(lambda: [0, 0])
        track_stats: dict[tuple[str, str, str], list[int]] = defaultdict(lambda: [0, 0])

        def tally(group, _preds, kv, kv_preds):
            last = torch.tensor([len(e.inputs) - 1 for e in group]).unsqueeze(-1)
            for step, logits in kv_preds:
                mask, keys = kv["key_value_mask"][:, step], kv["key_value_entity_ids"][:, step]
                hit = logits.argmax(-1).cpu().eq(kv["key_value_targets"][:, step])
                retention = last.eq(step).expand_as(mask)
                kinds = torch.full_like(keys, 3)
                for lo, hi, idx in _SLOT_RANGES5 if lt.settings["stage5"] else _SLOT_RANGES:
                    kinds[(keys >= lo) & (keys < hi)] = idx
                for phase, flag in (("write", False), ("retention", True)):
                    in_phase = mask & (retention == flag)
                    for idx, name in enumerate(_SLOT_KINDS):
                        sel = in_phase & kinds.eq(idx)
                        stats[(phase, name)][0] += int((hit & sel).sum())
                        stats[(phase, name)][1] += int(sel.sum())
                        for row, episode in enumerate(group):
                            row_sel = sel[row]
                            track_stat = track_stats[(episode.track, phase, name)]
                            track_stat[0] += int((hit[row] & row_sel).sum())
                            track_stat[1] += int(row_sel.sum())
                    if not flag:
                        for i in range(8):
                            sel = in_phase & kinds.lt(3) & (keys % 8).eq(i)
                            stats[(phase, f"fact_{i}")][0] += int((hit & sel).sum())
                            stats[(phase, f"fact_{i}")][1] += int(sel.sum())
                            for row, episode in enumerate(group):
                                row_sel = sel[row]
                                track_stat = track_stats[(episode.track, phase, f"fact_{i}")]
                                track_stat[0] += int((hit[row] & row_sel).sum())
                                track_stat[1] += int(row_sel.sum())

        self._probe_forward(model, device, eps, tally, want_kv=True)
        return {
            f"slot_probe_{phase}/{name}": {"accuracy": 100.0 * h / n, "episodes": n}
            for (phase, name), (h, n) in sorted(stats.items())
            if n
        } | {
            f"slot_probe_track/{track}/{phase}/{name}": {
                "accuracy": 100.0 * h / n,
                "episodes": n,
            }
            for (track, phase, name), (h, n) in sorted(track_stats.items())
            if n
        }

    def _surgery_plans(self, lesson):
        if "lang_template" not in self._tracks() or not lt.settings["annotator"]:
            return []
        grammar, tk, rng = lg.grammar(), lt.tokenizer(), random.Random(91_800 + lesson)
        plans = []
        for _ in range(40 * self.diag_episodes):
            if len(plans) == self.diag_episodes:
                break
            ep = self._make_episode("lang_template", lesson, rng)
            md = ep.metadata
            if md.get("progression_group") != "qa_wh":
                continue
            reads = [(k, t) for _s, k, t in md["workspace_query_reads"] if 32 <= k < 40]
            objects = {k: t for _s, k, t in md["workspace_event_writes"] if 32 <= k < 40}
            if len(reads) != 1:
                continue
            ka, ta = reads[0]
            other = [(k, t) for k, t in objects.items() if k != ka and t != ta]
            if not other:
                continue
            kb, tb = other[0]
            enc = lambda t: list(tk.word(grammar.word[t - FACT_BASE]))  # noqa: E731
            plans.append((ep, len(ep.inputs) - 1, ka, kb, enc(ta), enc(tb)))
        return plans

    def _slot_surgery(self, model, device, lesson):
        """Swap two object-slot payloads at the last input step; the answer should follow the swap."""
        plans = self._surgery_plans(lesson)
        if not plans:
            return {}
        by_id, eps = {id(p[0]): p for p in plans}, [p[0] for p in plans]
        address_dim = model.workspace._address_dim

        def make_observer(group):
            at_step = defaultdict(list)
            for row, e in enumerate(group):
                _e, step, ka, kb, _oi, _oj = by_id[id(e)]
                at_step[step].append((row, ka, kb))

            def observe(state, call_index):
                for row, ka, kb in at_step.get(call_index, ()):
                    address = state[row, :, :address_dim]
                    occupied = address.square().sum(-1) > 1e-12
                    slot = address.argmax(-1)
                    ia, ib = (
                        ((slot == ka) & occupied).nonzero(),
                        ((slot == kb) & occupied).nonzero(),
                    )
                    if ia.numel() and ib.numel():
                        ia, ib = ia[0, 0], ib[0, 0]
                        swap = state[row, ia, address_dim:].clone()
                        state[row, ia, address_dim:] = state[row, ib, address_dim:]
                        state[row, ib, address_dim:] = swap

            return observe

        base, edited = {}, {}

        def sink(store):
            def update(group, preds, *_):
                store.update({id(e): p for e, p in zip(group, preds)})

            return update

        self._probe_forward(model, device, eps, sink(base))
        self._probe_forward(model, device, eps, sink(edited), make_observer)
        present = followed = changed = 0
        for ep, _step, _ka, _kb, obj_i, obj_j in plans:
            before, after = base[id(ep)], edited[id(ep)]
            if _contains(before, obj_i):
                present += 1
                followed += _contains(after, obj_j) and not _contains(after, obj_i)
                changed += after != before
        n = len(plans)
        return {
            "slot_surgery/object_swap": {
                "baseline_object_present_pct": 100.0 * present / n,
                "swap_follow_pct": 100.0 * followed / max(present, 1),
                "answer_changed_pct": 100.0 * changed / max(present, 1),
                "episodes": n,
            }
        }

    def _paraphrase_probe(self, model, device, lesson):
        if "lang_template" not in self._tracks():
            return {}
        cfg, rng, pairs = (
            difficulty_for("lang_template", lesson),
            random.Random(91_700 + lesson),
            [],
        )
        for _ in range(40 * self.diag_episodes):
            if len(pairs) == self.diag_episodes:
                break
            try:
                pairs.append(ltr.template_pair(cfg, rng))
            except OverflowError:
                continue
        if not pairs:
            return {}
        with contextlib.redirect_stdout(io.StringIO()):
            self._evaluate_episodes(model, device, [e for pair in pairs for e in pair])
        p, n = self._last_eval_preds, len(pairs)
        same = sum(p[id(a)] == p[id(b)] for a, b in pairs)
        both = sum(p[id(a)] == list(a.answer) and p[id(b)] == list(b.answer) for a, b in pairs)
        return {
            "paraphrase_consistency": {
                "consistency_pct": 100.0 * same / n,
                "both_correct_pct": 100.0 * both / n,
                "episodes": n,
            }
        }
