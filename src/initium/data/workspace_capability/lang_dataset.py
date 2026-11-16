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
    EOS,
    FACT_BASE,
    FACT_COUNT,
    RELATION_BASE,
    RELATION_COUNT,
    validate_episode,
)
from initium.data.workspace_capability.tracks import _dev_lexicon
from initium.data.workspace_capability.tracks import _lang_grammar as lg
from initium.data.workspace_capability.tracks import _lang_text as lt
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
_SLOT_KINDS = ("subject", "object", "relation", "word")
_SLOT_RANGES = ((24, 32, 0), (32, 40, 1), (48, 56, 2))


def _contains(seq, sub):
    n = len(sub)
    return n > 0 and any(seq[i : i + n] == sub for i in range(len(seq) - n + 1))


class LanguageDataset(WorkspaceCapabilityDataset):
    """Language tracks lang_template/nlg/story_qa/dialogue/syntax_probes/open and the lang_all mix."""

    def __init__(self, task_track: str = "lang_all", *, batch_size_hint: int = 16):
        super().__init__(task_track, batch_size_hint=batch_size_hint)
        self.input_dim = self.output_dim = lt.tokenizer().total
        self.diag_episodes = 48  # episodes per slot-probe / surgery / paraphrase diagnostic
        self.generation_eval_per_group = 32
        self.generation_max_tokens = 64
        self._lang_ready = True
        self._weighted_sampling = True

    @staticmethod
    def slot_text_columns():
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
        return lt.available_tracks() if self.task_track == "lang_all" else (self.task_track,)

    @staticmethod
    def _curriculum_groups(track, lesson):
        return tuple(difficulty_for(track, lesson)["groups"])

    def _make_episode(self, track, level, rng, *, ood=False, ood_axes=None, generalization_axis=None):
        if track == "lang_all":
            pool = self._tracks()
            track = pool[min(int(rng.random() * len(pool)), len(pool) - 1)]
        cfg = difficulty_for(track, level, ood=ood, ood_axes=ood_axes)
        generate = _GEN[track]
        for _ in range(_RETRIES):
            try:
                return validate_episode(generate(cfg, rng))
            except OverflowError:  # generator-signalled resample (length/feasibility)
                continue
        raise RuntimeError(f"{track}: no valid episode after {_RETRIES} attempts at level {level}")

    # ------------------------------------------------------------------ workspace metadata
    def workspace_entity_address_metadata(self, input_vectors, *, episodes=None, **_):
        examples = self.last_batch_examples if episodes is None else episodes
        batch, time = input_vectors.shape[:2]
        if len(examples) != batch:
            raise RuntimeError("language address metadata lost batch/episode alignment")
        rows, pos, keys, writes = [], [], [], []
        for row, episode in enumerate(examples):
            for p, key, write in episode.metadata.get("lang_addr", ()):
                if p < time:
                    rows.append(row)
                    pos.append(p)
                    keys.append(key)
                    writes.append(write)
        ids = torch.full((batch, time), -1, dtype=torch.long)
        mask = torch.zeros((batch, time), dtype=torch.bool)
        if rows:
            r, p = torch.tensor(rows), torch.tensor(pos)
            ids[r, p] = torch.tensor(keys)
            mask[r, p] = torch.tensor(writes)
        device = input_vectors.device
        return {"entity_ids": ids.to(device), "entity_write_mask": mask.to(device)}

    def workspace_key_value_metadata(self, input_vectors, episodes=None, *, include_adaptive_v3=False, **_):
        examples = self.last_batch_examples if episodes is None else episodes
        batch, time = input_vectors.shape[:2]
        if len(examples) != batch:
            raise RuntimeError("language key-value metadata lost batch/episode alignment")
        cap = KEY_VALUE_OPERAND_CAPACITY
        used: dict[tuple[int, int], int] = {}
        rr, ss, cc, kk, tt = [], [], [], [], []
        for row, episode in enumerate(examples):
            limit = min(time, len(episode.inputs))
            md = episode.metadata
            for step, key, target in (*md.get("workspace_event_writes", ()), *md.get("workspace_query_reads", ())):
                col = used.get((row, step), 0)
                if step >= limit or col >= cap:
                    continue
                used[(row, step)] = col + 1
                rr.append(row), ss.append(step), cc.append(col), kk.append(key), tt.append(target)
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
            "key_value_steps": tuple(sorted(set(ss))),
            "key_value_output_programs": (),
            "key_value_argmatch_locations": {},
        }
        if include_adaptive_v3:  # language tracks carry no operator programs: V3/V4 stay inert
            result["adaptive_operator_candidate_mask"] = torch.zeros((batch, time), dtype=torch.bool)
            result["adaptive_operator_execution_mask"] = torch.ones((batch, time), dtype=torch.bool)
            result["adaptive_operator_steps"] = ()
            result["adaptive_operator_teacher_steps"] = ()
        return result

    # ------------------------------------------------------------------ display
    @staticmethod
    def _fmt_answer(e, pred):
        kind = e.metadata.get("answer_kind", "text")

        def fmt(tokens):
            toks = [int(t) for t in tokens]
            toks = toks[: toks.index(EOS)] if EOS in toks else toks
            if kind == "bool":
                return " ".join("yes" if t == BOOL_TRUE else "no" if t == BOOL_FALSE else "?" for t in toks)
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
        # label / length / combo / combined splits; 'lang_all' resolves its track per paired seed
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
        return super()._generation_metrics(model, device, picked, cap=cap) if picked else {}

    def evaluate_curriculum_id(self, model, device, lesson_idx, *, field_log=None):
        result = super().evaluate_curriculum_id(model, device, lesson_idx, field_log=field_log)
        self._track_weights = {
            t: 0.02 if t == "lang_open" else 0.02 + max(0.0, 0.9 - v["exact_match"] / 100.0)
            for t, v in self.last_eval_diagnostics.get("tracks", {}).items()
        }
        diagnostics, episodes, preds = self.last_eval_diagnostics, self._last_id_eval_episodes, self._last_eval_preds
        self._print_samples(episodes, "id")
        diagnostics["stage3_counterfactual"] = self._stage3_metrics(episodes)
        extra: dict = {}
        try:
            if self.generation_eval:
                diagnostics["production_generation"] = {"all": self._generation_metrics(model, device, episodes)}
            workspace = getattr(model, "workspace", None)
            if workspace is not None and getattr(model, "workspace_semantic_entity_addressing", False):
                if getattr(model, "workspace_key_value_readout_enabled", False):
                    extra.update(self._slot_probe(model, device, episodes))
                extra.update(self._slot_surgery(model, device, int(lesson_idx)))
            extra.update(self._paraphrase_probe(model, device, int(lesson_idx)))
        except (RuntimeError, ValueError, KeyError) as exc:  # diagnostics must not kill a long run
            print(f"[lang-diagnostics] skipped: {exc!r}")
        diagnostics["production"] = extra
        self.last_eval_diagnostics, self._last_eval_preds = diagnostics, preds
        return result

    # ------------------------------------------------------------------ section-8 diagnostics
    def _probe_forward(self, model, device, episodes, on_batch, make_observer=None, want_kv=False):
        """Microbatched eval forward; calls on_batch(group, answer_preds, kv_meta, kv_preds)."""
        workspace = model.workspace
        prior, was_training = getattr(workspace, "_initium_recurrent_state_observer", None), model.training
        model.eval()
        try:
            with torch.no_grad():
                for off in range(0, len(episodes), self.batch_size_hint):
                    group = episodes[off : off + self.batch_size_hint]
                    x, _targets, answer_mask = self._collate(group, token_ids_only=True)
                    kwargs = {
                        k: v.to(device) for k, v in self.workspace_entity_address_metadata(x, episodes=group).items()
                    }
                    kv = self.workspace_key_value_metadata(x, group) if want_kv else None
                    if kv is not None:
                        kwargs.update(
                            key_value_entity_ids=kv["key_value_entity_ids"].to(device),
                            key_value_mask=kv["key_value_mask"].to(device),
                            key_value_steps=kv["key_value_steps"],
                        )
                    workspace._initium_recurrent_state_observer = make_observer(group) if make_observer else None
                    output, state = model(x.to(device), (None, None, None), reset_experience=True, **kwargs)
                    features = output.transpose(0, 1)
                    logits = self._output_proj(features)
                    if getattr(model, "workspace_slot_readout_enabled", False):
                        logits = model.apply_slot_readout(logits, features, state, answer_mask.to(device))
                    tokens = logits.argmax(-1).cpu().tolist()
                    preds = [t[len(e.inputs) : len(e.inputs) + len(e.answer)] for e, t in zip(group, tokens)]
                    on_batch(group, preds, kv, workspace.pop_key_value_predictions() if want_kv else None)
        finally:
            workspace._initium_recurrent_state_observer = prior
            model.train(was_training)

    def _slot_probe(self, model, device, episodes):
        """Slot-content decode accuracy at write time and at the final query (retention)."""
        eps = [e for e in episodes if e.metadata.get("workspace_event_writes")][: self.diag_episodes]
        if not eps:
            return {}
        stats: dict[tuple[str, str], list[int]] = defaultdict(lambda: [0, 0])

        def tally(group, _preds, kv, kv_preds):
            last = torch.tensor([len(e.inputs) - 1 for e in group]).unsqueeze(-1)
            for step, logits in kv_preds:
                mask, keys = kv["key_value_mask"][:, step], kv["key_value_entity_ids"][:, step]
                hit = logits.argmax(-1).cpu().eq(kv["key_value_targets"][:, step])
                retention = last.eq(step).expand_as(mask)
                kinds = torch.full_like(keys, 3)
                for lo, hi, idx in _SLOT_RANGES:
                    kinds[(keys >= lo) & (keys < hi)] = idx
                for phase, flag in (("write", False), ("retention", True)):
                    in_phase = mask & (retention == flag)
                    for idx, name in enumerate(_SLOT_KINDS):
                        sel = in_phase & kinds.eq(idx)
                        stats[(phase, name)][0] += int((hit & sel).sum())
                        stats[(phase, name)][1] += int(sel.sum())
                    if not flag:
                        for i in range(8):
                            sel = in_phase & kinds.lt(3) & (keys % 8).eq(i)
                            stats[(phase, f"fact_{i}")][0] += int((hit & sel).sum())
                            stats[(phase, f"fact_{i}")][1] += int(sel.sum())

        self._probe_forward(model, device, eps, tally, want_kv=True)
        return {
            f"slot_probe_{phase}/{name}": {"accuracy": 100.0 * h / n, "episodes": n}
            for (phase, name), (h, n) in sorted(stats.items())
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
                    ia, ib = ((slot == ka) & occupied).nonzero(), ((slot == kb) & occupied).nonzero()
                    if ia.numel() and ib.numel():
                        ia, ib = ia[0, 0], ib[0, 0]
                        swap = state[row, ia, address_dim:].clone()
                        state[row, ia, address_dim:] = state[row, ib, address_dim:]
                        state[row, ib, address_dim:] = swap

            return observe

        base, edited = {}, {}
        sink = lambda store: lambda group, preds, *_: store.update({id(e): p for e, p in zip(group, preds)})  # noqa: E731
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
        cfg, rng, pairs = difficulty_for("lang_template", lesson), random.Random(91_700 + lesson), []
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