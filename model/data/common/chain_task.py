"""
file: data/common/chain_task.py

Generic "K facts, then M chained queries" recall+aggregation task -- shared
scaffolding for the text / audio / video dataset modules. Mirrors
data/graph_traversal/graph_traversal.py's overall shape (digit-encoded
fields, curriculum, evaluate_*, depth breakdown, OOD eval against a fixed
held-out table, memory-ablation eval, robustness eval) but factored so the
mechanical plumbing isn't re-derived three times.

What stays per-modality (the "treat each modality individually" seam, NOT
shared): the semantic meaning of a "fact"/"query" (see each of
data/text/text_dataset.py, data/audio/audio_dataset.py,
data/video/video_dataset.py), the OOD generalization axis
(_synthetic_ood_facts), and the robustness perturbation (perturb_fact).

Real-data hook (v2): `_fact_pool` / `_test_fact_pool` (None by default --
unchanged synthetic behavior). A subclass whose __init__ was given a real
dataset_link/test_dataset_link (see data/common/real_data.py) sets these to
a real (key,value) fact pool built from that source; build_episode() then
samples FROM that pool instead of drawing fresh random labels, and
build_ood_facts() uses the disjoint real test pool instead of the seeded
synthetic table -- the modality analogue of "train on synthetic graphs,
test on the held-out London Underground graph."

Task shape: present `num_facts` (key, value) pairs (digit-coded labels in
[0, label_range)), shuffled, first item phase-tagged (mirrors graph's own
convention exactly). Then `num_queries` query keys in sequence; each answer
step must predict:
  - `value`  : the fact value for that key (single-hop recall -- analogue of
               graph's dst|src+edge lookup check)
  - `cumsum` : (running sum of every value answered so far) mod label_range
               (forces genuine sequential state -- analogue of graph's
               src|prev_dst_ok chain check)
Depth axis = num_queries (this family's analogue of "hop count").
"""

from __future__ import annotations

import random

import torch

from data.base_dataset import BaseDataset
from data.common.digit_codec import DigitCodec, digit_field_diversity, digit_field_loss
from memory_manipulation.dynamic_memory_resize import resize_memory

NUM_FIELDS = 2  # [value, cumsum]
NUM_PHASE_CHANNELS = 2


def score_answer_steps(output, target_digits, answer_mask, codec, verbose=False, field_log=None):
    """Shared per-episode answer scoring: decode each answer step's (value,
    cumsum) fields, compare to target_digits, optionally print, optionally
    accumulate a per-(episode_depth, hop_position) breakdown. Used by both
    KVChainDataset.evaluate_chain (text/audio/video) and
    MultimodalDataset.evaluate_chain (data/multimodal/multimodal_dataset.py)
    so the two families can never silently drift in what counts as
    'correct', how depth/perfect-episode bookkeeping is derived, or how the
    field_log is keyed.
    -> (ep_total, ep_correct, episode_perfect, correct_value, correct_cumsum, n_steps)
    ep_total/ep_correct: this episode's step-count / correct-field-count
        (2 fields per step), for the caller's by_depth breakdown.
    correct_value/correct_cumsum/n_steps: per-field correct counts and step
        count, for the caller's running totals (separate value vs cumsum
        accuracy in the final summary print).
    field_log: optional dict, keyed by (depth, hop_position) -> [n, value_ok,
        cumsum_ok], filled in place. depth = this episode's total number of
        answer steps (its 'num_queries' -- the family's hop-count analogue),
        so this groups hop position WITHIN episodes of the same length,
        mirroring graph_traversal.py's (path_length, hop_position) keying.
        Lets a caller ask "does cumsum specifically degrade at later hop
        positions within an episode (a capacity/forgetting signal), or is it
        uniformly low across positions (a training-progress signal)?"
    """
    D = codec.label_dim
    nd = codec.num_digits
    answer_idx = (answer_mask == 1).nonzero(as_tuple=True)[0]
    depth = int(answer_idx.shape[0])
    ep_total = ep_correct = 0
    episode_perfect = True
    correct_value = correct_cumsum = n_steps = 0
    for hop_pos, idx in enumerate(answer_idx, start=1):
        v_pred = codec.decode_field(output[idx][0:D])
        c_pred = codec.decode_field(output[idx][D : 2 * D])
        td = target_digits[idx].tolist()
        v_tgt = int("".join(map(str, td[0:nd])))
        c_tgt = int("".join(map(str, td[nd : 2 * nd])))
        v_ok, c_ok = v_pred == v_tgt, c_pred == c_tgt
        correct_value += int(v_ok)
        correct_cumsum += int(c_ok)
        ep_correct += int(v_ok) + int(c_ok)
        n_steps += 1
        ep_total += 1
        if not (v_ok and c_ok):
            episode_perfect = False
        if verbose:
            print(
                f"  step {hop_pos}: value {v_pred}/{v_tgt} ok={v_ok} | cumsum {c_pred}/{c_tgt} ok={c_ok}"
            )
        if field_log is not None:
            fl = field_log.setdefault((depth, hop_pos), [0, 0, 0])
            fl[0] += 1
            fl[1] += int(v_ok)
            fl[2] += int(c_ok)
    return ep_total, ep_correct, episode_perfect, correct_value, correct_cumsum, n_steps


def write_value_cumsum_field_log(writer, file, step, lesson, eval_type, field_log):
    """Shared field_log writer for the value/cumsum family (KVChainDataset
    and MultimodalDataset both fill field_log via score_answer_steps above,
    so they share this same (depth, hop_position) -> [n, value_ok,
    cumsum_ok] schema and can share how it's dumped/printed)."""
    for (depth, hop_pos), (n, v_ok, c_ok) in sorted(field_log.items()):
        writer.writerow(
            [step, lesson, eval_type, depth, hop_pos, n, v_ok / n * 100, c_ok / n * 100]
        )
    file.flush()
    agg = {}
    for (_depth, hop_pos), v in field_log.items():
        a = agg.setdefault(hop_pos, [0, 0, 0])
        for i in range(3):
            a[i] += v[i]
    parts = []
    for hop_pos in sorted(agg)[:4]:
        n, v_ok, c_ok = agg[hop_pos]
        parts.append(
            f"hop{hop_pos}: value {v_ok / n * 100:.0f} cumsum {c_ok / n * 100:.0f} (n={n})"
        )
    print(f"    [field acc {eval_type} step {step}] " + " | ".join(parts))


def value_cumsum_field_log_header():
    return ["depth", "hop_position", "n_steps", "value_acc", "cumsum_acc"]


class ChainCurriculum:
    def __init__(self, dataset, table, lesson_nr_cells):
        assert len(table) == len(lesson_nr_cells)
        self.dataset = dataset
        self.table = table
        self.lesson = 0
        self.lesson_nr_cells = list(lesson_nr_cells)

    def maybe_advance(self, model, device, step=None, optimizer=None):
        nf, nq = self.table[self.lesson]
        combined_acc, perfect_frac, breakdown = self.dataset.evaluate_chain(
            model,
            device,
            num_episodes=self.dataset.eval_batch_size,
            num_facts_range=nf,
            num_queries_range=nq,
            step_breakdown=True,
            field_log=(field_log := {}),
        )
        if breakdown:
            print(
                f"    [lesson {self.lesson + 1} eval by depth] "
                + ", ".join(
                    f"{d}-step: acc {a:.1f}% perfect {p:.1f}% (n={n})"
                    for d, (a, p, n) in breakdown.items()
                )
            )
            hop_writer = getattr(self, "hop_log_writer", None)
            if hop_writer is not None:
                for d, (a, p, n) in sorted(breakdown.items()):
                    hop_writer.writerow([step, self.lesson + 1, "id", d, a, p, n])
                self.hop_log_file.flush()

        field_writer = getattr(self, "field_log_writer", None)
        if field_writer is not None and field_log:
            self.dataset.write_field_log(
                field_writer, self.field_log_file, step, self.lesson + 1, "id", field_log
            )

        if (
            combined_acc / 100.0 >= self.dataset.advance_threshold
            and self.lesson < len(self.table) - 1
        ):
            self.lesson += 1
            new_n = self.lesson_nr_cells[self.lesson]
            if new_n != self.lesson_nr_cells[self.lesson - 1]:
                resize_memory(model, new_n, device=device, optimizer=optimizer)
                print(f">>> Memory resized to nr_cells={new_n} for lesson {self.lesson + 1}")
            print(f">>> Curriculum advanced to lesson {self.lesson + 1}/{len(self.table)}")
            writer = getattr(self, "advance_log_writer", None)
            if writer is not None:
                writer.writerow([step, self.lesson + 1, len(self.table)])
                self.advance_log_file.flush()
        return self.lesson, combined_acc, perfect_frac


def _as_range(v):
    """Accept either a fixed int or an (lo, hi) range; normalize to a range."""
    return tuple(v) if isinstance(v, tuple | list) else (v, v)


class KVChainDataset(BaseDataset):
    """Parameterized by a few modality-specific hooks (see below). Concrete
    text/audio/video dataset classes only override `name`, the curriculum
    table/lesson_nr_cells, `_synthetic_ood_facts()`, and optionally
    `perturb_fact()` -- plus, when a real dataset_link is given, they set
    `_fact_pool`/`_test_fact_pool` in their own __init__ (see module
    docstring's "Real-data hook")."""

    advance_threshold = 0.85
    label_range = 1000
    codec = DigitCodec(num_digits=3, digit_base=10)
    input_dim = 2 * codec.label_dim + NUM_PHASE_CHANNELS  # 62
    output_dim = NUM_FIELDS * codec.label_dim  # 60
    eval_batch_size = 100
    old_lesson_mix_rate = 0.10
    ood_query_range = (3, 8)

    # Real-data hook (v2): None -> unchanged synthetic behavior everywhere
    # below. Set by a subclass's __init__ when a real dataset_link/
    # test_dataset_link was given.
    _fact_pool = None
    _test_fact_pool = None

    # ---- modality-specific hooks -------------------------------------------
    def build_ood_facts(self, n_facts: int = 20):
        """Fixed held-out fact table -- the analogue of graph's
        build_london_underground_eval(): unseen at training time,
        reproducible across runs. Uses the real, disjoint-key test pool
        when one was loaded (see module docstring); otherwise falls back to
        each subclass's own seeded synthetic table via
        _synthetic_ood_facts()."""
        if self._test_fact_pool is not None:
            rng = random.Random(999)
            pool = self._test_fact_pool
            return rng.sample(pool, min(n_facts, len(pool)))
        return self._synthetic_ood_facts(n_facts)

    def _synthetic_ood_facts(self, n_facts: int = 20):
        """Default synthetic seeded held-out table. Override per modality
        for a more domain-meaningful OOD story (see each of
        data/text/text_dataset.py etc.)."""
        rng = random.Random(1234)
        keys = rng.sample(range(self.label_range), n_facts)
        vals = [rng.randrange(self.label_range) for _ in range(n_facts)]
        return list(zip(keys, vals))

    def perturb_fact(self, key_vec: torch.Tensor, value_vec: torch.Tensor, rng, severity: float):
        """Robustness hook (Layer D). Default: flip a random one-hot
        digit-slot with prob=severity ("token corruption"). Override for a
        more domain-meaningful perturbation."""

        def _flip(vec):
            vec = vec.clone()
            if rng.random() < severity:
                nd, db = self.codec.num_digits, self.codec.digit_base
                pos = rng.randrange(nd)
                vec[pos * db : (pos + 1) * db] = 0.0
                vec[pos * db + rng.randrange(db)] = 1.0
            return vec

        return _flip(key_vec), _flip(value_vec)

    # ---- curriculum ---------------------------------------------------------
    def make_curriculum(self):
        return ChainCurriculum(self, self._table, self._lesson_nr_cells)

    # ---- episode construction ------------------------------------------------
    def _encode_fact(self, key, value, phase_transition, perturb=None, rng=None):
        key_vec = self.codec.encode_label(key)
        val_vec = self.codec.encode_label(value)
        if perturb:
            key_vec, val_vec = self.perturb_fact(key_vec, val_vec, rng or random, perturb)
        return torch.cat([key_vec, val_vec, torch.tensor([phase_transition, 0.0])])

    def _encode_query(self, key, phase_transition):
        return torch.cat(
            [
                self.codec.encode_label(key),
                torch.zeros(self.codec.label_dim),
                torch.tensor([phase_transition, 0.0]),
            ]
        )

    def _encode_answer(self, phase_transition):
        return torch.cat(
            [torch.zeros(2 * self.codec.label_dim), torch.tensor([phase_transition, 1.0])]
        )

    def build_episode(self, num_facts_range, num_queries_range, rng=None, perturb=None):
        rng = rng if rng is not None else random
        num_facts_range = _as_range(num_facts_range)
        num_queries_range = _as_range(num_queries_range)
        num_facts = rng.randint(*num_facts_range)
        if self._fact_pool is not None:
            # Real-data hook: sample real (key,value) facts from the loaded
            # pool instead of drawing fresh random labels every episode.
            num_facts = min(num_facts, len(self._fact_pool))
            facts = rng.sample(self._fact_pool, num_facts)
        else:
            keys = rng.sample(range(self.label_range), num_facts)
            vals = [rng.randrange(self.label_range) for _ in range(num_facts)]
            facts = list(zip(keys, vals))
        rng.shuffle(facts)
        num_queries = rng.randint(*num_queries_range)
        fact_keys = [k for k, _ in facts]
        query_keys = [rng.choice(fact_keys) for _ in range(num_queries)]

        NF = self.codec.num_digits
        inputs, target_digits, answer_mask = [], [], []
        for i, (k, v) in enumerate(facts):
            inputs.append(self._encode_fact(k, v, 1.0 if i == 0 else 0.0, perturb=perturb, rng=rng))
            target_digits.append([0] * (2 * NF))
            answer_mask.append(0)
        for i, k in enumerate(query_keys):
            inputs.append(self._encode_query(k, 1.0 if i == 0 else 0.0))
            target_digits.append([0] * (2 * NF))
            answer_mask.append(0)
        value_of, cumsum = dict(facts), 0
        for i, k in enumerate(query_keys):
            cumsum = (cumsum + value_of[k]) % self.label_range
            inputs.append(self._encode_answer(1.0 if i == 0 else 0.0))
            target_digits.append(
                self.codec.label_to_digit_targets(value_of[k])
                + self.codec.label_to_digit_targets(cumsum)
            )
            answer_mask.append(1)
        return (
            torch.stack(inputs),
            torch.tensor(target_digits, dtype=torch.long),
            torch.tensor(answer_mask, dtype=torch.float32),
            num_queries,
        )

    def collate(self, batch):
        max_len = max(len(x[0]) for x in batch)
        B = len(batch)
        padded_input = torch.zeros(B, max_len, self.input_dim)
        padded_targets = torch.zeros(
            B, max_len, NUM_FIELDS * self.codec.num_digits, dtype=torch.long
        )
        padded_mask = torch.zeros(B, max_len)
        for i, (inp, tgt, mask, _) in enumerate(batch):
            T = inp.size(0)
            padded_input[i, :T] = inp
            padded_targets[i, :T] = tgt
            padded_mask[i, :T] = mask
        return padded_input, padded_targets, padded_mask

    def sample_batch(self, curriculum, batch_size):
        episodes = []
        for _ in range(batch_size):
            idx = (
                random.randint(0, curriculum.lesson - 1)
                if curriculum.lesson > 0 and random.random() < self.old_lesson_mix_rate
                else curriculum.lesson
            )
            nf, nq = curriculum.table[idx]
            episodes.append(self.build_episode(nf, nq))
        return self.collate(episodes)

    # ---- loss / diversity -----------------------------------------------------
    def loss(self, output, target, mask):
        return digit_field_loss(output, target, mask, NUM_FIELDS, self.codec.digit_base)

    def diversity(self, output, target, mask):
        return digit_field_diversity(
            output, mask, NUM_FIELDS * self.codec.num_digits, self.codec.digit_base
        )

    def set_output_proj(self, proj):
        self._output_proj = proj

    # ---- evaluation -------------------------------------------------------------
    def _decode_answer(self, output_step):
        D = self.codec.label_dim
        return self.codec.decode_field(output_step[0:D]), self.codec.decode_field(
            output_step[D : 2 * D]
        )

    def evaluate_chain(
        self,
        model,
        device,
        num_episodes,
        num_facts_range=None,
        num_queries_range=None,
        fixed_facts=None,
        rng=None,
        ablate_memory=False,
        step_breakdown=False,
        perturb=None,
        verbose_n=0,
        skip_stages=None,
        field_log=None,
    ):
        model.eval()
        total = correct_value = correct_cumsum = perfect_episodes = tested = 0
        by_depth = {}
        _rng = rng or random
        with torch.no_grad():
            while tested < num_episodes:
                if fixed_facts is not None:
                    keys = [k for k, _ in fixed_facts]
                    vals_map = dict(fixed_facts)
                    nq = _rng.randint(*_as_range(num_queries_range or self.ood_query_range))
                    query_keys = [_rng.choice(keys) for _ in range(nq)]
                    facts = list(fixed_facts)
                    _rng.shuffle(facts)
                    NF = self.codec.num_digits
                    inputs, target_digits, answer_mask = [], [], []
                    for i, (k, v) in enumerate(facts):
                        inputs.append(
                            self._encode_fact(
                                k, v, 1.0 if i == 0 else 0.0, perturb=perturb, rng=_rng
                            )
                        )
                        target_digits.append([0] * (2 * NF))
                        answer_mask.append(0)
                    for i, k in enumerate(query_keys):
                        inputs.append(self._encode_query(k, 1.0 if i == 0 else 0.0))
                        target_digits.append([0] * (2 * NF))
                        answer_mask.append(0)
                    cumsum = 0
                    for i, k in enumerate(query_keys):
                        cumsum = (cumsum + vals_map[k]) % self.label_range
                        inputs.append(self._encode_answer(1.0 if i == 0 else 0.0))
                        target_digits.append(
                            self.codec.label_to_digit_targets(vals_map[k])
                            + self.codec.label_to_digit_targets(cumsum)
                        )
                        answer_mask.append(1)
                    ep = (
                        torch.stack(inputs),
                        torch.tensor(target_digits, dtype=torch.long),
                        torch.tensor(answer_mask, dtype=torch.float32),
                        nq,
                    )
                else:
                    ep = self.build_episode(
                        num_facts_range, num_queries_range, rng=_rng, perturb=perturb
                    )
                input_seq, target_digits, answer_mask, _ = ep
                input_seq = input_seq.unsqueeze(0).to(device)
                output, _ = model(
                    input_seq,
                    (None, None, None),
                    reset_experience=True,
                    pass_through_memory=not ablate_memory,
                    **({"combiner_skip_stages": skip_stages} if skip_stages else {}),
                )
                output = self._output_proj(output.transpose(0, 1).contiguous().squeeze(0))

                ep_total, ep_correct, episode_perfect, ep_cv, ep_cc, ep_n = score_answer_steps(
                    output,
                    target_digits,
                    answer_mask,
                    self.codec,
                    verbose=(tested < verbose_n),
                    field_log=field_log,
                )
                correct_value += ep_cv
                correct_cumsum += ep_cc
                total += ep_n
                perfect_episodes += int(episode_perfect)
                tested += 1
                if step_breakdown:
                    acc = by_depth.setdefault(ep_total, [0, 0, 0, 0])
                    acc[0] += 2 * ep_total
                    acc[1] += ep_correct
                    acc[2] += 1
                    acc[3] += int(episode_perfect)

        combined_acc = 100.0 * (correct_value + correct_cumsum) / max(2 * total, 1)
        perfect_frac = 100.0 * perfect_episodes / max(tested, 1)
        print(
            f"Eval [{self.name}]: value {100.0 * correct_value / max(total, 1):.2f}% | "
            f"cumsum {100.0 * correct_cumsum / max(total, 1):.2f}% | combined {combined_acc:.2f}% | "
            f"perfect {perfect_frac:.2f}% ({tested} episodes)"
        )
        model.train()
        if step_breakdown:
            breakdown = {
                d: (100.0 * c / max(t, 1), 100.0 * p / max(e, 1), e)
                for d, (t, c, e, p) in sorted(by_depth.items())
            }
            return combined_acc, perfect_frac, breakdown
        return combined_acc, perfect_frac

    def evaluate_ood(self, model, device, num_episodes, rng, verbose_n=0, field_log=None):
        return self.evaluate_chain(
            model,
            device,
            num_episodes,
            fixed_facts=self.build_ood_facts(),
            num_queries_range=self.ood_query_range,
            rng=rng,
            step_breakdown=True,
            verbose_n=verbose_n,
            field_log=field_log,
        )

    def evaluate_id_ablated(self, model, device, curriculum, lesson_idx):
        nf, nq = curriculum.table[lesson_idx]
        return self.evaluate_chain(
            model,
            device,
            self.eval_batch_size,
            num_facts_range=nf,
            num_queries_range=nq,
            ablate_memory=True,
        )

    def evaluate_id_combiner_stage_ablated(
        self, model, device, curriculum, lesson_idx, skip_stages
    ):
        nf, nq = curriculum.table[lesson_idx]
        return self.evaluate_chain(
            model,
            device,
            self.eval_batch_size,
            num_facts_range=nf,
            num_queries_range=nq,
            skip_stages=skip_stages,
        )

    def evaluate_robustness(self, model, device, curriculum, lesson_idx, perturbation, rng):
        nf, nq = curriculum.table[lesson_idx]
        severity = perturbation.get("severity", 0.2) if isinstance(perturbation, dict) else 0.2
        return self.evaluate_chain(
            model,
            device,
            self.eval_batch_size,
            num_facts_range=nf,
            num_queries_range=nq,
            rng=rng,
            perturb=severity,
        )

    def field_log_header(self):
        return value_cumsum_field_log_header()

    def write_field_log(self, writer, file, step, lesson, eval_type, field_log):
        write_value_cumsum_field_log(writer, file, step, lesson, eval_type, field_log)
