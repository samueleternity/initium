"""
file: data/multimodal/multimodal_dataset.py

Fuses two or more of {text, audio, video} chain datasets. The SAME (key,
value) facts/queries draw is shown through EACH active modality's own
encoding channel simultaneously, concatenated into one wider per-timestep
input vector (input_dim = sum of active modalities' input_dims). This is
what makes "which modality does the model actually use" answerable via a
direct ablation: zero one modality's channel slice and re-evaluate -
Delta_<modality>, the Dependency-layer (Layer C) analogue of
evaluate_id_ablated (memory) / evaluate_id_combiner_stage_ablated
(component), for modalities.

Only the FIRST listed modality's targets are scored ("primary") -- this
tests whether auxiliary modalities help/are relied upon for the primary
modality's own recall+chain task, not multi-task multi-output prediction.
"""
import random

import torch

from data.base_dataset import BaseDataset
from data.common.chain_task import ChainCurriculum
from data.text.text_dataset import TextChainDataset
from data.audio.audio_dataset import AudioChainDataset
from data.video.video_dataset import VideoChainDataset

_MODALITY_CLASSES = {"text": TextChainDataset, "audio": AudioChainDataset, "video": VideoChainDataset}


class MultimodalDataset(BaseDataset):
    # NOTE: real-data dataset_link/test_dataset_link (see data/common/real_data.py)
    # are not wired for multimodal yet -- each sub-dataset here is always built
    # synthetic-only (dataset_link is reserved for the '+'-joined modality list).
    # Wiring real per-modality sources through is a natural follow-up once the
    # per-modality real-data pipelines above are validated individually.
    def __init__(self, modalities):
        modalities = [m.strip().lower() for m in modalities if m.strip()]
        if len(modalities) < 2:
            raise ValueError(f"MultimodalDataset needs >=2 modalities, got {modalities!r}")
        unknown = [m for m in modalities if m not in _MODALITY_CLASSES]
        if unknown:
            raise ValueError(f"MultimodalDataset: unknown modalities {unknown}, "
                              f"expected a subset of {sorted(_MODALITY_CLASSES)}")
        self.modalities = modalities
        self.subs = [_MODALITY_CLASSES[m]() for m in modalities]
        self.primary = self.subs[0]
        self.name = "multimodal[" + "+".join(modalities) + "]"

        off = 0
        self._offsets = []
        for s in self.subs:
            self._offsets.append((off, off + s.input_dim)); off += s.input_dim
        self.input_dim = off
        self.output_dim = self.primary.output_dim
        self.eval_batch_size = self.primary.eval_batch_size
        self.advance_threshold = self.primary.advance_threshold
        self._table = self.primary._table
        self._lesson_nr_cells = self.primary._lesson_nr_cells

    def make_curriculum(self):
        return ChainCurriculum(self, self._table, self._lesson_nr_cells)

    def set_output_proj(self, proj):
        self._output_proj = proj

    def _build_fused_episode(self, num_facts_range, num_queries_range, rng=None,
                              skip_modalities=None, perturb=None):
        rng = rng if rng is not None else random
        skip_modalities = skip_modalities or set()
        codec, label_range = self.primary.codec, self.primary.label_range
        NF = codec.num_digits

        num_facts = rng.randint(*num_facts_range)
        keys = rng.sample(range(label_range), num_facts)
        vals = [rng.randrange(label_range) for _ in range(num_facts)]
        facts = list(zip(keys, vals)); rng.shuffle(facts)
        num_queries = rng.randint(*num_queries_range)
        query_keys = [rng.choice(keys) for _ in range(num_queries)]

        def fused(build_fns):
            return torch.cat([torch.zeros(sub.input_dim) if i in skip_modalities else build_fns[i]()
                               for i, sub in enumerate(self.subs)])

        inputs, target_digits, answer_mask = [], [], []
        for i, (k, v) in enumerate(facts):
            pt = 1.0 if i == 0 else 0.0
            inputs.append(fused([(lambda s=sub: s._encode_fact(k, v, pt, perturb=perturb, rng=rng))
                                  for sub in self.subs]))
            target_digits.append([0] * (2 * NF)); answer_mask.append(0)
        for i, k in enumerate(query_keys):
            pt = 1.0 if i == 0 else 0.0
            inputs.append(fused([(lambda s=sub: s._encode_query(k, pt)) for sub in self.subs]))
            target_digits.append([0] * (2 * NF)); answer_mask.append(0)
        value_of, cumsum = dict(facts), 0
        for i, k in enumerate(query_keys):
            cumsum = (cumsum + value_of[k]) % label_range
            pt = 1.0 if i == 0 else 0.0
            inputs.append(fused([(lambda s=sub: s._encode_answer(pt)) for sub in self.subs]))
            target_digits.append(codec.label_to_digit_targets(value_of[k]) + codec.label_to_digit_targets(cumsum))
            answer_mask.append(1)
        return (torch.stack(inputs), torch.tensor(target_digits, dtype=torch.long),
                torch.tensor(answer_mask, dtype=torch.float32), num_queries)

    def sample_batch(self, curriculum, batch_size):
        episodes = []
        for _ in range(batch_size):
            idx = (random.randint(0, curriculum.lesson - 1)
                   if curriculum.lesson > 0 and random.random() < 0.10 else curriculum.lesson)
            nf, nq = curriculum.table[idx]
            episodes.append(self._build_fused_episode(nf, nq))
        return self.primary.collate(episodes)

    def loss(self, output, target, mask):
        return self.primary.loss(output, target, mask)

    def diversity(self, output, target, mask):
        return self.primary.diversity(output, target, mask)

    def evaluate_chain(self, model, device, num_episodes, num_facts_range=None, num_queries_range=None,
                        rng=None, ablate_memory=False, step_breakdown=False, skip_modalities=None,
                        verbose_n=0):
        model.eval()
        total = correct = perfect_episodes = tested = 0
        by_depth = {}
        codec = self.primary.codec
        with torch.no_grad():
            while tested < num_episodes:
                input_seq, target_digits, answer_mask, depth = self._build_fused_episode(
                    num_facts_range, num_queries_range, rng=rng, skip_modalities=skip_modalities)
                input_seq = input_seq.unsqueeze(0).to(device)
                output, _ = model(input_seq, (None, None, None), reset_experience=True,
                                   pass_through_memory=not ablate_memory)
                output = self._output_proj(output.transpose(0, 1).contiguous().squeeze(0))
                answer_idx = (answer_mask == 1).nonzero(as_tuple=True)[0]
                ep_total = ep_correct = 0
                episode_perfect = True
                D, nd = codec.label_dim, codec.num_digits
                for idx in answer_idx:
                    v_pred = codec.decode_field(output[idx][0:D])
                    c_pred = codec.decode_field(output[idx][D:2 * D])
                    td = target_digits[idx].tolist()
                    v_tgt = int("".join(map(str, td[0:nd]))); c_tgt = int("".join(map(str, td[nd:2 * nd])))
                    v_ok, c_ok = v_pred == v_tgt, c_pred == c_tgt
                    correct += int(v_ok) + int(c_ok); total += 1; ep_total += 1
                    ep_correct += int(v_ok) + int(c_ok)
                    if not (v_ok and c_ok):
                        episode_perfect = False
                perfect_episodes += int(episode_perfect); tested += 1
                if step_breakdown:
                    acc = by_depth.setdefault(ep_total, [0, 0, 0, 0])
                    acc[0] += 2 * ep_total; acc[1] += ep_correct; acc[2] += 1; acc[3] += int(episode_perfect)
        combined_acc = 100.0 * correct / max(2 * total, 1)
        perfect_frac = 100.0 * perfect_episodes / max(tested, 1)
        model.train()
        if step_breakdown:
            breakdown = {d: (100.0 * c / max(t, 1), 100.0 * p / max(e, 1), e)
                         for d, (t, c, e, p) in sorted(by_depth.items())}
            return combined_acc, perfect_frac, breakdown
        return combined_acc, perfect_frac

    def evaluate_ood(self, model, device, num_episodes, rng, verbose_n=0, field_log=None):
        return self.evaluate_chain(model, device, num_episodes, num_facts_range=(15, 20),
                                    num_queries_range=self.primary.ood_query_range, rng=rng, step_breakdown=True)

    def evaluate_id_ablated(self, model, device, curriculum, lesson_idx):
        nf, nq = curriculum.table[lesson_idx]
        return self.evaluate_chain(model, device, self.eval_batch_size,
                                    num_facts_range=nf, num_queries_range=nq, ablate_memory=True)

    def evaluate_modality_ablated(self, model, device, curriculum, lesson_idx, modality: str):
        """Delta_<modality> = full_acc - ablated_acc, per the research doc's
        modality-dependency proposal."""
        idx = self.modalities.index(modality)
        nf, nq = curriculum.table[lesson_idx]
        return self.evaluate_chain(model, device, self.eval_batch_size,
                                    num_facts_range=nf, num_queries_range=nq, skip_modalities={idx})

    def write_field_log(self, writer, file, step, lesson, eval_type, field_log):
        pass