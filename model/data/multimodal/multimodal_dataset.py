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

Only the FIRST listed modality's targets are scored ("primary") - this
tests whether auxiliary modalities help/are relied upon for the primary
modality's own recall+chain task, not multi-task multi-output prediction.

Real-data wiring: each modality in the `modalities` list passed to
__init__ may be a bare name (synthetic, unchanged default) or
"modality:path" to source that modality's own real fact pool from `path`
via its existing single-modality real-data pipeline (data/common/
real_data.py) -- see _parse_modality_specs() below. Repeating the SAME
path across modalities (e.g. "video:clip.mp4+audio:clip.mp4") draws both
modalities' real content from one shared file, such as a video that also
carries an audio track -- the flagship "upload one video, learn its
picture AND its soundtrack" case. See MultimodalDataset.__init__'s
`link_role` for how training vs. inference (MultimodalTask) route a
spec's path to the right underlying *ChainDataset kwarg.
"""
import random

import torch

from data.base_dataset import BaseDataset
from data.common.chain_task import ChainCurriculum
from data.text.text_dataset import TextChainDataset
from data.audio.audio_dataset import AudioChainDataset
from data.video.video_dataset import VideoChainDataset

_MODALITY_CLASSES = {"text": TextChainDataset, "audio": AudioChainDataset, "video": VideoChainDataset}


def _parse_modality_specs(modalities):
    """Each entry of `modalities` is either a bare modality name ("text",
    "audio", "video") -- synthetic facts, unchanged default -- or
    "modality:path" to source that modality's OWN real (key,value) fact
    pool from `path`, via the exact same per-modality real-data pipeline
    single-modality datasets already use (data/common/real_data.py). The
    SAME path can be given to more than one modality -- e.g.
    "video:clip.mp4+audio:clip.mp4" -- to draw both modalities' real
    content from one shared file, such as a video that also carries an
    audio track: video_token_stream (opencv) and audio_token_stream
    (torchaudio) each open the container with their own library and only
    ever read the stream they care about, so pointing both at the same
    path is safe and is the intended way to say "learn from this video's
    picture AND its soundtrack." -> [(name, path_or_None), ...].
    """
    parsed = []
    for spec in modalities:
        spec = (spec or "").strip()
        if not spec:
            continue
        name, _, path = spec.partition(":")
        parsed.append((name.strip().lower(), path.strip() or None))
    return parsed


class MultimodalDataset(BaseDataset):
    # Real-data wiring: see _parse_modality_specs() above for the
    # "modality[:path]" spec syntax and MultimodalTask (inference/tasks/
    # multimodal_task.py) for how --dataset-link reaches here at inference.
    def __init__(self, modalities, link_role: str = "dataset_link"):
        """
        modalities: iterable of modality specs (see _parse_modality_specs).
        link_role: "dataset_link" (default; training-side) -- a spec's path
            becomes that modality's OWN training source (auto-split into a
            disjoint train/test pool by that modality's dataset class,
            exactly like a standalone *ChainDataset(dataset_link=...)), or
            "test_dataset_link" (inference-side, see MultimodalTask) -- a
            spec's path becomes a fixed, already-real TEST pool directly.
        """
        if link_role not in ("dataset_link", "test_dataset_link"):
            raise ValueError("MultimodalDataset: link_role must be 'dataset_link' "
                              f"or 'test_dataset_link', got {link_role!r}")
        specs = _parse_modality_specs(modalities)
        if len(specs) < 2:
            raise ValueError(f"MultimodalDataset needs >=2 modalities, got {modalities!r}")
        names = [n for n, _ in specs]
        unknown = [n for n in names if n not in _MODALITY_CLASSES]
        if unknown:
            raise ValueError(f"MultimodalDataset: unknown modalities {unknown}, "
                              f"expected a subset of {sorted(_MODALITY_CLASSES)}")
        self.modalities = names
        self.subs = [_MODALITY_CLASSES[name](**({link_role: path} if path else {}))
                     for name, path in specs]
        self.primary = self.subs[0]
        self.name = "multimodal[" + "+".join(names) + "]"

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

    def _facts_pool(self, use_test_pool: bool = False):
        """Real (key,value) pool to draw a fused episode from, or None for
        the fully-synthetic draw (unchanged default when no modality was
        given a real source). use_test_pool selects each sub's held-out
        `_test_fact_pool` (evaluate_ood) vs its training `_fact_pool`
        (sample_batch / evaluate_id_ablated / evaluate_modality_ablated) --
        same train/test convention every single-modality dataset already
        uses. Values always come from the PRIMARY modality's pool (only the
        primary is scored -- see class docstring); keys are restricted to
        ones ALSO present in every OTHER real-sourced modality's pool, so
        any channel built from real data genuinely has real content for
        every key an episode can draw (e.g. a video file's frame-derived
        pool intersected with that SAME file's audio-track-derived pool).
        Falls back to the unrestricted primary pool if the intersection is
        empty, rather than raising mid-run."""
        attr = "_test_fact_pool" if use_test_pool else "_fact_pool"
        primary_pool = getattr(self.primary, attr)
        if primary_pool is None:
            return None
        other_pools = [p for p in (getattr(s, attr) for s in self.subs[1:]) if p is not None]
        if not other_pools:
            return primary_pool
        shared_keys = set(k for k, _ in primary_pool)
        for p in other_pools:
            shared_keys &= set(k for k, _ in p)
        restricted = [(k, v) for k, v in primary_pool if k in shared_keys]
        return restricted or primary_pool

    def make_curriculum(self):
        return ChainCurriculum(self, self._table, self._lesson_nr_cells)

    def set_output_proj(self, proj):
        self._output_proj = proj

    def _build_fused_episode(self, num_facts_range, num_queries_range, rng=None,
                              skip_modalities=None, perturb=None, use_test_pool=False):
        rng = rng if rng is not None else random
        skip_modalities = skip_modalities or set()
        codec, label_range = self.primary.codec, self.primary.label_range
        NF = codec.num_digits

        num_facts = rng.randint(*num_facts_range)
        pool = self._facts_pool(use_test_pool=use_test_pool)
        if pool is not None:
            num_facts = min(num_facts, len(pool))
            facts = rng.sample(pool, num_facts)
        else:
            keys = rng.sample(range(label_range), num_facts)
            vals = [rng.randrange(label_range) for _ in range(num_facts)]
            facts = list(zip(keys, vals))
        rng.shuffle(facts)
        keys = [k for k, _ in facts]
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
                        verbose_n=0, use_test_pool=False):
        model.eval()
        total = correct = perfect_episodes = tested = 0
        by_depth = {}
        codec = self.primary.codec
        with torch.no_grad():
            while tested < num_episodes:
                input_seq, target_digits, answer_mask, depth = self._build_fused_episode(
                    num_facts_range, num_queries_range, rng=rng, skip_modalities=skip_modalities,
                    use_test_pool=use_test_pool)
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
                                    num_queries_range=self.primary.ood_query_range, rng=rng,
                                    step_breakdown=True, use_test_pool=True)

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