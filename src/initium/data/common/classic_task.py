"""Long-window next-token training with explicit, memory-dependent recall probes.

This is an additive task family. The existing KV-chain curriculum and its
training/evaluation contract remain unchanged.
"""

from __future__ import annotations

import random
import numpy as np
import torch

from initium.data.base_dataset import BaseDataset
from initium.data.common.digit_codec import DigitCodec, digit_field_loss
from initium.config.classic_config import (
    CLASSIC_EVAL_EPISODES,
    CLASSIC_OOD_EVAL_EPISODES,
    CLASSIC_PROBE_DISTANCES,
    CLASSIC_PROBE_GAMMA,
    CLASSIC_WINDOW_SIZE,
)
from initium.data.common.real_data import (
    audio_token_stream_chunks,
    resolve_link_paths,
    text_token_stream,
    text_token_streams,
    video_token_stream,
)

TOKEN_COUNT = 1000


def _parse_specs(spec: str | None, modalities: list[str]) -> dict[str, str]:
    if not spec:
        return {}
    if len(modalities) == 1:
        return {modalities[0]: spec}
    result = {}
    for item in spec.split("+"):
        modality, sep, path = item.partition(":")
        if not sep or modality.strip() not in modalities or not path.strip():
            raise ValueError(
                "multimodal-classic links must use modality:path entries joined with '+', "
                "for example text:corpus.txt+audio:clip.wav"
            )
        result[modality.strip()] = path.strip()
    if set(result) != set(modalities):
        raise ValueError(f"provide exactly one source for each modality: {modalities}")
    return result


def _read_streams(modality: str, link: str) -> list[np.ndarray]:
    paths = resolve_link_paths(link)
    streams = []
    for path in paths:
        if modality == "text":
            tokens = text_token_stream(path, TOKEN_COUNT)
        elif modality == "audio":
            chunks = list(audio_token_stream_chunks(path, TOKEN_COUNT))
            tokens = np.concatenate(chunks) if chunks else np.empty(0, dtype=np.int64)
        elif modality == "video":
            tokens = video_token_stream(path, TOKEN_COUNT)
        else:
            raise ValueError(f"unsupported classic modality {modality!r}")
        stream = np.asarray(tokens, dtype=np.int64)
        if stream.size >= 2:
            streams.append(stream)
    if not streams:
        raise ValueError(f"{modality} source {link!r} produced no usable token stream")
    return streams


class ClassicCurriculum:
    """One fixed long-window lesson; it does not resize memory or mutate BaseDataset."""

    def __init__(self, dataset):
        self.dataset = dataset
        self.lesson = 0
        self.table = [(dataset.window_size, dataset.window_size)]
        self.lesson_nr_cells = [128]

    def maybe_advance(self, model, device, step=None, optimizer=None):
        self.last_eval_field_log = {}
        accuracy, perfect, _ = self.dataset.evaluate(
            model, device, self.dataset.train_sequences, self.dataset.eval_episodes,
            random.Random(8100 + self.lesson), field_log=self.last_eval_field_log,
        )
        return self.lesson, accuracy, perfect


class ClassicDataset(BaseDataset):
    """Real-data language/audio/video windows with prediction and recall losses.

    Inputs encode each token with the existing three-digit one-hot codec
    (30 values) plus a probe marker per modality, avoiding a 1000-wide
    one-hot feature vector. Targets/masks have two columns: next-token
    prediction and recall probe.
    """

    classic_track = True
    advance_threshold = 0.0

    def __init__(
        self,
        modalities: list[str],
        dataset_link: str | None,
        test_dataset_link: str | None = None,
        window_size: int = CLASSIC_WINDOW_SIZE,
        probe_distances: tuple[int, ...] = CLASSIC_PROBE_DISTANCES,
        probe_gamma: float = CLASSIC_PROBE_GAMMA,
        eval_episodes: int = CLASSIC_EVAL_EPISODES,
        ood_eval_episodes: int = CLASSIC_OOD_EVAL_EPISODES,
    ):
        self.modalities = list(modalities)
        if not self.modalities or any(m not in ("text", "audio", "video") for m in self.modalities):
            raise ValueError(f"classic modalities must be text/audio/video, got {modalities!r}")
        self.window_size = int(window_size)
        self.probe_distances = tuple(sorted({int(d) for d in probe_distances}))
        if self.window_size < 2 or not self.probe_distances or min(self.probe_distances) < 1:
            raise ValueError("window size must be >=2 and probe distances must be positive")
        if max(self.probe_distances) >= self.window_size:
            raise ValueError("each probe distance must be smaller than the classic window size")
        self.probe_gamma = float(probe_gamma)
        self.eval_episodes = int(eval_episodes)
        self.ood_eval_episodes = int(ood_eval_episodes)
        self.name = "multimodal-classic" if len(modalities) > 1 else f"{modalities[0]}-classic"
        self.codec = DigitCodec(num_digits=3, digit_base=10)
        self.input_dim = len(modalities) * (self.codec.label_dim + 1)
        self.output_dim = self.codec.label_dim
        self._output_proj = None
        self._build_token_encodings()

        train_links = _parse_specs(dataset_link, self.modalities)
        test_links = _parse_specs(test_dataset_link, self.modalities) if test_dataset_link else {}
        if not train_links:
            raise ValueError(f"{self.name} requires --dataset-link with real data")
        self.train_sequences, self.test_sequences = [], []
        for modality in self.modalities:
            if modality == "text" and modality in test_links:
                train_paths = resolve_link_paths(train_links[modality])
                test_paths = resolve_link_paths(test_links[modality])
                train_streams, test_streams = text_token_streams(
                    train_paths, test_paths, TOKEN_COUNT
                )
                self.train_sequences.append([np.asarray(s, dtype=np.int64) for s in train_streams])
                self.test_sequences.append([np.asarray(s, dtype=np.int64) for s in test_streams])
                continue
            sources = _read_streams(modality, train_links[modality])
            if modality in test_links:
                self.train_sequences.append(sources)
                self.test_sequences.append(_read_streams(modality, test_links[modality]))
            else:
                train_parts, test_parts = [], []
                for stream in sources:
                    split = int(stream.size * 0.8)
                    if min(split, stream.size - split) < self.window_size:
                        continue
                    train_parts.append(stream[:split])
                    test_parts.append(stream[split:])
                if not train_parts or not test_parts:
                    raise ValueError(
                        f"{modality} source is too short for a disjoint train/test window split; "
                        "provide a longer source or a separate --test-dataset-link"
                    )
                self.train_sequences.append(train_parts)
                self.test_sequences.append(test_parts)
        self._last_loss_terms = {"predict": 0.0, "probe": 0.0}

    def _build_token_encodings(self):
        token_ids = torch.arange(TOKEN_COUNT)
        places = torch.tensor(
            [self.codec.digit_base**i for i in reversed(range(self.codec.num_digits))]
        )
        token_digits = (token_ids[:, None] // places[None, :]) % self.codec.digit_base
        self._token_encodings = torch.zeros(TOKEN_COUNT, self.codec.label_dim)
        digit_indexes = torch.arange(self.codec.num_digits) * self.codec.digit_base
        self._token_encodings[
            torch.arange(TOKEN_COUNT)[:, None], digit_indexes[None, :] + token_digits
        ] = 1.0

    @property
    def _channel_size(self):
        return self.codec.label_dim + 1

    def make_curriculum(self):
        return ClassicCurriculum(self)

    def configure(self, window_size=None, probe_distances=None, probe_gamma=None,
                  ood_eval_episodes=None):
        """Apply runtime settings to a loaded prepared dataset.

        Prepared data stores token streams; window length, probe selection,
        and loss weighting are runtime choices and should remain overridable.
        """
        window_size = self.window_size if window_size is None else int(window_size)
        probe_distances = self.probe_distances if probe_distances is None else tuple(
            sorted({int(d) for d in probe_distances})
        )
        if window_size < 2 or not probe_distances or min(probe_distances) < 1:
            raise ValueError("window size must be >=2 and probe distances must be positive")
        if max(probe_distances) >= window_size:
            raise ValueError("each probe distance must be smaller than the classic window size")
        self.window_size = window_size
        self.probe_distances = probe_distances
        self.input_dim = len(self.modalities) * (self.codec.label_dim + 1)
        if not hasattr(self, "_token_encodings"):
            self._build_token_encodings()
        if probe_gamma is not None:
            self.probe_gamma = float(probe_gamma)
        if ood_eval_episodes is not None:
            self.ood_eval_episodes = int(ood_eval_episodes)

    def set_output_proj(self, proj):
        self._output_proj = proj

    def _window_tokens(self, sequences, rng):
        n_modalities = len(self.modalities)
        for _ in range(100):
            if n_modalities == 1:
                stream = rng.choice(sequences[0])
                if len(stream) < self.window_size:
                    continue
                start = rng.randrange(len(stream) - self.window_size + 1)
                tokens = [(0, int(t)) for t in stream[start : start + self.window_size]]
            else:
                available = min(len(s) for group in sequences for s in group)
                n_events = self.window_size // n_modalities
                if available < n_events:
                    continue
                start = rng.randrange(available - n_events + 1)
                chosen = [rng.choice(group) for group in sequences]
                tokens = [
                    (m, int(chosen[m][start + j]))
                    for j in range(n_events)
                    for m in range(n_modalities)
                ][: self.window_size]
                while len(tokens) < self.window_size:
                    tokens.append(tokens[-1])
            probe = self._select_probe(tokens, rng)
            if probe is not None:
                return tokens, probe
        raise ValueError(
            f"could not find a repeated token at requested probe distances {self.probe_distances}; "
            "use a larger/less uniform source, a smaller --classic-window, or change "
            "--probe-distances"
        )

    def _select_probe(self, tokens, rng):
        candidates = {requested: [] for requested in self.probe_distances}
        history = {}
        for pos, (query_modality, token) in enumerate(tokens):
            prior = history.get(token)
            if prior is not None and prior[1] == 1:
                source_pos = prior[0]
                actual = pos - source_pos
                cross_modal = len(self.modalities) == 1 or tokens[source_pos][0] != query_modality
                if cross_modal:
                    for requested in self.probe_distances:
                        if abs(actual - requested) <= max(1, requested // 8):
                            candidates[requested].append((pos, source_pos, requested))
            history[token] = (prior[0], prior[1] + 1) if prior is not None else (pos, 1)
        candidates_by_distance = [rng.choice(items) for items in candidates.values() if items]
        return rng.choice(candidates_by_distance) if candidates_by_distance else None

    def _encode(self, tokens, probe):
        x = torch.zeros(self.window_size, self.input_dim, dtype=torch.float32)
        targets = torch.zeros(self.window_size, 2, self.codec.num_digits, dtype=torch.long)
        masks = torch.zeros(self.window_size, 2, dtype=torch.float32)
        for i, (modality, token) in enumerate(tokens):
            base = modality * self._channel_size
            if i == probe[0]:
                x[i, base + self.codec.label_dim] = 1.0
            else:
                x[i, base : base + self.codec.label_dim] = self._token_encodings[token]
            if i < self.window_size - 1:
                targets[i, 0] = torch.tensor(self.codec.label_to_digit_targets(tokens[i + 1][1]))
                masks[i, 0] = 1.0
        p, source, requested = probe
        targets[p, 1] = torch.tensor(self.codec.label_to_digit_targets(tokens[p][1]))
        masks[p, 1] = 1.0
        return x, targets, masks, {"probe_pos": p, "source_pos": source, "distance": p - source,
                                  "requested_distance": requested, "modality": self.modalities[tokens[p][0]]}

    def _episode(self, sequences, rng):
        tokens, probe = self._window_tokens(sequences, rng)
        return self._encode(tokens, probe)

    def sample_batch(self, curriculum, batch_size):
        samples = [self._episode(self.train_sequences, random) for _ in range(batch_size)]
        x, targets, masks, _meta = zip(*samples)
        return torch.stack(x), torch.stack(targets), torch.stack(masks)

    def loss(self, output, target, mask):
        predict = digit_field_loss(
            output, target[..., 0, :], mask[..., 0], num_fields=1,
            digit_base=self.codec.digit_base,
        )
        probe = digit_field_loss(
            output, target[..., 1, :], mask[..., 1], num_fields=1,
            digit_base=self.codec.digit_base,
        )
        # Keep scalar metrics on device; the training loop transfers their
        # accumulated values only once per log interval.
        self._last_loss_terms = {"predict": predict.detach(), "probe": probe.detach()}
        return predict + self.probe_gamma * probe

    def diversity(self, output, target, mask):
        probe_mask = mask[..., 1].bool()
        logits = output.view(*output.shape[:-1], self.codec.num_digits, self.codec.digit_base)
        digits = logits[probe_mask].argmax(dim=-1)
        places = torch.tensor(
            [self.codec.digit_base**i for i in reversed(range(self.codec.num_digits))],
            device=digits.device,
        )
        labels = (digits * places).sum(dim=-1).detach().cpu().tolist()
        return float(len(set(labels)))

    def field_log_header(self):
        return ["probe_distance", "modality", "probe_acc", "count"]

    def write_field_log(self, writer, file, step, lesson, eval_type, field_log):
        for (distance, modality), (correct, count) in sorted(field_log.items()):
            writer.writerow([step, lesson, eval_type, distance, modality,
                             100.0 * correct / max(count, 1), count])
        file.flush()

    @torch.no_grad()
    def evaluate(self, model, device, sequences, num_episodes, rng, ablate_memory=False,
                 field_log=None, skip_modalities=None):
        model.eval()
        total = correct = perfect = 0
        local_field_log = field_log if field_log is not None else {}
        was_training = model.training
        for _ in range(num_episodes):
            x, target, mask, meta = self._episode(sequences, rng)
            if skip_modalities:
                x = x.clone()
                for modality in skip_modalities:
                    index = self.modalities.index(modality)
                    x[:, index * self._channel_size : (index + 1) * self._channel_size] = 0
            output, _ = model(
                x.unsqueeze(0).to(device), (None, None, None), reset_experience=True,
                pass_through_memory=not ablate_memory,
            )
            output = output.transpose(0, 1).contiguous()
            logits = self._output_proj(output).squeeze(0)
            pos = meta["probe_pos"]
            prediction = self.codec.decode_field(logits[pos])
            expected = self.codec.decode_field(target[pos, 1].float())
            ok = int(prediction == expected)
            total += 1
            correct += ok
            perfect += ok
            if local_field_log is not None:
                key = (meta["distance"], meta["modality"])
                counts = local_field_log.setdefault(key, [0, 0])
                counts[0] += ok
                counts[1] += 1
        if was_training:
            model.train()
        breakdown = {}
        for (distance, _modality), (field_correct, count) in local_field_log.items():
            aggregate = breakdown.setdefault(distance, [0, 0])
            aggregate[0] += field_correct
            aggregate[1] += count
        breakdown = {
            distance: (100.0 * n_correct / max(count, 1),
                       100.0 * n_correct / max(count, 1), count)
            for distance, (n_correct, count) in breakdown.items()
        }
        return (
            100.0 * correct / max(total, 1),
            100.0 * perfect / max(num_episodes, 1),
            breakdown,
        )

    def evaluate_ood(self, model, device, num_episodes, rng, verbose_n=0, field_log=None,
                     ablate_memory=False):
        return self.evaluate(model, device, self.test_sequences, num_episodes, rng,
                             ablate_memory, field_log)

    def evaluate_ood_ablated(self, model, device, num_episodes, rng, field_log=None):
        return self.evaluate(model, device, self.test_sequences, num_episodes, rng,
                             ablate_memory=True, field_log=field_log)[:2]

    def evaluate_id_ablated(self, model, device, curriculum, lesson_idx, field_log=None):
        return self.evaluate(model, device, self.train_sequences, self.eval_episodes,
                             random.Random(8100 + lesson_idx), ablate_memory=True,
                             field_log=field_log)[:2]

    def evaluate_id(self, model, device, curriculum, lesson_idx):
        return self.evaluate(model, device, self.train_sequences, self.eval_episodes,
                             random.Random(8100 + lesson_idx))[:2]

    def evaluate_robustness(self, model, device, curriculum, lesson_idx, perturbation, rng):
        return self.evaluate_id(model, device, curriculum, lesson_idx)

    def evaluate_id_combiner_stage_ablated(self, model, device, curriculum, lesson_idx, skip_stages):
        raise NotImplementedError("classic-track combiner-stage ablation is not implemented")

    def evaluate_modality_ablated(self, model, device, curriculum, lesson_idx, modality):
        if modality not in self.modalities:
            raise ValueError(f"unknown modality {modality!r}; expected one of {self.modalities}")
        return self.evaluate(
            model, device, self.train_sequences, self.eval_episodes,
            random.Random(8100 + lesson_idx), skip_modalities={modality},
        )[:2]

    def generation_probe(self, rng=None):
        rng = rng or random.Random(123)
        tokens, probe = self._window_tokens(self.test_sequences, rng)
        x, target, _mask, meta = self._encode(tokens, probe)
        return x, self.codec.decode_field(target[meta["probe_pos"], 1].float()), meta

    def encode_generated_token(self, token, position=0):
        x = torch.zeros(self.input_dim, dtype=torch.float32)
        modality = int(position) % len(self.modalities)
        base = modality * self._channel_size
        x[base : base + self.codec.label_dim] = self._token_encodings[int(token)]
        return x

    def decode_logits(self, logits):
        return self.codec.decode_field(logits)

    def sample_token(self, logits, temperature=1.0, top_k=0):
        digit_logits = logits.view(self.codec.num_digits, self.codec.digit_base) / temperature
        labels = torch.arange(self.codec.label_range, device=logits.device)
        places = torch.tensor(
            [self.codec.digit_base**i for i in reversed(range(self.codec.num_digits))],
            device=logits.device,
        )
        label_digits = (labels[:, None] // places[None, :]) % self.codec.digit_base
        log_probs = torch.log_softmax(digit_logits, dim=-1)
        digit_indexes = torch.arange(self.codec.num_digits, device=logits.device).unsqueeze(1)
        label_log_probs = log_probs[digit_indexes, label_digits.T].sum(dim=0)
        if top_k:
            k = min(top_k, self.codec.label_range)
            threshold = torch.topk(label_log_probs, k).values[-1]
            label_log_probs = label_log_probs.masked_fill(label_log_probs < threshold, float("-inf"))
        return int(torch.multinomial(torch.softmax(label_log_probs, dim=-1), 1).item())
