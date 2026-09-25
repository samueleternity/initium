"""
file: data/graph_traversal/graph_traversal.py

Graph-traversal dataset (Graves 2016-style triple encoding, curriculum over
synthetic graphs, London Underground as held-out OOD and controller test). Everything
graph-specific lives here; core_training.py only sees GraphTraversalDataset.

History of graph-specific changes (moved from core_training.py):
- v2: build_london_underground_eval() uses a local random.Random(1234), never
  the global `random.seed` (a global reseed would repeat training batches after
  every periodic OOD eval).
- v4 gate fix: evaluate_traversal() resamples num_nodes per episode over
  nodes_range (same as training) instead of pinning to the hardest size, and
  the curriculum advances on triple_acc >= ADVANCE_THRESHOLD instead of
  perfect_frac (perfect_frac compounds ~acc**hops and made deep lessons
  unreachable). perfect_frac is still computed/logged/returned.
  hop_breakdown=True buckets episodes by walk length and is printed/logged at
  every advance-check.
- v5: build_traversal_episode[_from_graph] and evaluate_traversal take an
  optional `rng` (random.Random); None -> global `random` (byte-identical to
  v4). OOD calls pass a dedicated rng so OOD walk sampling is decoupled from
  the training curriculum's RNG stream.
- v16: per-field / per-hop-position accuracy (field_log, write_field_log).
- Static Option 2: per-lesson memory size LESSON_NR_CELLS, now held on the
  curriculum instance (`curriculum.lesson_nr_cells`) so core can override it
  per run (isolate_link_ablation / dynamic_n_mode) without touching module
  globals.
"""

import random

import numpy as np
import torch
from initium.memory_manipulation.dynamic_memory_resize import resize_memory

from initium.data.base_dataset import BaseDataset
from initium.data.common.graph_io import load_graph

# ==========================================
# CONFIGURATION
# ==========================================
LABEL_RANGE = 1000
LABEL_DIGITS = 3  # each label is a 3-digit number, 0-999
DIGIT_BASE = 10  # one-hot over digits 0-9
LABEL_DIM = LABEL_DIGITS * DIGIT_BASE  # 30
TRIPLE_DIM = 3 * LABEL_DIM  # 90  (source + edge + destination)
NUM_PHASE_CHANNELS = 2  # [phase-transition, prediction-required]
INPUT_DIM = TRIPLE_DIM + NUM_PHASE_CHANNELS  # 92, matches Methods

TRAVERSAL_CURRICULUM = [
    ((3, 10), (2, 4), (1, 1)),
    ((3, 10), (2, 4), (1, 2)),
    ((5, 10), (2, 4), (1, 3)),
    ((5, 10), (2, 4), (1, 4)),
    ((10, 15), (2, 4), (1, 4)),
    ((10, 15), (2, 4), (1, 5)),
    ((10, 20), (2, 4), (1, 5)),
    ((10, 20), (2, 4), (1, 6)),
    ((10, 30), (2, 4), (1, 6)),
    ((10, 30), (2, 4), (1, 7)),
    ((10, 30), (2, 4), (1, 8)),
    ((10, 30), (2, 4), (1, 9)),
    ((10, 40), (2, 6), (1, 10)),
    ((10, 40), (2, 6), (1, 20)),
]

# Static Option 2: per-lesson memory size N, indexed like TRAVERSAL_CURRICULUM.
LESSON_NR_CELLS = [
    128,
    128,
    128,
    128,  # lessons 1-4  (<=10 nodes)
    160,
    160,
    160,
    160,  # lessons 5-8  (<=20 nodes)
    192,
    192,
    192,
    192,  # lessons 9-12 (<=30 nodes)
    256,
    256,  # lessons 13-14 (<=40 nodes, path_length up to 20)
]
assert len(LESSON_NR_CELLS) == len(TRAVERSAL_CURRICULUM)

ADVANCE_THRESHOLD = 0.85  # 85% per-triple accuracy
OLD_LESSON_MIX_RATE = 0.10  # 10% of exemplars drawn from earlier lessons
EVAL_BATCH_SIZE = 100  # episodes per lesson-completion check
OOD_PATH_LENGTH_RANGE = (3, 5)

LONDON_UNDERGROUND_EDGES_RAW = [
    ("OxfordCircus", "TottenhamCtRd", "Central"),
    ("TottenhamCtRd", "OxfordCircus", "Central"),
    ("OxfordCircus", "PiccadillyCircus", "Bakerloo"),
    ("PiccadillyCircus", "OxfordCircus", "Bakerloo"),
    ("OxfordCircus", "NottingHillGate", "Central"),
    ("OxfordCircus", "Euston", "Victoria"),
    ("BakerSt", "Marylebone", "Circle"),
    ("BakerSt", "Marylebone", "Bakerloo"),
    ("BakerSt", "OxfordCircus", "Bakerloo"),
    ("LeicesterSq", "CharingCross", "Northern"),
    ("TottenhamCtRd", "LeicesterSq", "Northern"),
    ("LeicesterSq", "PiccadillyCircus", "Piccadilly"),
    ("PiccadillyCircus", "LeicesterSq", "Piccadilly"),
    ("PiccadillyCircus", "GreenPark", "Piccadilly"),
    ("GreenPark", "PiccadillyCircus", "Piccadilly"),
    ("GreenPark", "OxfordCircus", "Victoria"),
    ("GreenPark", "Victoria", "Victoria"),
    ("Victoria", "GreenPark", "Victoria"),
    ("CharingCross", "PiccadillyCircus", "Bakerloo"),
    ("PiccadillyCircus", "CharingCross", "Bakerloo"),
    ("LeicesterSq", "TottenhamCtRd", "Northern"),
    ("CharingCross", "LeicesterSq", "Northern"),
]

# Module-level indirection so evaluate_traversal can be reused without
# threading output_proj through every call. Set via set_output_proj().
output_proj_current = None


def set_output_proj(proj):
    global output_proj_current
    output_proj_current = proj


# ==========================================
# GRAPH GENERATOR (fresh random graph on every call, nothing cached)
# ==========================================
def generate_graph(num_nodes, k_range, label_range=LABEL_RANGE):
    """Returns (edges [(src_label, edge_label, dst_label)], node_labels,
    adjacency {node_idx: [(dest_idx, edge_label)]})."""
    points = np.random.uniform(0.0, 1.0, size=(num_nodes, 2))
    node_labels = random.sample(range(label_range), num_nodes)

    dists = np.linalg.norm(points[:, None, :] - points[None, :, :], axis=-1)
    np.fill_diagonal(dists, np.inf)

    k_min, k_max = k_range
    adjacency = {i: [] for i in range(num_nodes)}
    edges = []

    for i in range(num_nodes):
        k = random.randint(k_min, min(k_max, num_nodes - 1))
        nearest_idx = np.argsort(dists[i])[:k]

        edge_labels = random.sample(node_labels, k)

        for dest_idx, edge_label in zip(nearest_idx, edge_labels):
            src_label = node_labels[i]
            dst_label = node_labels[int(dest_idx)]
            edges.append((src_label, edge_label, dst_label))
            adjacency[i].append((int(dest_idx), edge_label))

    return edges, node_labels, adjacency


# ==========================================
# TRIPLE ENCODER
# ==========================================
def encode_label(label):
    vec = torch.zeros(LABEL_DIM)
    if label is None:
        return vec
    digits = f"{label:03d}"
    for pos, ch in enumerate(digits):
        vec[pos * DIGIT_BASE + int(ch)] = 1.0
    return vec


def encode_triple(source, edge, dest, phase_transition=0.0, prediction_required=0.0):
    src_vec = encode_label(source)
    edge_vec = encode_label(edge)
    dst_vec = encode_label(dest)
    phase_vec = torch.tensor([phase_transition, prediction_required], dtype=torch.float32)
    return torch.cat([src_vec, edge_vec, dst_vec, phase_vec])


def collate_fn(batch):
    max_len = max(len(x[0]) for x in batch)
    B = len(batch)

    padded_input = torch.zeros(B, max_len, INPUT_DIM)
    padded_targets = torch.zeros(B, max_len, 9, dtype=torch.long)
    padded_mask = torch.zeros(B, max_len)

    for i, (inp, tgt, mask) in enumerate(batch):
        T = inp.size(0)
        padded_input[i, :T] = inp
        padded_targets[i, :T] = tgt
        padded_mask[i, :T] = mask

    return padded_input, padded_targets, padded_mask


# ==========================================
# LOSS + DIAGNOSTICS
# ==========================================
def digit_loss(output, target_digits, answer_mask):
    B, T, _ = output.shape
    logits = output.view(B, T, 9, DIGIT_BASE)
    log_probs = torch.log_softmax(logits, dim=-1)

    gathered = torch.gather(log_probs, -1, target_digits.unsqueeze(-1)).squeeze(-1)
    per_step_loss = -gathered.sum(dim=-1)

    mask = answer_mask.float()
    total = (per_step_loss * mask).sum()
    denom = mask.sum().clamp(min=1.0)
    return total / denom


def prediction_diversity(output, target_digits, answer_mask):
    mask = answer_mask.bool()
    if mask.sum() == 0:
        return 0.0
    logits = output.view(*output.shape[:2], 9, DIGIT_BASE)
    preds = logits.argmax(dim=-1)  # (B,T,9)
    preds_masked = preds[mask]  # (num_answer_steps, 9)
    if preds_masked.numel() == 0:
        return 0.0
    diversities = [preds_masked[:, d].unique().numel() for d in range(9)]
    return sum(diversities) / 9.0


# ==========================================
# EPISODE CONSTRUCTION
# ==========================================
def label_to_digits(label):
    s = f"{label:03d}"
    return [int(c) for c in s]


def triple_to_digit_targets(source, edge, dest):
    return label_to_digits(source) + label_to_digits(edge) + label_to_digits(dest)


def build_traversal_episode_from_graph(
    edges, node_labels, adjacency, num_nodes, path_length_range, rng=None
):
    """rng: optional random.Random; None -> global `random` (see header, v5)."""
    rng = rng if rng is not None else random
    inputs, target_digits, answer_mask = [], [], []

    def add_step(src, edge, dst, phase_transition, prediction_required, target_triple=None):
        inputs.append(encode_triple(src, edge, dst, phase_transition, prediction_required))
        if target_triple is not None:
            target_digits.append(triple_to_digit_targets(*target_triple))
            answer_mask.append(1)
        else:
            target_digits.append([0] * 9)
            answer_mask.append(0)

    shuffled_edges = edges[:]
    rng.shuffle(shuffled_edges)
    for i, (s, e, d) in enumerate(shuffled_edges):
        add_step(s, e, d, 1.0 if i == 0 else 0.0, 0.0)

    path_length = rng.randint(*path_length_range)
    start_idx = rng.randrange(num_nodes)
    cur = start_idx
    walk = []
    for _ in range(path_length):
        if not adjacency[cur]:
            break
        dst_idx, edge_label = rng.choice(adjacency[cur])
        walk.append((cur, edge_label, dst_idx))
        cur = dst_idx
    if not walk:
        return None

    for i, (src_idx, edge_label, _dst_idx) in enumerate(walk):
        src = node_labels[src_idx] if i == 0 else None
        add_step(src, edge_label, None, 1.0 if i == 0 else 0.0, 0.0)

    for i, (src_idx, edge_label, dst_idx) in enumerate(walk):
        target = (node_labels[src_idx], edge_label, node_labels[dst_idx])
        add_step(None, None, None, 1.0 if i == 0 else 0.0, 1.0, target_triple=target)

    input_seq = torch.stack(inputs)
    target_digits_t = torch.tensor(target_digits, dtype=torch.long)
    answer_mask_t = torch.tensor(answer_mask, dtype=torch.float32)
    return input_seq, target_digits_t, answer_mask_t


def build_traversal_episode(num_nodes, k_range, path_length_range, rng=None):
    # generate_graph() always uses the global random/numpy streams; it is only
    # called for ID episodes, never for the fixed-graph OOD path.
    edges, node_labels, adjacency = generate_graph(num_nodes, k_range)
    return build_traversal_episode_from_graph(
        edges, node_labels, adjacency, num_nodes, path_length_range, rng=rng
    )


# ==========================================
# CURRICULUM
# ==========================================
class TraversalCurriculum:
    def __init__(self, table=TRAVERSAL_CURRICULUM, custom_graph=None):
        self.table = table
        self.lesson = 0
        # per-instance so core can override per run (isolate_link_ablation /
        # dynamic_n_mode) without mutating module globals
        self.lesson_nr_cells = list(LESSON_NR_CELLS)
        # custom_graph: (edges, node_labels, adjacency) from a user-supplied
        # graph file (GraphTraversalDataset's dataset_link), or None for the
        # default per-episode synthetic generate_graph() behavior. When set,
        # sample_episode() walks THIS fixed graph instead of generating a
        # fresh random one -- node count is then whatever the supplied graph
        # has, not the per-lesson nodes_range; only path_length_range still
        # scales difficulty across lessons.
        self.custom_graph = custom_graph

    def _sample_lesson_params(self):
        if self.lesson > 0 and random.random() < OLD_LESSON_MIX_RATE:
            idx = random.randint(0, self.lesson - 1)
        else:
            idx = self.lesson
        return self.table[idx]

    def sample_episode(self):
        nodes_range, out_degree_range, path_len_range = self._sample_lesson_params()
        if self.custom_graph is not None:
            edges, node_labels, adjacency = self.custom_graph
            ep = None
            while ep is None:
                ep = build_traversal_episode_from_graph(
                    edges, node_labels, adjacency, len(node_labels), path_len_range
                )
            return ep
        ep = None
        while ep is None:
            num_nodes = random.randint(*nodes_range)
            ep = build_traversal_episode(num_nodes, out_degree_range, path_len_range)
        return ep

    def maybe_advance(self, model, device, step=None, optimizer=None):
        """Returns (lesson, id_triple_acc, id_perfect_frac). Eval samples
        num_nodes over the lesson's nodes_range (as trained), or walks the
        fixed custom_graph when one was supplied; the gate advances on
        triple_acc >= ADVANCE_THRESHOLD (see header, v4)."""
        nodes_range, out_degree_range, path_len_range = self.table[self.lesson]
        if self.custom_graph is not None:
            edges, node_labels, adjacency = self.custom_graph
            graph_kwargs = dict(
                fixed_graph=(edges, node_labels, adjacency, len(node_labels)),
                path_length_range=path_len_range,
            )
        else:
            graph_kwargs = dict(
                nodes_range=nodes_range, k_range=out_degree_range, path_length_range=path_len_range
            )
        triple_acc, perfect_frac, hop_breakdown = evaluate_traversal(
            model,
            device,
            num_episodes=EVAL_BATCH_SIZE,
            verbose_n=0,
            field_log=(field_log := {}),
            hop_breakdown=True,
            **graph_kwargs,
        )
        if hop_breakdown:
            breakdown_str = ", ".join(
                f"{hops}-hop: acc {acc:.1f}% perfect {pf:.1f}% (n={n})"
                for hops, (acc, pf, n) in hop_breakdown.items()
            )
            print(f"    [lesson {self.lesson + 1} eval by hop count] {breakdown_str}")
            hop_writer = getattr(self, "hop_log_writer", None)
            if hop_writer is not None:
                for hops, (acc, pf, n) in sorted(hop_breakdown.items()):
                    hop_writer.writerow([step, self.lesson + 1, "id", hops, acc, pf, n])
                self.hop_log_file.flush()

        field_writer = getattr(self, "field_log_writer", None)
        if field_writer is not None and field_log:
            write_field_log(
                field_writer, self.field_log_file, step, self.lesson + 1, "id", field_log
            )

        if triple_acc / 100.0 >= ADVANCE_THRESHOLD and self.lesson < len(self.table) - 1:
            self.lesson += 1
            new_n = self.lesson_nr_cells[self.lesson]
            if new_n != self.lesson_nr_cells[self.lesson - 1]:
                # optimizer=optimizer: without it a live resize orphans the
                # rebuilt Memory sublayers from Adam's param_groups.
                resize_memory(model, new_n, device=device, optimizer=optimizer)
                print(f">>> Memory resized to nr_cells={new_n} for lesson {self.lesson + 1}")
            print(f">>> Curriculum advanced to lesson {self.lesson + 1}/{len(self.table)}")
            writer = getattr(self, "advance_log_writer", None)
            if writer is not None:
                writer.writerow([step, self.lesson + 1, len(self.table)])
                self.advance_log_file.flush()
        return self.lesson, triple_acc, perfect_frac


def sample_batch(curriculum, batch_size):
    episodes = [curriculum.sample_episode() for _ in range(batch_size)]
    return collate_fn(episodes)


# ==========================================
# EVALUATION
# ==========================================
def decode_prediction(output_step):
    logits = output_step.view(9, DIGIT_BASE)
    digit_preds = logits.argmax(dim=-1).tolist()
    src = int("".join(str(d) for d in digit_preds[0:3]))
    edge = int("".join(str(d) for d in digit_preds[3:6]))
    dst = int("".join(str(d) for d in digit_preds[6:9]))
    return src, edge, dst


def evaluate_traversal(
    model,
    device,
    num_episodes=100,
    verbose_n=3,
    num_nodes=None,
    nodes_range=None,
    k_range=None,
    path_length_range=None,
    fixed_graph=None,
    hop_breakdown=False,
    rng=None,
    ablate_memory=False,
    field_log=None,
    model_kwargs=None,
):
    """
    num_nodes: fixed graph size for every episode. nodes_range: (lo, hi),
        num_nodes resampled per episode (same distribution as training).
        Passing both is an error; neither is only valid with fixed_graph.
    hop_breakdown: also return {path_length: (triple_acc, perfect_frac, n)}.
    rng: optional random.Random for this call's sampling (None -> global).
    ablate_memory: passes pass_through_memory=False to the model (memory
        read AND write skipped) -- functional-usage check (LB-9/Concept 6).
    field_log: optional dict filled with per-field / per-hop-position counts
        (see write_field_log).
    """
    if num_nodes is not None and nodes_range is not None:
        raise ValueError("evaluate_traversal: pass num_nodes or nodes_range, not both")

    model.eval()
    total_triples, correct_triples = 0, 0
    perfect_episodes = 0
    tested = 0
    by_hops = {}  # path_length -> [triples_total, triples_correct, episodes, episodes_perfect]

    with torch.no_grad():
        while tested < num_episodes:
            if fixed_graph is not None:
                edges, node_labels, adjacency, n = fixed_graph
                ep = build_traversal_episode_from_graph(
                    edges, node_labels, adjacency, n, path_length_range, rng=rng
                )
            else:
                n = (rng or random).randint(*nodes_range) if nodes_range is not None else num_nodes
                ep = build_traversal_episode(n, k_range, path_length_range, rng=rng)
            if ep is None:
                continue
            input_seq, target_digits, answer_mask = ep
            input_seq = input_seq.unsqueeze(0).to(device)

            hidden = (None, None, None)
            output, _ = model(
                input_seq,
                hidden,
                reset_experience=True,
                pass_through_memory=not ablate_memory,
                **(model_kwargs or {}),
            )
            output = output.transpose(0, 1).contiguous().squeeze(0)  # (T, 92)
            output = output_proj_current(output)  # (T, 90)

            answer_idx = (answer_mask == 1).nonzero(as_tuple=True)[0]
            episode_perfect = True
            ep_total = ep_correct = 0
            _prev_dst_ok = None  # was the previous hop's dst correct (chain-propagation diagnostic)
            for hop_pos, idx in enumerate(answer_idx, start=1):
                pred = decode_prediction(output[idx])
                tgt_digits = target_digits[idx].tolist()
                tgt = (
                    int("".join(map(str, tgt_digits[0:3]))),
                    int("".join(map(str, tgt_digits[3:6]))),
                    int("".join(map(str, tgt_digits[6:9]))),
                )
                is_correct = pred == tgt
                if field_log is not None:  # keyed by (episode path length, hop position)
                    _s, _e, _d = pred[0] == tgt[0], pred[1] == tgt[1], pred[2] == tgt[2]
                    _fl = field_log.setdefault((len(answer_idx), hop_pos), [0] * 9)
                    _fl[0] += 1  # n triples
                    _fl[1] += int(_s)
                    _fl[2] += int(_e)
                    _fl[3] += int(_d)
                    _fl[4] += int(is_correct)
                    if _s and _e:  # lookup check: dst given src+edge right
                        _fl[5] += 1
                        _fl[6] += int(_d)
                    if _prev_dst_ok:  # chain check: src given previous dst right
                        _fl[7] += 1
                        _fl[8] += int(_s)
                    _prev_dst_ok = _d
                correct_triples += int(is_correct)
                total_triples += 1
                ep_correct += int(is_correct)
                ep_total += 1
                if not is_correct:
                    episode_perfect = False
                if tested < verbose_n:
                    print(f"  Pred: {pred} | Target: {tgt} | Correct: {is_correct}")

            perfect_episodes += int(episode_perfect)
            tested += 1

            if hop_breakdown:
                # ep_total == walk length (one answer step per hop)
                acc = by_hops.setdefault(ep_total, [0, 0, 0, 0])
                acc[0] += ep_total
                acc[1] += ep_correct
                acc[2] += 1
                acc[3] += int(episode_perfect)

    triple_acc = correct_triples / max(total_triples, 1) * 100
    perfect_frac = perfect_episodes / tested * 100
    print(
        f"Eval: triple-level acc {triple_acc:.2f}% | "
        f"perfect-traversal fraction {perfect_frac:.2f}% ({tested} episodes)"
    )
    model.train()

    if hop_breakdown:
        breakdown = {
            hops: (
                correct / max(total, 1) * 100,
                n_perfect / max(n_eps, 1) * 100,
                n_eps,
            )
            for hops, (total, correct, n_eps, n_perfect) in sorted(by_hops.items())
        }
        return triple_acc, perfect_frac, breakdown
    return triple_acc, perfect_frac


def write_field_log(writer, file, step, lesson, eval_type, field_log):
    """Dump one evaluate_traversal(field_log=...) dict to the field-breakdown
    CSV and print a compact console summary (hop positions 1-4).
    field_log[(path_length, hop_position)] = [n, src_ok, edge_ok, dst_ok, triple_ok,
        n_src_edge_ok, dst_ok_given_src_edge, n_prev_dst_ok, src_ok_given_prev_dst]."""
    nan = float("nan")
    for (path_len, hop_pos), (n, s, e, d, t, n_se, d_se, n_pd, s_pd) in sorted(field_log.items()):
        writer.writerow(
            [
                step,
                lesson,
                eval_type,
                path_len,
                hop_pos,
                n,
                s / n * 100,
                e / n * 100,
                d / n * 100,
                t / n * 100,
                (d_se / n_se * 100) if n_se else "",
                n_se,
                (s_pd / n_pd * 100) if n_pd else "",
                n_pd,
            ]
        )
    file.flush()
    agg = {}
    for (_path_len, hop_pos), v in field_log.items():
        a = agg.setdefault(hop_pos, [0] * 9)
        for i in range(9):
            a[i] += v[i]
    parts = []
    for hop_pos in sorted(agg)[:4]:
        n, s, e, d, t, n_se, d_se, n_pd, s_pd = agg[hop_pos]
        part = (
            f"hop{hop_pos}: src {s / n * 100:.0f} edge {e / n * 100:.0f} dst {d / n * 100:.0f} "
            f"dst|src+edge {(d_se / n_se * 100) if n_se else nan:.0f}"
        )
        if hop_pos > 1:
            part += f" src|prev_dst_ok {(s_pd / n_pd * 100) if n_pd else nan:.0f}"
        parts.append(part + f" (n={n})")
    print(f"    [field acc {eval_type} step {step}] " + " | ".join(parts))


def build_london_underground_eval():
    """Fixed, reproducible OOD graph/label mapping. Uses a local
    random.Random(1234) so it never touches the global RNG stream."""
    stations = sorted(
        {s for s, d, _ in LONDON_UNDERGROUND_EDGES_RAW}
        | {d for s, d, _ in LONDON_UNDERGROUND_EDGES_RAW}
    )
    lines = sorted({l for _, _, l in LONDON_UNDERGROUND_EDGES_RAW})

    rng = random.Random(1234)
    all_labels = rng.sample(range(1000), len(stations) + len(lines))
    station_to_label = dict(zip(stations, all_labels[: len(stations)]))
    line_to_label = dict(zip(lines, all_labels[len(stations) :]))

    edges = [
        (station_to_label[s], line_to_label[l], station_to_label[d])
        for s, d, l in LONDON_UNDERGROUND_EDGES_RAW
    ]

    adjacency = {i: [] for i in range(len(stations))}
    station_idx = {s: i for i, s in enumerate(stations)}
    for s, d, l in LONDON_UNDERGROUND_EDGES_RAW:
        adjacency[station_idx[s]].append((station_idx[d], line_to_label[l]))

    node_labels = [station_to_label[s] for s in stations]
    return edges, node_labels, adjacency


# ==========================================
# DATASET ADAPTER (what core_training.py uses)
# ==========================================
class GraphTraversalDataset(BaseDataset):
    name = "graph-traversal"
    input_dim = INPUT_DIM
    output_dim = TRIPLE_DIM
    advance_threshold = ADVANCE_THRESHOLD
    _custom_graph: (
        tuple[list[tuple[int, int, int]], list[int], dict[int, list[tuple[int, int]]]] | None
    )
    _custom_test_graph: (
        tuple[list[tuple[int, int, int]], list[int], dict[int, list[tuple[int, int]]]] | None
    )

    def __init__(self, dataset_link: str | None = None, test_dataset_link: str | None = None):
        # dataset_link: path to a real edge file (see data/common/graph_io.py
        # for the accepted formats) to TRAIN on, instead of the default
        # per-episode synthetic generate_graph(). None (default) -> unchanged
        # synthetic behavior.
        # test_dataset_link: path to a real edge file to use as the fixed
        # OOD TEST graph, instead of the built-in London Underground -- the
        # graph analogue of text/audio/video's test_dataset_link. None
        # (default) -> the built-in London Underground, unchanged.
        self._custom_graph = load_graph(dataset_link) if dataset_link is not None else None
        self._custom_test_graph = (
            load_graph(test_dataset_link) if test_dataset_link is not None else None
        )
        if self._custom_graph is not None:
            print(
                f"[graph-traversal] training graph loaded from {dataset_link}: "
                f"{len(self._custom_graph[1])} nodes, {len(self._custom_graph[0])} edges"
            )
        if self._custom_test_graph is not None:
            print(
                f"[graph-traversal] test graph loaded from {test_dataset_link}: "
                f"{len(self._custom_test_graph[1])} nodes, {len(self._custom_test_graph[0])} edges"
            )

    def make_curriculum(self):
        return TraversalCurriculum(custom_graph=self._custom_graph)

    def sample_batch(self, curriculum, batch_size):
        return sample_batch(curriculum, batch_size)

    def loss(self, output, target, mask):
        return digit_loss(output, target, mask)

    def diversity(self, output, target, mask):
        return prediction_diversity(output, target, mask)

    def set_output_proj(self, proj):
        set_output_proj(proj)

    def evaluate_ood(self, model, device, num_episodes, rng, verbose_n=0, field_log=None):
        if self._custom_test_graph is not None:
            edges, node_labels, adjacency = self._custom_test_graph
        else:
            edges, node_labels, adjacency = build_london_underground_eval()
        return evaluate_traversal(
            model,
            device,
            num_episodes=num_episodes,
            verbose_n=verbose_n,
            field_log=field_log,
            fixed_graph=(edges, node_labels, adjacency, len(node_labels)),
            path_length_range=OOD_PATH_LENGTH_RANGE,
            rng=rng,
            hop_breakdown=True,
        )

    def _lesson_eval_graph_kwargs(self, curriculum, lesson_idx):
        """-> kwargs for evaluate_traversal() covering the 'which graph(s)'
        half of a lesson-distribution eval: nodes_range/k_range (synthetic,
        default) or a fixed_graph (when dataset_link supplied a custom
        training graph -- see __init__ above). Either way path_length_range
        still comes from the lesson table."""
        nodes_range, out_degree_range, path_len_range = curriculum.table[lesson_idx]
        if self._custom_graph is not None:
            edges, node_labels, adjacency = self._custom_graph
            return dict(
                fixed_graph=(edges, node_labels, adjacency, len(node_labels)),
                path_length_range=path_len_range,
            )
        return dict(
            nodes_range=nodes_range, k_range=out_degree_range, path_length_range=path_len_range
        )

    def evaluate_id_ablated(self, model, device, curriculum, lesson_idx):
        return evaluate_traversal(
            model,
            device,
            num_episodes=EVAL_BATCH_SIZE,
            verbose_n=0,
            ablate_memory=True,
            **self._lesson_eval_graph_kwargs(curriculum, lesson_idx),
        )

    def evaluate_id_combiner_stage_ablated(
        self, model, device, curriculum, lesson_idx, skip_stages
    ):
        """Same lesson-distribution eval as evaluate_id_ablated, but bypasses
        one stage of a hybrid split-graph combiner (see
        ChainedControllerWrapper.forward's skip_stages / SplitGraphDNC's
        combiner_skip_stages) instead of Memory. Only meaningful when
        model.combiner_wrapper is a ChainedControllerWrapper."""
        return evaluate_traversal(
            model,
            device,
            num_episodes=EVAL_BATCH_SIZE,
            verbose_n=0,
            model_kwargs={"combiner_skip_stages": skip_stages},
            **self._lesson_eval_graph_kwargs(curriculum, lesson_idx),
        )

    def field_log_header(self):
        return [
            "path_length",
            "hop_position",
            "n_triples",
            "src_acc",
            "edge_acc",
            "dst_acc",
            "triple_acc",
            "dst_acc_given_src_edge",
            "n_src_edge_correct",
            "src_acc_given_prev_dst_correct",
            "n_prev_dst_correct",
        ]

    def write_field_log(self, writer, file, step, lesson, eval_type, field_log):
        write_field_log(writer, file, step, lesson, eval_type, field_log)
