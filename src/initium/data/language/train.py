"""Runnable standalone trainer for the broad text + learned world-state model."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import random

import torch

from initium.data.language.batch_stream import JsonlBatchStream
from initium.data.language.checkpoint import load_checkpoint, save_checkpoint, file_sha256
from initium.data.language.collators import LanguageCollator
from initium.data.language.config import ArchitectureConfig, LanguageRunConfig, LossWeights
from initium.data.language.dialogue import Dialogue, DialogueTurn, encode_dialogue
from initium.data.language.model import LanguageWorldModel, pool_sequence
from initium.data.language.objectives import (
    LossBundle,
    causal_lm_loss,
    logic_state_loss,
    same_state_loss,
    transition_loss,
)
from initium.data.language.tokenizer import load_tokenizer
from initium.data.language.trainer import InterleavedTrainer
from initium.training.neuromodulator import Neuromodulator


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("data/language/run_config.json"))
    parser.add_argument("--corpus-manifest", type=Path, required=True,
                        help="corpus_manifest.json produced by build_language_corpus.py")
    parser.add_argument("--world-data-dir", type=Path,
                        help="directory containing state_views.jsonl, transitions.jsonl, logic.jsonl")
    parser.add_argument("--dialogue-jsonl", type=Path, nargs="+", default=[],
                        help="one or more licensed multi-turn dialogue JSONL shards")
    parser.add_argument("--output-dir", type=Path, default=Path("data/language/runs/candidate"))
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--hidden-size", type=int)
    parser.add_argument("--learning-rate", "--lr", dest="learning_rate", type=float)
    parser.add_argument("--minimum-learning-rate", "--lr-min", dest="minimum_learning_rate", type=float)
    parser.add_argument("--weight-decay", type=float)
    parser.add_argument("--warmup-steps", type=int)
    parser.add_argument("--evaluate-every", type=int)
    parser.add_argument("--evaluation-batches", type=int)
    parser.add_argument("--max-grad-norm", type=float)
    parser.add_argument("--log-every", type=int)
    parser.add_argument("--total-steps", type=int)
    parser.add_argument("--checkpoint-every", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--fp32", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--tokenizer-path", type=str)
    parser.add_argument("--vocab-size", type=int)
    parser.add_argument("--max-sequence-tokens", type=int)

    # Split-graph Mamba1 knobs. The parallel dense workspace currently requires
    # this variant, so incompatible variants fail at argparse/config validation.
    parser.add_argument("--split-graph", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--split-graph-variant", choices=("mamba1",))
    parser.add_argument("--split-graph-num-blocks", type=int)
    parser.add_argument("--split-graph-mamba-d-state", type=int)
    parser.add_argument("--split-graph-mamba-d-conv", type=int)
    parser.add_argument("--split-graph-mamba-expand", type=int)
    parser.add_argument("--workspace-token-id-input", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--workspace-parallel-dense", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--workspace-block-index", type=int)
    parser.add_argument("--workspace-num-slots", type=int)
    parser.add_argument("--workspace-key-dim", type=int)
    parser.add_argument("--workspace-scan-chunk", type=int)

    parser.add_argument("--modulation", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--mod-targets", nargs="+", choices=("dt", "a", "B", "C", "D"))
    parser.add_argument("--mod-blocks", nargs="+", type=int)
    parser.add_argument("--mod-source", choices=("workspace", "backbone", "static"))
    parser.add_argument("--mod-rank", type=int)
    for key in ("dt", "a", "B", "C", "D"):
        parser.add_argument(f"--mod-alpha-{key}", type=float)
    parser.add_argument("--mod-lambda", type=float)
    parser.add_argument("--mod-temporal-lambda", type=float)
    parser.add_argument("--mod-lr-scale", type=float)
    parser.add_argument("--mod-alpha-warmup-steps", type=int)

    parser.add_argument("--neuromod", choices=("off", "lr", "plasticity", "gate", "responsiveness",
                                                 "consolidation", "random"))
    parser.add_argument("--neuromod-combine", nargs="+", choices=("lr", "plasticity", "gate",
                                                                    "responsiveness", "consolidation"))
    parser.add_argument("--no-neuromod-combine", action="store_true")
    parser.add_argument("--neuromod-groups", nargs="+", choices=("backbone", "moe", "workspace",
                                                                   "modulation", "task_head"))
    parser.add_argument("--neuromod-range", type=float)
    parser.add_argument("--neuromod-every", type=int)
    parser.add_argument("--neuromod-slew", type=float)
    parser.add_argument("--neuromod-seed", type=int)
    parser.add_argument("--neuromod-gate-kappa", type=float)
    parser.add_argument("--neuromod-consolidation-lambda", type=float)
    parser.add_argument("--neuromod-retention-control", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--neuromod-loss-outlier-gating", action=argparse.BooleanOptionalAction, default=None)

    for name in ("text", "same-state", "transition", "logic", "dialogue"):
        dest = name.replace("-", "_")
        parser.add_argument(f"--{name}-loss-weight", dest=f"{dest}_loss_weight", type=float)
        parser.add_argument(f"--{name}-sampling-weight", dest=f"{dest}_sampling_weight", type=float)
    return parser.parse_args()


def _batches(paths, batch_size, seed, state=None, split="train"):
    return JsonlBatchStream(paths, batch_size=batch_size, seed=seed, state=state, split=split)


def _optimizer_groups(model, weight_decay, modulation_lr_scale=1.0):
    grouped = {}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        group = InterleavedTrainer._parameter_group(name)
        decay = weight_decay if parameter.ndim >= 2 and not name.endswith("embedding.weight") else 0.0
        grouped.setdefault((group, decay), []).append(parameter)
    return [{"params": params, "nm_group": group, "weight_decay": decay,
             "lr_multiplier": modulation_lr_scale if group == "modulation" else 1.0}
            for (group, decay), params in grouped.items()]


def _apply_cli_overrides(config, args):
    arch_fields = {
        "split_graph": "split_graph",
        "split_graph_variant": "split_graph_variant",
        "split_graph_num_blocks": "split_graph_num_blocks",
        "split_graph_mamba_d_state": "mamba_d_state",
        "split_graph_mamba_d_conv": "mamba_d_conv",
        "split_graph_mamba_expand": "mamba_expand",
        "workspace_token_id_input": "workspace_token_id_input",
        "workspace_parallel_dense": "workspace_parallel_dense",
        "workspace_block_index": "workspace_block_index",
        "workspace_num_slots": "workspace_num_slots",
        "workspace_key_dim": "workspace_key_dim",
        "workspace_scan_chunk": "workspace_scan_chunk",
        "modulation": "modulation",
        "mod_source": "modulation_source",
        "mod_rank": "modulation_rank",
        "mod_lambda": "modulation_lambda",
        "mod_temporal_lambda": "modulation_temporal_lambda",
        "mod_lr_scale": "modulation_lr_scale",
        "mod_alpha_warmup_steps": "modulation_alpha_warmup_steps",
        "neuromod": "neuromod",
        "neuromod_groups": "neuromod_groups",
        "neuromod_range": "neuromod_range",
        "neuromod_every": "neuromod_every",
        "neuromod_slew": "neuromod_slew",
        "neuromod_seed": "neuromod_seed",
        "neuromod_gate_kappa": "neuromod_gate_kappa",
        "neuromod_consolidation_lambda": "neuromod_consolidation_lambda",
        "neuromod_retention_control": "neuromod_retention_control",
        "neuromod_loss_outlier_gating": "neuromod_loss_outlier_gating",
        "fp32": "fp32",
        "total_steps": "total_steps",
        "checkpoint_every": "checkpoint_every",
        "seed": "seed",
    }
    arch_values = {field: getattr(args, arg) for arg, field in arch_fields.items()
                   if getattr(args, arg) is not None}
    if args.neuromod_combine is not None and args.no_neuromod_combine:
        raise SystemExit("use either --neuromod-combine or --no-neuromod-combine")
    if args.neuromod_combine is not None:
        arch_values["neuromod_combine"] = tuple(args.neuromod_combine)
    elif args.no_neuromod_combine:
        arch_values["neuromod_combine"] = ()
    if args.mod_targets is not None:
        arch_values["modulation_targets"] = tuple(args.mod_targets)
    if args.mod_blocks is not None:
        arch_values["modulation_blocks"] = tuple(args.mod_blocks)
    alpha_fields = {"dt": args.mod_alpha_dt, "a": args.mod_alpha_a, "B": args.mod_alpha_B,
                    "C": args.mod_alpha_C, "D": args.mod_alpha_D}
    if any(value is not None for value in alpha_fields.values()):
        alphas = dict(config.architecture.modulation_alphas)
        alphas.update({key: value for key, value in alpha_fields.items() if value is not None})
        arch_values["modulation_alphas"] = alphas

    root_fields = {
        "tokenizer_path": "tokenizer_path", "vocab_size": "vocab_size",
        "hidden_size": "hidden_size", "batch_size": "batch_size",
        "learning_rate": "learning_rate", "minimum_learning_rate": "minimum_learning_rate",
        "weight_decay": "weight_decay", "warmup_steps": "warmup_steps",
        "evaluate_every": "evaluate_every", "evaluation_batches": "evaluation_batches",
        "max_grad_norm": "max_grad_norm", "log_every": "log_every",
        "max_sequence_tokens": "max_sequence_tokens",
    }
    root_values = {field: getattr(args, arg) for arg, field in root_fields.items()
                   if getattr(args, arg) is not None}
    if args.seed is not None:
        root_values["seed"] = args.seed
    weight_fields = {name: getattr(args, f"{name}_loss_weight")
                     for name in ("text", "same_state", "transition", "logic", "dialogue")}
    if any(value is not None for value in weight_fields.values()):
        root_values["loss_weights"] = replace(
            config.loss_weights, **{name: value for name, value in weight_fields.items()
                                    if value is not None}
        )
    sampling = dict(config.objective_sampling_weights)
    for name in ("text", "same_state", "transition", "logic", "dialogue"):
        value = getattr(args, f"{name}_sampling_weight")
        if value is not None:
            sampling[name] = value
    if any(getattr(args, f"{name}_sampling_weight") is not None
           for name in ("text", "same_state", "transition", "logic", "dialogue")):
        root_values["objective_sampling_weights"] = sampling
    if "seed" in root_values:
        arch_values["seed"] = root_values["seed"]
    architecture = replace(config.architecture, **arch_values)
    return replace(config, architecture=architecture, **root_values)


def _make_dialogues(rows, tokenizer, max_length, *, split="train"):
    encoded = []
    for row in rows:
        if row.get("split", "train") != split:
            continue
        dialogue = Dialogue(
            str(row["conversation_id"]),
            tuple(DialogueTurn(str(turn["role"]), str(turn["content"])) for turn in row["turns"]),
            split, row.get("metadata", {}),
        )
        encoded.append(encode_dialogue(dialogue, tokenizer))
    return encoded


def _sum_auxiliary_losses(*losses):
    present = [loss for loss in losses if loss is not None]
    return sum(present[1:], present[0]) if present else None


def _forward_loss(model, collator, tokenizer, device, objective, rows, *, split="train"):
    if objective == "text":
        batch = collator.text(rows)
        ids, labels = batch["input_ids"].to(device), batch["labels"].to(device)
        output = model(ids)
        auxiliary = model.pop_modulation_aux_loss()
        return LossBundle(text=causal_lm_loss(output.logits, labels), auxiliary=auxiliary)
    if objective == "same_state":
        batch = collator.same_state(rows)
        first, _ = model.encode(batch["first_ids"].to(device))
        first_aux = model.pop_modulation_aux_loss()
        second, _ = model.encode(batch["second_ids"].to(device))
        second_aux = model.pop_modulation_aux_loss()
        a = model.objective_heads.state(pool_sequence(first, batch["first_mask"].to(device)))
        b = model.objective_heads.state(pool_sequence(second, batch["second_mask"].to(device)))
        return LossBundle(same_state=same_state_loss(a, b),
                          auxiliary=_sum_auxiliary_losses(first_aux, second_aux))
    if objective == "transition":
        batch = collator.transition(rows)
        before, _ = model.encode(batch["before_ids"].to(device))
        before_aux = model.pop_modulation_aux_loss()
        after, _ = model.encode(batch["after_ids"].to(device))
        after_aux = model.pop_modulation_aux_loss()
        before_state = model.objective_heads.state(
            pool_sequence(before, batch["before_mask"].to(device))
        )
        future_state = model.objective_heads.state(
            pool_sequence(after, batch["after_mask"].to(device))
        ).detach()
        prediction = model.objective_heads.transition(before_state, batch["action_ids"].to(device))
        valid = batch["valid"].to(device)
        if valid.any():
            loss = transition_loss(prediction[valid], future_state[valid])
        else:
            loss = prediction.sum() * 0.0
        return LossBundle(transition=loss, auxiliary=_sum_auxiliary_losses(before_aux, after_aux))
    if objective == "logic":
        batch = collator.logic(rows)
        output = model(batch["input_ids"].to(device))
        auxiliary = model.pop_modulation_aux_loss()
        pooled = pool_sequence(output.hidden, batch["attention_mask"].to(device))
        logits = model.objective_heads.logic(pooled)
        return LossBundle(logic=logic_state_loss(logits, batch["labels"].to(device)),
                          auxiliary=auxiliary)
    if objective == "dialogue":
        encoded = _make_dialogues(rows, tokenizer, collator.max_length, split=split)
        if not encoded:
            raise ValueError("dialogue batch contains no training-split conversations")
        batch = collator.dialogue(encoded)
        ids, labels = batch["input_ids"].to(device), batch["labels"].to(device)
        output = model(ids)
        auxiliary = model.pop_modulation_aux_loss()
        return LossBundle(dialogue=causal_lm_loss(output.logits, labels), auxiliary=auxiliary)
    raise ValueError(f"unknown objective {objective!r}")


def _evaluate_objectives(model, streams, collator, tokenizer, device, batches: int):
    model.eval()
    metrics = {}
    with torch.inference_mode():
        for objective, stream in streams.items():
            values = []
            for _ in range(batches):
                rows = stream.next_batch()
                bundle = _forward_loss(model, collator, tokenizer, device, objective, rows,
                                       split="validation")
                value = getattr(bundle, objective)
                if value is not None:
                    values.append(float(value))
            metrics[f"{objective}_loss"] = sum(values) / max(len(values), 1)
    model.train()
    return metrics


def main():
    args = _parse_args()
    config = _apply_cli_overrides(LanguageRunConfig.load(args.config), args)
    for name in ("batch_size", "hidden_size", "learning_rate", "minimum_learning_rate",
                 "weight_decay", "warmup_steps", "evaluate_every", "evaluation_batches",
                 "max_grad_norm", "log_every", "max_sequence_tokens"):
        setattr(args, name, getattr(config, name))
    if args.batch_size < 1 or args.evaluate_every < 1 or args.evaluation_batches < 1:
        raise SystemExit("batch size and evaluation intervals must be positive")
    architecture = config.architecture
    random.seed(config.seed)
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)
    device = torch.device(args.device)
    manifest_path = args.corpus_manifest.resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    corpus_root = manifest_path.parent
    train_shards = [corpus_root / p for p in manifest["shards"].get("train", [])]
    validation_shards = [corpus_root / p for p in manifest["shards"].get("validation", [])]
    if not train_shards or not validation_shards:
        raise SystemExit("corpus manifest must contain train and validation shards")
    tokenizer_path = Path(config.tokenizer_path)
    if not tokenizer_path.is_absolute():
        tokenizer_path = Path.cwd() / tokenizer_path
    tokenizer_path = tokenizer_path.resolve()
    tokenizer = load_tokenizer(tokenizer_path)
    vocab_size = tokenizer.get_vocab_size()
    if vocab_size != config.vocab_size:
        raise SystemExit(f"config vocab_size={config.vocab_size} but tokenizer has {vocab_size}")
    collator = LanguageCollator(tokenizer, max_length=config.max_sequence_tokens)

    sources = {"text": train_shards}
    world_root = args.world_data_dir.resolve() if args.world_data_dir else None
    sources.update({
        "same_state": [world_root / "state_views.jsonl"] if world_root else [],
        "transition": [world_root / "transitions.jsonl"] if world_root else [],
        "logic": [world_root / "logic.jsonl"] if world_root else [],
        "dialogue": [path.resolve() for path in args.dialogue_jsonl],
    })
    weights = config.loss_weights
    for objective in ("text", "same_state", "transition", "logic", "dialogue"):
        if (getattr(weights, objective) > 0
                and (not sources[objective] or not all(Path(p).is_file() for p in sources[objective]))):
            raise SystemExit(f"enabled objective {objective!r} is missing its data file(s)")

    available = [name for name, paths in sources.items()
                 if getattr(weights, name) > 0 and paths and all(Path(p).is_file() for p in paths)]
    sampling = {name: float(config.objective_sampling_weights.get(name, 0.0)) for name in available}
    if any(value < 0 for value in sampling.values()) or sum(sampling.values()) <= 0:
        raise SystemExit("active objectives need nonnegative sampling weights with positive total")
    for name, value in sampling.items():
        if value <= 0:
            raise SystemExit(f"active objective {name!r} needs a positive sampling weight")

    model = LanguageWorldModel(vocab_size, hidden_size=config.hidden_size,
                               architecture=architecture).to(device=device, dtype=torch.float32)
    optimizer = torch.optim.AdamW(
        _optimizer_groups(model, args.weight_decay, architecture.modulation_lr_scale),
        lr=args.learning_rate,
    )
    nm_arms = set(architecture.effective_neuromod_arms)
    nm_mode = "plasticity" if "plasticity" in nm_arms else (
        "lr" if "lr" in nm_arms else "consolidation" if "consolidation" in nm_arms else
        "random" if "random" in nm_arms else "off"
    )
    nm_groups = list(architecture.neuromod_groups)
    neuromodulator = Neuromodulator(
        groups=nm_groups, r=architecture.neuromod_range, update_every=architecture.neuromod_every,
        slew=architecture.neuromod_slew, mode=nm_mode,
        seed=config.seed if architecture.neuromod_seed == 0 else architecture.neuromod_seed,
    )
    trainer = InterleavedTrainer(
        model, optimizer, weights=weights, grad_clip=args.max_grad_norm,
        neuromodulator=neuromodulator, base_lr=args.learning_rate,
        min_lr=args.minimum_learning_rate, total_steps=architecture.total_steps,
        warmup_steps=args.warmup_steps,
        loss_outlier_gating=architecture.neuromod_loss_outlier_gating,
        gate_kappa=architecture.neuromod_gate_kappa,
        consolidation_lambda=architecture.neuromod_consolidation_lambda,
        consolidation_enabled="consolidation" in nm_arms,
        gradient_gating="gate" in nm_arms,
        modulation_alpha_warmup_steps=architecture.modulation_alpha_warmup_steps,
    )
    if architecture.total_steps < 1 or architecture.checkpoint_every < 1:
        raise SystemExit("total_steps and checkpoint_every in the run config must be positive")
    input_hashes = {
        name: [file_sha256(path) for path in paths]
        for name, paths in sources.items() if paths and all(Path(path).is_file() for path in paths)
    }
    run_config = config.to_dict() | {
        "corpus_manifest_sha256": file_sha256(manifest_path),
        "input_hashes": input_hashes,
        "validation_text_hashes": [file_sha256(path) for path in validation_shards],
        "batch_size": args.batch_size, "hidden_size": args.hidden_size,
        "learning_rate": args.learning_rate, "minimum_learning_rate": args.minimum_learning_rate,
        "warmup_steps": args.warmup_steps, "weight_decay": args.weight_decay,
        "max_grad_norm": args.max_grad_norm,
        "evaluate_every": args.evaluate_every,
        "evaluation_batches": args.evaluation_batches,
        "device": str(device),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "resolved_config.json").write_text(
        json.dumps(run_config, indent=2) + "\n", encoding="utf-8"
    )
    if not args.resume:
        (args.output_dir / "evaluations.jsonl").write_text("", encoding="utf-8")
    checkpoint_path = args.output_dir / "latest.pt"
    saved_data_state = {}
    prior_validation = None
    pending_retention_drop = 0.0
    pending_lesson_event = False
    if args.resume:
        resume = load_checkpoint(args.resume, model=model, optimizer=optimizer, config=run_config,
                                 tokenizer_path=tokenizer_path, map_location="cpu")
        trainer.load_state_dict(resume["extra_state"]["trainer"])
        saved_data_state = resume["data_state"]
        prior_validation = resume["extra_state"].get("validation_loss")
        saved_sampler_state = resume["extra_state"].get("sampler_rng")
        pending_retention_drop = float(resume["extra_state"].get("pending_retention_drop", 0.0))
        pending_lesson_event = bool(resume["extra_state"].get("pending_lesson_event", False))
    else:
        saved_sampler_state = None

    streams = {name: _batches(paths, args.batch_size, config.seed + i * 37,
                              saved_data_state.get(name))
               for i, (name, paths) in enumerate(sources.items()) if name in available}
    validation_paths = {
        "text": validation_shards,
        "same_state": [world_root / "state_views.jsonl"] if world_root else [],
        "transition": [world_root / "transitions.jsonl"] if world_root else [],
        "logic": [world_root / "logic.jsonl"] if world_root else [],
        "dialogue": [path.resolve() for path in args.dialogue_jsonl],
    }
    validation_streams = {
        name: _batches(validation_paths[name], args.batch_size, config.seed + 909 + i * 41,
                       saved_data_state.get(f"validation_{name}"), split="validation")
        for i, name in enumerate(available)
    }
    sampler_rng = random.Random(config.seed + trainer.step_index)
    if saved_sampler_state is not None:
        sampler_rng.setstate(saved_sampler_state)
    print(f"device={device} vocab={vocab_size} objectives={','.join(available)} "
          f"steps={architecture.total_steps} start={trainer.step_index}", flush=True)

    while trainer.step_index < architecture.total_steps:
        names = list(streams)
        objective = sampler_rng.choices(names, weights=[sampling[name] for name in names], k=1)[0]
        rows = streams[objective].next_batch()
        if objective == "dialogue":
            rows = [row for row in rows if row.get("split", "train") == "train"]
            if not rows:
                continue
        eval_signals = ({"retention_drop": pending_retention_drop}
                        if pending_lesson_event else {})
        result = trainer.step(
            {objective: rows},
            lambda name, batch: _forward_loss(model, collator, tokenizer, device, name, batch),
            lesson_event=pending_lesson_event,
            eval_signals=eval_signals,
        )
        pending_retention_drop, pending_lesson_event = 0.0, False
        if result.step % args.log_every == 0:
            components = " ".join(f"{name}={value:.4f}" for name, value in result.components.items())
            print(f"step={result.step} loss={result.loss:.5f} objective={objective} {components}", flush=True)
        evaluate_now = result.step % args.evaluate_every == 0
        if evaluate_now:
            validation_metrics = _evaluate_objectives(model, validation_streams, collator,
                                                      tokenizer, device, args.evaluation_batches)
            validation_loss = validation_metrics["text_loss"]
            retention_drop = (max(0.0, (validation_loss - prior_validation) / max(prior_validation, 1e-8))
                              if prior_validation is not None else 0.0)
            pending_retention_drop = retention_drop
            pending_lesson_event = architecture.neuromod_retention_control
            prior_validation = validation_loss
            print(f"step={result.step} validation={json.dumps(validation_metrics, sort_keys=True)} "
                  f"retention_drop={retention_drop:.5f}", flush=True)
            with (args.output_dir / "evaluations.jsonl").open("a", encoding="utf-8") as report:
                report.write(json.dumps({"step": result.step, **validation_metrics}) + "\n")
        checkpoint_now = result.step % architecture.checkpoint_every == 0
        if checkpoint_now:
            trainer.refresh_consolidation_anchor()
            data_state = {name: stream.state_dict() for name, stream in streams.items()}
            data_state.update({f"validation_{name}": stream.state_dict()
                               for name, stream in validation_streams.items()})
            save_checkpoint(checkpoint_path, model=model, optimizer=optimizer, step=result.step,
                            config=run_config, tokenizer_path=tokenizer_path,
                            data_state=data_state, extra_state={"trainer": trainer.state_dict(),
                                                              "validation_loss": prior_validation,
                                                              "sampler_rng": sampler_rng.getstate(),
                                                              "pending_retention_drop": pending_retention_drop,
                                                              "pending_lesson_event": pending_lesson_event})

    data_state = {name: stream.state_dict() for name, stream in streams.items()}
    data_state.update({f"validation_{name}": stream.state_dict()
                       for name, stream in validation_streams.items()})
    save_checkpoint(checkpoint_path, model=model, optimizer=optimizer, step=trainer.step_index,
                    config=run_config, tokenizer_path=tokenizer_path, data_state=data_state,
                    extra_state={"trainer": trainer.state_dict(), "validation_loss": prior_validation,
                                 "sampler_rng": sampler_rng.getstate(),
                                 "pending_retention_drop": pending_retention_drop,
                                 "pending_lesson_event": pending_lesson_event})
    print(f"training complete at step {trainer.step_index}; checkpoint={checkpoint_path}", flush=True)


if __name__ == "__main__":
    main()
