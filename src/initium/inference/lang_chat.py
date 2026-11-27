"""Interactive Stage 5 language inference for workspace-capability checkpoints.

Run with ``python -m initium.inference.lang_chat CHECKPOINT.pt``. The model is
queried through the Stage 5 ``lang_act`` prompt convention. Semantic-workspace
checkpoints use their annotated slot interface; parallel-dense checkpoints use
plain token IDs and carry latent workspace state between turns. Mamba sequence
state is recomputed per turn, as in the training/evaluation code.
"""

from __future__ import annotations

import argparse
import os
import sys
from collections import OrderedDict

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import torch  # noqa: E402

from initium.data.workspace_capability.episode import (  # noqa: E402
    ANSWER,
    BOS,
    EOS,
    LANG_TRACK_BASE,
    SEP,
    SPK_SYS,
    SPK_USER,
    TEXT_BASE,
)  # noqa: E402
from initium.data.workspace_capability.tracks import _lang_text  # noqa: E402
from initium.inference.checkpoint_io import load_checkpoint  # noqa: E402
from initium.inference.model_loader import load_model  # noqa: E402
from initium.memory_manipulation.nvrtc_compat import patch_prod_jiterator  # noqa: E402

patch_prod_jiterator()

_PUNCT = ".,?!;:"
_FUNCTION = frozenset(
    "the a an is are was were in can could be found used for to has have had by and not cannot does do did "
    "what where who it they this that yes no i know cause causes caused like describe tell me everything "
    "nobody nothing".split()
)
_STOPS = frozenset((*_FUNCTION, *_PUNCT, "|", ";"))
_SEED_NAMES = ("mary", "john", "sandra")
_SEED_PLACES = ("hallway", "kitchen", "garden")
_ACT_FUNCTION_WORDS = frozenset(
    "the a an is are was were in can could be found used for to has have had by and not cannot does do did "
    "what where who it they this that yes no i know cause causes caused like describe tell me everything "
    "nobody nothing you are welcome".split()
)
_ACT_TARGET_WORDS = frozenset(
    (
        *_ACT_FUNCTION_WORDS,
        *_SEED_NAMES,
        "daniel",
        "emma",
        "peter",
        "apple",
        "ball",
        "book",
        "key",
        "cup",
        "hat",
        *_SEED_PLACES,
        "office",
        "bedroom",
        "bathroom",
        "moved",
        "went",
        "travelled",
        "journeyed",
        "picked",
        "up",
        "took",
        "grabbed",
        "dropped",
        "put",
        "down",
        "discarded",
        "ok",
        "hello",
        "goodbye",
        "thanks",
        "mean",
        "carrying",
        "nothing",
    )
)
_MAX_SEQUENCE = 512
_MAX_USER_TOKENS = 128


def _clone_tree(value):
    if isinstance(value, torch.Tensor):
        return value.clone()
    if isinstance(value, dict):
        return {key: _clone_tree(child) for key, child in value.items()}
    if isinstance(value, list):
        return [_clone_tree(child) for child in value]
    if isinstance(value, tuple):
        return tuple(_clone_tree(child) for child in value)
    return value


class LanguageChat:
    def __init__(
        self,
        loaded,
        device,
        *,
        allocator_routing=False,
        max_new_tokens=64,
        answer_diagnostics=False,
    ):
        self.rnn = loaded.rnn
        self.output_proj = loaded.output_proj
        self.parallel_dense = bool(loaded.config.get("workspace_parallel_dense", False))
        self.teacher_forced_answers = bool(
            loaded.config.get("workspace_teacher_forced_answers", False)
        )
        if getattr(self.rnn, "modulation_head", None) is not None:
            warmup_steps = int(loaded.config.get("modulation_alpha_warmup_steps", 0))
            self.rnn.modulation_head.alpha_scale = min(
                1.0, loaded.step / max(1, warmup_steps)
            )
        self.device = device
        self.tk = _lang_text.tokenizer()
        self.state = None
        self.max_new_tokens = int(max_new_tokens)
        self.answer_diagnostics = bool(answer_diagnostics)
        self.last_answer_diagnostics: dict = {}
        self.entity_keys: OrderedDict[str, int] = OrderedDict()
        self.next_key = 0
        self.last_trace: dict = {}
        self.seeded_facts: list[tuple[str, str]] = []
        self.rnn.workspace_eval_bypass = False
        if not self.parallel_dense and getattr(self.rnn, "allocator", None) is not None:
            # Routing is opt-in: the training run used label-authoritative
            # addresses, so allocator-routed inference is an explicit probe.
            self.rnn.allocator_use_prob = 1.0 if allocator_routing else 0.0

    def _key_for(self, word: str) -> int:
        key = word.lower().strip(_PUNCT + "()\"'")
        if not key:
            return -1
        if key in self.entity_keys:
            self.entity_keys.move_to_end(key)
            return self.entity_keys[key]
        count = int(self.rnn.workspace._SEMANTIC_ENTITY_COUNT)
        if len(self.entity_keys) >= count:
            self.entity_keys.popitem(last=False)
        used = set(self.entity_keys.values())
        address = next((i for i in range(count) if i not in used), self.next_key % count)
        self.next_key = (address + 1) % count
        self.entity_keys[key] = address
        return address

    def _text_words(self, text: str):
        # Stage 4/5 encodes whitespace-delimited words independently. Keep
        # punctuation attached exactly as provided, and lowercase like the
        # synthetic training corpora.
        return text.lower().split()

    def _encode_words(self, words, *, query=False):
        token_ids: list[int] = []
        ids: list[int] = []
        writes: list[bool] = []
        copy_mask: list[bool] = []
        for word in words:
            pieces = list(self.tk.word(word))
            token_ids.extend(pieces)
            cleaned = word.strip(_PUNCT + "()\"'")
            is_stop = cleaned in _STOPS or not cleaned
            source_ok = not is_stop
            copy_mask.extend([source_ok] * len(pieces))
            if self.parallel_dense:
                # Dense workspace addresses are learned from token context;
                # this model has no semantic entity-ID write interface.
                ids.extend([-1] * len(pieces))
                writes.extend([False] * len(pieces))
                continue
            previously_known = cleaned in self.entity_keys
            key = self._key_for(cleaned) if not is_stop else -1
            # Training annotates one semantic address per word, at its final
            # SentencePiece position; continuation pieces remain unaddressed.
            ids.extend([-1] * (len(pieces) - 1) + [key] if key >= 0 else [-1] * len(pieces))
            # A normal observed utterance may establish an unseen lexical
            # value, as in the annotated episodes. User queries are read-only:
            # allocating a new word key must not overwrite conversation memory.
            writes.extend([False] * len(pieces))
            if key >= 0 and not previously_known and not query:
                writes[-1] = True
        return token_ids, ids, writes, copy_mask

    def _forward(self, tokens, entity_ids, write_mask, state):
        if len(tokens) > _MAX_SEQUENCE:
            raise ValueError(
                f"input is {len(tokens)} tokens; Stage 5 supports at most {_MAX_SEQUENCE}"
            )
        x = torch.tensor(tokens, dtype=torch.long, device=self.device).unsqueeze(0)
        hx = _clone_tree(state) if state is not None else (None, None, None)
        if self.parallel_dense:
            output, new_state = self.rnn(
                x,
                hx,
                reset_experience=state is None,
                pass_through_memory=True,
            )
        else:
            eids = torch.tensor(entity_ids, dtype=torch.long, device=self.device).unsqueeze(0)
            wm = torch.tensor(write_mask, dtype=torch.bool, device=self.device).unsqueeze(0)
            output, new_state = self.rnn(
                x,
                hx,
                reset_experience=state is None,
                pass_through_memory=True,
                entity_ids=eids,
                entity_write_mask=wm,
                # In legacy semantic workspaces, map observed mentions into
                # the slot readout's match-key state.
                workspace_match_entity_ids=eids,
                workspace_match_write_mask=eids.ge(0),
            )
        features = output.transpose(0, 1).contiguous()
        logits = self.output_proj(features).float()
        answer_mask = torch.zeros(x.shape, dtype=torch.float32, device=self.device)
        answer_mask[:, -1] = 1.0
        source_mask = torch.tensor(
            [
                [
                    int(t >= TEXT_BASE and ok)
                    for t, ok in zip(
                        tokens, self.last_trace.get("copy_mask", [False] * len(tokens))
                    )
                ]
            ],
            dtype=torch.bool,
            device=self.device,
        )
        source_ids = x
        if not self.parallel_dense and getattr(self.rnn, "workspace_slot_readout_enabled", False):
            logits = self.rnn.apply_slot_readout(
                logits,
                features,
                new_state,
                answer_mask,
                copy_source_token_ids=source_ids,
                copy_source_mask=source_mask,
            )
        return logits[0, -1], new_state

    @torch.inference_mode()
    def seed_world(self):
        """Install a small, explicit world for the intended location-query task."""
        self.state = None
        names = _SEED_NAMES
        places = _SEED_PLACES
        words: list[str] = []
        write_at: dict[int, int] = {}
        entity_ids_by_name = {name: i for i, name in enumerate(names)}
        self.entity_keys.clear()
        if not self.parallel_dense:
            for name, key in entity_ids_by_name.items():
                self.entity_keys[name] = key
        for i, (name, place) in enumerate(zip(names, places)):
            sentence = [name, "is", "in", "the", place + "."]
            base = len(words)
            words.extend(sentence)
            write_at[base + 4] = i
        body, body_ids, body_writes, body_copy = [], [], [], []
        for wi, word in enumerate(words):
            pieces = list(self.tk.word(word))
            body.extend(pieces)
            key = write_at.get(wi)
            if self.parallel_dense:
                body_ids.extend([-1] * len(pieces))
                body_writes.extend([False] * len(pieces))
            elif key is None:
                entity = entity_ids_by_name.get(word.strip(_PUNCT), -1)
                body_ids.extend(
                    [-1] * (len(pieces) - 1) + [entity] if entity >= 0 else [-1] * len(pieces)
                )
                body_writes.extend([False] * len(pieces))
            else:
                body_ids.extend([-1] * (len(pieces) - 1) + [key])
                body_writes.extend([False] * (len(pieces) - 1) + [True])
            content = word.strip(_PUNCT) not in _STOPS
            body_copy.extend([content] * len(pieces))
        prefix = [BOS, LANG_TRACK_BASE + 11, SEP]
        tokens = prefix + body
        ids = [-1] * len(prefix) + body_ids
        writes = [False] * len(prefix) + body_writes
        self.last_trace["copy_mask"] = [False] * len(prefix) + body_copy
        _logits, self.state = self._forward(tokens, ids, writes, None)
        self.seeded_facts = list(zip(names, places))
        self.last_trace = {"copy_mask": []}
        print("Seeded Stage 5 location context:")
        for name, place in zip(names, places):
            print(f"  {name} is in the {place}.")

    @torch.inference_mode()
    def reply(self, text: str) -> str:
        words = self._text_words(text)
        body, body_ids, body_writes, body_copy = self._encode_words(words, query=True)
        if len(body) > _MAX_USER_TOKENS:
            raise ValueError(f"user message is {len(body)} pieces; limit is {_MAX_USER_TOKENS}")
        prefix = [BOS, LANG_TRACK_BASE + 11, SEP, SPK_USER]
        base_tokens = prefix + body + [SPK_SYS]
        base_ids = [-1] * len(prefix) + body_ids + [-1]
        base_writes = [False] * len(prefix) + body_writes + [False]
        base_copy = [False] * len(prefix) + body_copy + [False]
        if self.state is None:
            prior_state = None
        else:
            prior_state = _clone_tree(self.state)
        generated: list[int] = []
        chosen_probabilities: list[float] = []
        emitted_eos = False
        for _ in range(self.max_new_tokens):
            # Training keeps one ANSWER input at the first answer position,
            # then teacher-forces the preceding gold token at each later
            # position. Mirror that shift during generation: ANSWER comes
            # before the generated prefix and appears exactly once.
            answer_prefix = [ANSWER] + generated
            tokens = base_tokens + answer_prefix
            ids = base_ids + [-1] * len(answer_prefix)
            writes = base_writes + [False] * len(answer_prefix)
            self.last_trace["copy_mask"] = base_copy + [False] * len(answer_prefix)
            logits, _ = self._forward(tokens, ids, writes, prior_state)
            next_token = int(logits.argmax(-1).item())
            if next_token == EOS:
                emitted_eos = True
                break
            if next_token < 0 or next_token >= self.tk.total:
                # Stage 5 answers may contain typed MR/control tokens, but the
                # conversational act track is expected to produce text only.
                break
            if self.answer_diagnostics:
                token_log_probability = logits[next_token] - torch.logsumexp(logits, dim=-1)
                chosen_probabilities.append(float(token_log_probability.exp().item()))
            generated.append(next_token)
        # A following training segment receives the state produced by the
        # previous episode's complete input sequence. With the default
        # unforced-answer policy, those answer positions are ANSWER placeholders;
        # when teacher forcing was enabled, they contain the preceding gold
        # answer tokens. Rebuild that carry state rather than persisting the
        # generated text, which training never fed back into later turns.
        answer_positions = len(generated) + int(emitted_eos)
        if answer_positions:
            if self.teacher_forced_answers:
                carry_answer_inputs = [ANSWER] + generated[: max(0, answer_positions - 1)]
            else:
                carry_answer_inputs = [ANSWER] * answer_positions
            carry_tokens = base_tokens + carry_answer_inputs
            carry_ids = base_ids + [-1] * len(carry_answer_inputs)
            carry_writes = base_writes + [False] * len(carry_answer_inputs)
            self.last_trace["copy_mask"] = base_copy + [False] * len(carry_answer_inputs)
            _carry_logits, self.state = self._forward(
                carry_tokens, carry_ids, carry_writes, prior_state
            )
        answer_text = self.tk.decode(generated).strip()
        if self.answer_diagnostics:
            content_words = [
                word.strip(_PUNCT + "()\"'").lower()
                for word in answer_text.split()
                if word.strip(_PUNCT + "()\"'").lower()
                and word.strip(_PUNCT + "()\"'").lower() not in _ACT_FUNCTION_WORDS
            ]
            unsupported = [word for word in content_words if word not in _ACT_TARGET_WORDS]
            self.last_answer_diagnostics = {
                "mean_token_probability": (
                    sum(chosen_probabilities) / len(chosen_probabilities)
                    if chosen_probabilities
                    else None
                ),
                "min_token_probability": min(chosen_probabilities)
                if chosen_probabilities
                else None,
                "ood_fraction": len(unsupported) / len(content_words) if content_words else None,
                "unsupported_words": unsupported,
                "content_word_count": len(content_words),
            }
        self.last_trace = {
            "input": base_tokens,
            "generated": generated,
            "copy_mask": [],
        }
        return answer_text

    def show_slots(self):
        if self.state is None or not isinstance(self.state[0], dict):
            print("Workspace is empty. Use /seed or send a message first.")
            return
        slot_state = self.state[0].get("_initium_workspace_state")
        if slot_state is None:
            print("Checkpoint returned no carried workspace state.")
            return
        if self.parallel_dense:
            norms = slot_state[0].float().norm(dim=-1)
            shown = sorted(
                enumerate(norms.tolist()), key=lambda item: item[1], reverse=True
            )[:40]
            print(f"Parallel workspace slot-value norms (top {len(shown)}):")
            for slot_index, norm in shown:
                print(f"  slot={slot_index:03d} value_norm={norm:.4g}")
            return
        address_dim = self.rnn.workspace._address_dim
        address = slot_state[:, :, :address_dim]
        occupied = address.norm(dim=-1)[0] > 1e-6
        reverse = {value: key for key, value in self.entity_keys.items()}
        shown = []
        for slot_index in torch.nonzero(occupied, as_tuple=False).flatten().tolist():
            key = int(address[0, slot_index].argmax().item())
            shown.append((slot_index, key, reverse.get(key, "?")))
        print(f"Workspace occupied slots: {len(shown)}")
        for slot_index, key, word in shown[:40]:
            print(f"  slot={slot_index:03d} entity_key={key:03d} label={word}")

    def reset(self):
        self.state = None
        self.entity_keys.clear()
        self.next_key = 0
        self.last_trace = {}
        self.seeded_facts.clear()
        print("Conversation workspace reset.")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Interactive Stage 5 lang_act inference.")
    parser.add_argument("checkpoint", help="core_training checkpoint (.pt)")
    parser.add_argument("--device", default=None, help="default: cuda:0 when available, else cpu")
    parser.add_argument("--max-new-tokens", type=int, default=64, choices=range(1, 257))
    parser.add_argument(
        "--seed-world",
        action="store_true",
        help="install a small location-query context before chat",
    )
    parser.add_argument(
        "--allocator-routing",
        action="store_true",
        help="route inference addresses through the learned allocator (experimental; off by default)",
    )
    parser.add_argument(
        "--answer-diagnostics",
        action="store_true",
        help="log mean/min greedy-token probability and a lang_act answer-vocabulary OOD proxy",
    )
    parser.add_argument(
        "--allow-legacy-tokenizer-manifest-mismatch",
        action="store_true",
        help=(
            "allow a legacy checkpoint's whole-corpus manifest to differ after "
            "checking vocabulary size; tokenizer identity cannot be proven"
        ),
    )
    parser.add_argument(
        "--tokenizer-dir",
        default=None,
        help=(
            "directory containing the exact training-time manifest.json and spm.model; "
            "the checkpoint tokenizer hash is still verified"
        ),
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    ckpt = load_checkpoint(args.checkpoint)
    config = ckpt["model_config"]
    if config.get("dataset_type") != "workspace-capability" or config.get(
        "workspace_task_track"
    ) not in ("lang_all", "lang_stage5"):
        raise SystemExit("checkpoint must be trained on workspace-capability lang_all/lang_stage5")
    if not config.get("split_graph_enabled"):
        raise SystemExit("Stage 5 language chat requires a split-graph checkpoint")
    if not config.get("workspace_token_id_input"):
        raise SystemExit("Stage 5 language chat requires token-ID input")
    parallel_dense = bool(config.get("workspace_parallel_dense", False))
    if parallel_dense:
        if int(config.get("workspace_num_slots", 0)) < 1:
            raise SystemExit("parallel dense workspace checkpoint must have at least one slot")
        if args.allocator_routing:
            raise SystemExit("--allocator-routing applies only to semantic workspace checkpoints")
    else:
        if not config.get("workspace_slot_readout"):
            raise SystemExit(
                "checkpoint must include either parallel dense workspace or semantic slot readout"
            )
        if not config.get("workspace_semantic_entity_addressing"):
            raise SystemExit("checkpoint must use semantic entity addressing")
        if int(config.get("workspace_semantic_entity_count", 64)) != 128:
            raise SystemExit("checkpoint must use the Stage 5 128-key semantic workspace")
        if int(config.get("workspace_num_slots", 0)) < 128:
            raise SystemExit("checkpoint must have at least 128 workspace slots")
    device = torch.device(args.device or ("cuda:0" if torch.cuda.is_available() else "cpu"))
    loaded = load_model(
        ckpt,
        device,
        allow_legacy_tokenizer_manifest_mismatch=args.allow_legacy_tokenizer_manifest_mismatch,
        language_tokenizer_dir=args.tokenizer_dir,
    )
    chat = LanguageChat(
        loaded,
        device,
        allocator_routing=args.allocator_routing,
        max_new_tokens=args.max_new_tokens,
        answer_diagnostics=args.answer_diagnostics,
    )
    print(
        f"Loaded {loaded.run_id} step={loaded.step} on {device}; "
        "Stage 5 language chat probe. This is a closed-task model, not an open-domain LM."
    )
    if args.seed_world:
        chat.seed_world()
    print(
        "Commands: /help /seed /reset /slots /facts /trace /bypass on|off /quit"
        + ("" if parallel_dense else " /allocator on|off")
    )
    if args.answer_diagnostics:
        print(
            "Answer diagnostics on: confidence is greedy-token probability; OOD is a lexical support proxy, not a calibrated probability."
        )
    while True:
        try:
            line = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not line:
            continue
        if line in ("/quit", "/exit"):
            break
        if line == "/help":
            print(
                "Type a message; use /seed for a small known location world, /reset to clear state, /slots and /trace to inspect the run."
            )
        elif line == "/seed":
            chat.seed_world()
        elif line == "/reset":
            chat.reset()
        elif line == "/slots":
            chat.show_slots()
        elif line == "/facts":
            print(f"Seed facts: {chat.seeded_facts or 'none'}")
            print(f"Lexical keys: {list(chat.entity_keys.items())[:40]}")
        elif line == "/trace":
            trace = chat.last_trace
            print(f"input token ids: {trace.get('input', [])}")
            print(f"generated token ids: {trace.get('generated', [])}")
            print(
                f"allocator route probability: {getattr(chat.rnn, 'allocator_use_prob', 0.0):.2f}"
            )
        elif line.startswith("/allocator"):
            parts = line.split()
            if len(parts) != 2 or parts[1] not in ("on", "off"):
                print("Usage: /allocator on|off")
            elif getattr(chat.rnn, "allocator", None) is None:
                print("This checkpoint has no learned allocator.")
            else:
                chat.rnn.allocator_use_prob = 1.0 if parts[1] == "on" else 0.0
                print(f"Allocator routing {parts[1]}.")
        elif line.startswith("/bypass"):
            parts = line.split()
            if len(parts) != 2 or parts[1] not in ("on", "off"):
                print("Usage: /bypass on|off")
            else:
                chat.rnn.workspace_eval_bypass = parts[1] == "on"
                print(f"Workspace bypass {parts[1]}.")
        else:
            try:
                answer = chat.reply(line)
            except (ValueError, RuntimeError) as exc:
                print(f"[inference error] {exc}")
                continue
            print(f"model> {answer or '[no text token generated]'}")
            if args.answer_diagnostics:
                d = chat.last_answer_diagnostics
                mean_p, min_p = d.get("mean_token_probability"), d.get("min_token_probability")
                if mean_p is None:
                    print(
                        "[answer diagnostics] confidence=n/a | answer OOD proxy=n/a (no generated text)"
                    )
                else:
                    n_words = int(d["content_word_count"])
                    ood = d.get("ood_fraction")
                    ood_label = (
                        f"{100.0 * ood:.0f}% ({len(d['unsupported_words'])}/{n_words} content words outside lang_act targets)"
                        if ood is not None
                        else "n/a (no content words)"
                    )
                    print(
                        f"[answer diagnostics] confidence={100.0 * mean_p:.0f}% mean token probability "
                        f"(min {100.0 * min_p:.0f}%) | answer OOD proxy={ood_label}"
                    )
                    if d["unsupported_words"]:
                        print(f"  unsupported answer words: {', '.join(d['unsupported_words'])}")


if __name__ == "__main__":
    main()
