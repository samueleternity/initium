"""Batch builders for corpus, paired-view, transition, logic, and dialogue rows."""

from __future__ import annotations

import torch


def _encode(tokenizer, text: str, max_length: int):
    ids = tokenizer.encode(text, add_special_tokens=False).ids[:max_length]
    return ids or [tokenizer.token_to_id("<unk>")]


def _pad(sequences, *, pad_id: int, max_length: int | None = None):
    width = max_length or max(len(row) for row in sequences)
    ids = torch.full((len(sequences), width), pad_id, dtype=torch.long)
    mask = torch.zeros((len(sequences), width), dtype=torch.bool)
    for index, sequence in enumerate(sequences):
        row = sequence[:width]
        ids[index, :len(row)] = torch.tensor(row, dtype=torch.long)
        mask[index, :len(row)] = True
    return ids, mask


class LanguageCollator:
    def __init__(self, tokenizer, *, max_length: int = 2048):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.pad_id = tokenizer.token_to_id("<pad>")
        self.eos_id = tokenizer.token_to_id("<eos>")
        self.unk_id = tokenizer.token_to_id("<unk>")
        if None in (self.pad_id, self.eos_id, self.unk_id):
            raise ValueError("tokenizer must define <pad>, <eos>, and <unk>")

    def text(self, rows):
        sequences = [list(row["input_ids"][:self.max_length]) for row in rows]
        ids, mask = _pad(sequences, pad_id=self.pad_id)
        labels = ids.clone().masked_fill(~mask, -100)
        return {"input_ids": ids, "attention_mask": mask, "labels": labels}

    def same_state(self, rows):
        if any(row.get("state_id") is None for row in rows):
            raise ValueError("paired views require explicit state_id labels")
        first = [_encode(self.tokenizer, row["first"], self.max_length) for row in rows]
        second = [_encode(self.tokenizer, row["second"], self.max_length) for row in rows]
        first_ids, first_mask = _pad(first, pad_id=self.pad_id)
        second_ids, second_mask = _pad(second, pad_id=self.pad_id)
        return {"first_ids": first_ids, "first_mask": first_mask,
                "second_ids": second_ids, "second_mask": second_mask,
                "state_ids": [row["state_id"] for row in rows]}

    @staticmethod
    def _state_text(state):
        relations = state["relations"]
        if not relations:
            return f"At time {state['time']}, entities: {', '.join(state['entities'])}."
        return " ".join(
            f"At time {state['time']}, {fact['subject']} {fact['relation']} {fact['object']}."
            for fact in relations
        )

    def transition(self, rows):
        before = [_encode(self.tokenizer, self._state_text(row["before"]), self.max_length) for row in rows]
        after = [_encode(self.tokenizer, self._state_text(row["after"]), self.max_length) for row in rows]
        actions = [_encode(self.tokenizer, row["action"], self.max_length)[0] for row in rows]
        before_ids, before_mask = _pad(before, pad_id=self.pad_id)
        after_ids, after_mask = _pad(after, pad_id=self.pad_id)
        return {"before_ids": before_ids, "before_mask": before_mask,
                "after_ids": after_ids, "after_mask": after_mask,
                "action_ids": torch.tensor(actions, dtype=torch.long),
                "valid": torch.tensor([row["valid"] for row in rows], dtype=torch.bool)}

    def logic(self, rows):
        ids = [_encode(self.tokenizer, row["query"], self.max_length) for row in rows]
        input_ids, mask = _pad(ids, pad_id=self.pad_id)
        label_ids = {
            "valid_transition": 0, "invalid_transition": 1,
            "relation_composition_true": 2, "relation_composition_false": 3,
            "contradiction": 4, "consistent": 5,
            "counterfactual_changed": 6, "counterfactual_unchanged": 7,
        }
        try:
            labels = torch.tensor([label_ids[row["label"]] for row in rows])
        except KeyError as exc:
            raise ValueError(f"unknown logic label: {exc.args[0]}") from exc
        return {"input_ids": input_ids, "attention_mask": mask, "labels": labels}

    def dialogue(self, encoded_rows):
        ids, masks = _pad([row["input_ids"][:self.max_length] for row in encoded_rows],
                          pad_id=self.pad_id)
        labels = torch.full_like(ids, -100)
        for index, row in enumerate(encoded_rows):
            target = row["labels"][:self.max_length]
            labels[index, :len(target)] = torch.tensor(target, dtype=torch.long)
        return {"input_ids": ids, "attention_mask": masks, "labels": labels}
