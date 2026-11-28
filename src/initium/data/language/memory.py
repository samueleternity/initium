"""Inspectable persistent-memory policy with provenance and explicit approval."""

from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path

from initium.data.language.schema import Fact


class MemoryStore:
    def __init__(self, facts=()):
        self._facts = list(facts)

    def insert(self, fact: Fact, *, user_approved: bool = False, durable: bool = False):
        if durable and not user_approved:
            raise PermissionError("durable user memory requires explicit user approval")
        updated = []
        for old in self._facts:
            if (old.subject == fact.subject and old.relation == fact.relation
                    and old.valid_to is None and old.valid_from <= fact.valid_from):
                updated.append(Fact(old.subject, old.relation, old.object, old.valid_from,
                                    fact.valid_from, old.source, old.confidence))
            else:
                updated.append(old)
        self._facts = updated
        self._facts.append(fact)

    def retrieve(self, subject: str | None = None, relation: str | None = None, *, at_time=None):
        return [fact for fact in self._facts if (subject is None or fact.subject == subject)
                and (relation is None or fact.relation == relation)
                and (at_time is None or fact.valid_from <= at_time
                     and (fact.valid_to is None or at_time < fact.valid_to))]

    def delete(self, subject: str, relation: str | None = None):
        before = len(self._facts)
        self._facts = [fact for fact in self._facts if not (
            fact.subject == subject and (relation is None or fact.relation == relation)
        )]
        return before - len(self._facts)

    def to_json(self, path: str | Path):
        Path(path).write_text(json.dumps([asdict(fact) for fact in self._facts], indent=2) + "\n",
                              encoding="utf-8")

    @classmethod
    def from_json(cls, path: str | Path):
        rows = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(Fact(**row) for row in rows)

    @property
    def facts(self):
        return tuple(self._facts)
