"""Stage 4 runtime: SentencePiece wrapper, settings, manifest (no network)."""

from __future__ import annotations

import functools
import hashlib
import json
from pathlib import Path

from initium.data.workspace_capability.episode import TEXT_BASE

LANG_SINGLE = (
    "lang_template", "lang_nlg", "lang_story_qa", "lang_dialogue", "lang_syntax_probes", "lang_open",
    "lang_parse", "lang_define", "lang_induce", "lang_logic", "lang_discourse", "lang_act", "lang_nlu_external",
)
LANG_V5_TRACKS = LANG_SINGLE[6:]
RETIRED = ("lang_open",)
LANG_TRACK_NAMES = (*LANG_SINGLE, "lang_all")
DATA_DIR = Path(__file__).resolve().parent.parent / "stage4"
EXTERNAL_FILES = {"nlg": "nlg.jsonl", "story": "story.jsonl", "accept": "accept.jsonl", "open": "open.jsonl",
                  "nlu_triples": "nlu_triples.jsonl", "nli": "nli.jsonl", "qa": "qa.jsonl"}
settings = {"annotator": True, "shuffled_context": False, "external": False, "identity": "truth", "stage5": False}


def configure(annotator=None, shuffled_context=None, external=None, identity=None, stage5=None):
    for key, value in (("annotator", annotator), ("shuffled_context", shuffled_context),
                       ("external", external), ("stage5", stage5)):
        if value is not None:
            settings[key] = bool(value)
    if identity is not None:
        if identity not in ("annotator", "truth"):
            raise ValueError("identity must be 'annotator' or 'truth'")
        settings["identity"] = identity


def _path(name):
    path = DATA_DIR / name
    if not path.exists():
        raise FileNotFoundError(f"{path} is missing; run tools/build_stage4_corpus.py first")
    return path


def manifest_sha256():
    return hashlib.sha256(_path("manifest.json").read_bytes()).hexdigest()


def ext_available(name):
    return (DATA_DIR / EXTERNAL_FILES[name]).exists()


def available_tracks():
    return tuple(
        t for t in LANG_SINGLE
        if t not in RETIRED and (t != "lang_nlu_external" or all(ext_available(n) for n in ("nlu_triples", "nli", "qa")))
    )


class LangTokenizer:
    """SentencePiece unigram; text piece id p -> token TEXT_BASE + p. Words are encoded independently
    (pieces never cross whitespace; equivalence with sentence-level encoding is verified by the builder)."""

    def __init__(self):
        try:
            import sentencepiece as spm
        except ImportError as exc:  # pragma: no cover
            raise ImportError("Stage 4 requires `pip install sentencepiece`") from exc
        manifest = json.loads(_path("manifest.json").read_text(encoding="utf-8"))
        model = _path("spm.model")
        if hashlib.sha256(model.read_bytes()).hexdigest() != manifest["sha256"]["spm.model"]:
            raise ValueError("spm.model does not match manifest.json (stale or corrupted tokenizer)")
        self.sp = spm.SentencePieceProcessor(model_file=str(model))
        self.text_size = self.sp.get_piece_size()
        if self.text_size != int(manifest["text_vocab_size"]):
            raise ValueError("tokenizer size differs from manifest")
        self.total = TEXT_BASE + self.text_size
        self._cache: dict[str, tuple[int, ...]] = {}

    def word(self, w: str) -> tuple[int, ...]:
        r = self._cache.get(w)
        if r is None:
            r = tuple(TEXT_BASE + i for i in self.sp.encode(w, out_type=int))
            if len(self._cache) >= 262_144:
                self._cache.clear()
            self._cache[w] = r
        return r

    def encode(self, text: str) -> list[int]:
        return [t for w in text.split() for t in self.word(w)]

    def decode(self, ids) -> str:
        p = [int(t) - TEXT_BASE for t in ids if int(t) >= TEXT_BASE]
        return self.sp.decode(p) if p else ""


@functools.lru_cache(maxsize=1)
def tokenizer() -> LangTokenizer:
    return LangTokenizer()