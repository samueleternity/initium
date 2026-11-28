"""Tokenizer training and loading with an explicit special-token contract."""

from __future__ import annotations

from pathlib import Path

SPECIAL_TOKENS = [
    "<pad>", "<unk>", "<bos>", "<eos>", "<system>", "<user>", "<assistant>",
    "<tool>", "<turn_end>", "<state>", "<action>", "<memory>", "<refusal>",
]


def train_tokenizer(text_files, output_path: str | Path, vocab_size: int = 32_000) -> dict:
    """Train a byte-level BPE tokenizer; requires the optional ``tokenizers`` extra."""
    try:
        from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
    except ImportError as exc:  # pragma: no cover - depends on optional extra
        raise RuntimeError("Install the 'text' extra (tokenizers) to train the language tokenizer") from exc
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=2,
        special_tokens=SPECIAL_TOKENS,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    )
    tokenizer.train([str(Path(p)) for p in text_files], trainer)
    if tokenizer.token_to_id("<pad>") != 0:
        raise RuntimeError("the model token-ID input path requires <pad> to have token ID 0")
    tokenizer.save(str(output))
    return {
        "path": str(output), "vocab_size": tokenizer.get_vocab_size(),
        "special_tokens": {token: tokenizer.token_to_id(token) for token in SPECIAL_TOKENS},
        "model": "byte-level-bpe",
    }


def load_tokenizer(path: str | Path):
    try:
        from tokenizers import Tokenizer
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("Install the 'text' extra (tokenizers) to load the language tokenizer") from exc
    tokenizer = Tokenizer.from_file(str(path))
    missing = [token for token in SPECIAL_TOKENS if tokenizer.token_to_id(token) is None]
    if missing:
        raise ValueError(f"tokenizer is missing required special tokens: {missing}")
    if tokenizer.token_to_id("<pad>") != 0:
        raise ValueError("the model token-ID input path requires <pad> to have token ID 0")
    return tokenizer
