"""
file: data/common/real_data.py

Turns a modality-specific raw source (a text corpus, an audio file, a video
file) into the ONE thing every data/common/chain_task.py KVChainDataset
needs: a pool of (key, value) integer-label facts, split into a TRAIN pool
(sampled from during curriculum training, replacing the synthetic
rng.sample(range(label_range), ...) draw) and a TEST pool (disjoint keys,
never seen during training - the modality's analogue of the graph
dataset's held-out London Underground graph; see graph_io.py for that one,
which is edge-structured rather than token-structured so it doesn't fit
this module's pipeline).

Pipeline, same for every modality (this is deliberate - see the "reduce
every modality to an integer token stream first" idea from the research
notes):

    raw source -> per-unit integer token id in [0, label_range)
               -> consecutive-pair (bigram) counts
               -> most-common value per key -> fact pool
               -> disjoint train/test key split

What differs per modality is only the FIRST step (raw source -> token
stream): BPE byte-pair merges for text, quantized spectrogram-frame bins
for audio, quantized downsampled-frame intensities for video. Each
modality's dataset module (text_dataset.py / audio_dataset.py /
video_dataset.py) calls its own `*_token_stream()` function below and then
the two shared helpers (`build_kv_pool_from_tokens`, `split_train_test_facts`).

None of this is meant to be a state-of-the-art tokenizer/codec - it's the
minimum real-data pipeline that plugs into the existing digit-coded KV
chain-task machinery unmodified. Swapping in a production BPE
implementation, a real mel-spectrogram front end, or a real video encoder
later only means replacing the token-stream function; build_kv_pool_from_tokens
/ split_train_test_facts and everything downstream (chain_task.py,
training, evaluation) stays the same.
"""
from __future__ import annotations

import random
from collections import Counter
from typing import List, Tuple

LABEL_RANGE = 1000


import numpy as np

try:
    from tokenizers import Tokenizer as _HFTokenizer
    from tokenizers.models import BPE as _HFBPEModel
    from tokenizers.trainers import BpeTrainer as _HFBpeTrainer
    from tokenizers.pre_tokenizers import ByteLevel as _HFByteLevel
    _HF_TOKENIZERS_AVAILABLE = True
except ImportError:
    _HF_TOKENIZERS_AVAILABLE = False

# BPE trainer is O(corpus_len * merges) in pure python -- fine for a
# "reasonable synthetic-real" corpus, not meant for GB-scale text. Cap the
# training sample so a huge file doesn't hang the training script's startup.
_BPE_TRAIN_CHARS_CAP = 300_000


# ==========================================================================
# text: byte-level BPE -> integer token stream
# ==========================================================================
class BPETokenizer:
    """Minimal byte-level BPE (Sennrich et al. 2016). Vocab starts as the
    256 raw bytes and merges the most frequent adjacent pair repeatedly
    until `vocab_size` is reached. `vocab_size` defaults to LABEL_RANGE so
    every emitted token id already fits DigitCodec's label_range=1000
    without any further hashing/modulo -- the one thing this module's
    callers rely on.
    """

    def __init__(self, vocab_size: int = LABEL_RANGE):
        if vocab_size <= 256:
            raise ValueError(f"BPETokenizer: vocab_size must be > 256 (raw bytes), got {vocab_size}")
        self.vocab_size = vocab_size
        self.merges: List[Tuple[int, int]] = []   # ordered (a, b) merges, applied in order at encode time

    def train(self, text: str) -> "BPETokenizer":
        data = text.encode("utf-8", errors="ignore")[:_BPE_TRAIN_CHARS_CAP]
        tokens = list(data)  # ids 0..255
        next_id = 256
        while next_id < self.vocab_size and len(tokens) > 1:
            pairs = Counter(zip(tokens, tokens[1:]))
            if not pairs:
                break
            (a, b), count = pairs.most_common(1)[0]
            if count < 2:
                break
            merged = next_id
            new_tokens, i = [], 0
            while i < len(tokens):
                if i < len(tokens) - 1 and tokens[i] == a and tokens[i + 1] == b:
                    new_tokens.append(merged)
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            tokens = new_tokens
            self.merges.append((a, b))
            next_id += 1
        return self

    def encode(self, text: str) -> List[int]:
        tokens = list(text.encode("utf-8", errors="ignore"))
        next_id = 256
        for a, b in self.merges:
            merged = next_id
            new_tokens, i = [], 0
            while i < len(tokens):
                if i < len(tokens) - 1 and tokens[i] == a and tokens[i + 1] == b:
                    new_tokens.append(merged)
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            tokens = new_tokens
            next_id += 1
        return tokens


_TEXT_STREAM_CHUNK_CHARS = 1_000_000   # read/encode in ~1M-char chunks -- the corpus is
                                        # never materialized whole as one Python str


def _iter_text_chunks(path: str, chunk_chars: int = _TEXT_STREAM_CHUNK_CHARS):
    """Yield the file's text in fixed-size chunks so callers never hold more
    than `chunk_chars` characters at once (the fix for f.read() loading a
    multi-GB corpus into one Python str)."""
    with open(path, encoding="utf-8", errors="ignore") as f:
        while True:
            chunk = f.read(chunk_chars)
            if not chunk:
                return
            yield chunk


def _text_token_stream_hf(path: str, vocab_size: int):
    """Rust-backed BPE (Sennrich et al. 2016) via the `tokenizers` package.
    Trains directly from the file on disk (tokenizers' own file-based
    train() reads it itself, rather than requiring the corpus in memory
    first) and encodes the corpus in fixed-size chunks -- neither step
    materializes the whole file as one Python str or one Python list of
    boxed ints, which is what made the full 2GB TinyStories corpus OOM
    before. Returns a numpy int64 array, not a Python list, for the same
    memory reason on the output side.
    """
    tok = _HFTokenizer(_HFBPEModel(unk_token=None))
    tok.pre_tokenizer = _HFByteLevel(add_prefix_space=False)
    trainer = _HFBpeTrainer(vocab_size=vocab_size, min_frequency=2, show_progress=False)
    tok.train([path], trainer=trainer)  # streams the file itself; no full-corpus string needed

    chunks = [np.asarray(tok.encode(chunk_text).ids, dtype=np.int64)
              for chunk_text in _iter_text_chunks(path)]
    ids = np.concatenate(chunks) if chunks else np.empty(0, dtype=np.int64)

    # ByteLevel BPE's vocabulary is capped at vocab_size by the trainer
    # (alphabet + merges), so ids should already be in [0, vocab_size) --
    # verify instead of silently folding out-of-range ids with modulo,
    # which can collide two genuinely distinct tokens into the same id.
    bad = (ids < 0) | (ids >= vocab_size)
    if bad.any():
        raise ValueError(
            f"_text_token_stream_hf: {int(bad.sum())} token id(s) fell outside "
            f"[0, {vocab_size}) -- tokenizer vocab is misconfigured; fix the "
            "trainer's vocab_size instead of folding ids with modulo."
        )
    return ids


def text_token_stream(path: str, vocab_size: int = LABEL_RANGE) -> List[int]:
    if _HF_TOKENIZERS_AVAILABLE:
        with open(path, encoding="utf-8", errors="ignore") as f:
            if not f.read(1):
                raise ValueError(f"{path}: empty text file")
        return _text_token_stream_hf(path, vocab_size)
    # Fallback pure-python tokenizer only: still needs the full text in
    # memory (BPETokenizer.train() itself caps at _BPE_TRAIN_CHARS_CAP).
    with open(path, encoding="utf-8", errors="ignore") as f:
        text = f.read()
    if not text.strip():
        raise ValueError(f"{path}: empty text file")
    tok = BPETokenizer(vocab_size=vocab_size).train(text)
    return tok.encode(text)


# ==========================================================================
# audio: waveform -> quantized spectrogram-frame token stream
# ==========================================================================
def _load_waveform(path: str):
    try:
        import torchaudio
    except ImportError as e:
        raise ImportError(
            "audio real-data loading requires `torchaudio` (`pip install torchaudio`). "
            f"Original import error: {e}"
        )
    waveform, sample_rate = torchaudio.load(path)
    return waveform.mean(dim=0), sample_rate  # mono


def audio_token_stream(path: str, label_range: int = LABEL_RANGE,
                       n_fft: int = 400, hop_length: int = 160) -> List[int]:
    """Each STFT frame's magnitude spectrum is collapsed to a single integer
    id in [0, label_range) via its dominant-frequency-bin index, rescaled
    into the label range. This is a coarse "spectral shape" codebook, not a
    real acoustic-unit tokenizer -- adequate to build a real, non-synthetic
    fact stream out of an actual audio file."""
    import torch
    waveform, _sr = _load_waveform(path)
    spec = torch.stft(waveform, n_fft=n_fft, hop_length=hop_length,
                      window=torch.hann_window(n_fft), return_complex=True)
    mag = spec.abs()  # (freq_bins, n_frames)
    freq_bins = mag.shape[0]
    dominant_bin = mag.argmax(dim=0)  # (n_frames,) in [0, freq_bins)
    scaled = (dominant_bin.float() / max(freq_bins - 1, 1) * (label_range - 1)).round().long()
    return scaled.clamp(0, label_range - 1).tolist()


# ==========================================================================
# video: frames -> quantized downsampled-frame token stream
# ==========================================================================
def _iter_video_frames(path: str):
    try:
        import cv2
    except ImportError as e:
        raise ImportError(
            "video real-data loading requires `opencv-python` (`pip install opencv-python`). "
            f"Original import error: {e}"
        )
    cap = cv2.VideoCapture(path)
    try:
        n = 0
        ok, frame = cap.read()
        while ok:
            yield frame, cv2
            n += 1
            ok, frame = cap.read()
        if n == 0:
            raise ValueError(f"{path}: no frames read (unsupported codec / bad path?)")
    finally:
        cap.release()

def video_token_stream(path: str, label_range: int = LABEL_RANGE, grid: int = 8) -> List[int]:
    """Each frame is downsampled to a `grid`x`grid` grayscale thumbnail and
    its mean intensity quantized to [0, label_range) -- a coarse "scene
    brightness/shape" codebook, analogous in spirit to audio_token_stream's
    dominant-frequency-bin codebook. Adequate to build a real, non-synthetic
    fact stream out of an actual video file without a full CV pipeline.

    Frames are consumed and discarded one at a time (never collected into a
    full-length list first) -- a long/high-resolution source would otherwise
    hold every original full-size frame in RAM before any processing."""
    means = []
    for frame, cv2 in _iter_video_frames(path):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        small = cv2.resize(gray, (grid, grid), interpolation=cv2.INTER_AREA)
        means.append(float(small.mean()))
    means_arr = np.asarray(means, dtype=np.float32)
    scaled = np.clip((means_arr / 255.0 * (label_range - 1)).round().astype(np.int64),
                     0, label_range - 1)
    return scaled.tolist()


# ==========================================================================
# shared: token stream -> fact pool -> train/test split
# ==========================================================================
def build_kv_pool_from_tokens(token_ids: List[int], label_range: int = LABEL_RANGE) -> List[Tuple[int, int]]:
    """Consecutive-pair (bigram) counts -> one (key, most_common_value) fact
    per distinct key seen. This is what turns an arbitrary token stream
    into the same 'glossary' shape build_episode()/evaluate_chain() already
    expect (a KV dictionary, not a raw sequence).

    Vectorized with numpy instead of a Python dict-of-Counter scan: pack
    each (key, value) bigram into one int64 (key*label_range + value),
    count distinct pairs with np.unique, then take -- per key -- the
    highest-count value via a lexsort (key asc, count desc) + first-per-
    group mask. O(n log n) instead of O(n) with heavy per-element Python
    dict/Counter overhead, and no allocation of a Counter object per key.
    NOTE: ties (two values equally common for the same key) may resolve
    differently than Counter.most_common's insertion-order tie-break --
    harmless here, this only seeds a coarse synthetic/real fact pool."""
    ids = np.asarray(token_ids, dtype=np.int64) % label_range
    if ids.size < 2:
        raise ValueError("real-data source produced 0 distinct facts "
                         "(need at least 4) -- source is too short/uniform to train on.")
    keys, vals = ids[:-1], ids[1:]
    combined = keys * label_range + vals
    uniq, counts = np.unique(combined, return_counts=True)
    u_keys, u_vals = uniq // label_range, uniq % label_range
    order = np.lexsort((-counts, u_keys))          # sort by key, then by count desc within key
    u_keys, u_vals = u_keys[order], u_vals[order]
    first_per_key = np.concatenate(([True], u_keys[1:] != u_keys[:-1]))
    pool = list(zip(u_keys[first_per_key].tolist(), u_vals[first_per_key].tolist()))
    if len(pool) < 4:
        raise ValueError(f"real-data source produced only {len(pool)} distinct facts "
                         "(need at least 4) -- source is too short/uniform to train on.")
    return pool


def split_train_test_facts(pool: List[Tuple[int, int]], test_frac: float = 0.1,
                           seed: int = 1234) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]:
    """Disjoint-by-key split so the test pool is genuinely unseen during
    training -- the token-stream analogue of the graph dataset's
    London-Underground-is-a-different-graph OOD guarantee."""
    pool = list(pool)
    rng = random.Random(seed)
    rng.shuffle(pool)
    n_test = max(1, int(len(pool) * test_frac))
    test_pool, train_pool = pool[:n_test], pool[n_test:]
    if len(train_pool) < 4:
        raise ValueError(f"real-data source only has {len(train_pool)} training facts after "
                         f"reserving {n_test} for the test split -- source is too short.")
    return train_pool, test_pool