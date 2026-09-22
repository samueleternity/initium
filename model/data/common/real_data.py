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


def text_token_stream(path: str, vocab_size: int = LABEL_RANGE) -> List[int]:
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
def _load_frames(path: str):
    try:
        import cv2
    except ImportError as e:
        raise ImportError(
            "video real-data loading requires `opencv-python` (`pip install opencv-python`). "
            f"Original import error: {e}"
        )
    cap = cv2.VideoCapture(path)
    frames = []
    ok, frame = cap.read()
    while ok:
        frames.append(frame)
        ok, frame = cap.read()
    cap.release()
    if not frames:
        raise ValueError(f"{path}: no frames read (unsupported codec / bad path?)")
    return frames, cv2


def video_token_stream(path: str, label_range: int = LABEL_RANGE, grid: int = 8) -> List[int]:
    """Each frame is downsampled to a `grid`x`grid` grayscale thumbnail and
    its mean intensity quantized to [0, label_range) -- a coarse "scene
    brightness/shape" codebook, analogous in spirit to audio_token_stream's
    dominant-frequency-bin codebook. Adequate to build a real, non-synthetic
    fact stream out of an actual video file without a full CV pipeline."""
    frames, cv2 = _load_frames(path)
    ids = []
    for frame in frames:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        small = cv2.resize(gray, (grid, grid), interpolation=cv2.INTER_AREA)
        mean_intensity = float(small.mean())  # 0..255
        token_id = int(mean_intensity / 255.0 * (label_range - 1))
        ids.append(max(0, min(label_range - 1, token_id)))
    return ids


# ==========================================================================
# shared: token stream -> fact pool -> train/test split
# ==========================================================================
def build_kv_pool_from_tokens(token_ids: List[int], label_range: int = LABEL_RANGE) -> List[Tuple[int, int]]:
    """Consecutive-pair (bigram) counts -> one (key, most_common_value) fact
    per distinct key seen. This is what turns an arbitrary token stream
    into the same 'glossary' shape build_episode()/evaluate_chain() already
    expect (a KV dictionary, not a raw sequence)."""
    token_ids = [t % label_range for t in token_ids]
    counts: dict = {}
    for a, b in zip(token_ids, token_ids[1:]):
        counts.setdefault(a, Counter())[b] += 1
    pool = [(k, v_counter.most_common(1)[0][0]) for k, v_counter in counts.items()]
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