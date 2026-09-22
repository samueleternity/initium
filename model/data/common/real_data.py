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

Note for later:
Audio computes torch.stft over the whole waveform in one call — a real streaming fix there needs a windowed-STFT 
rewrite, which is out of scope here; flagging it rather than pretending to fix it.
"""
from __future__ import annotations

import glob as _glob
import os
import random
from collections import Counter
from typing import List, Tuple

LABEL_RANGE = 1000


def resolve_link_paths(link) -> List[str]:
    """Expand a dataset link into a sorted, de-duplicated list of one or
    more file paths. Accepts: a single file path; a directory (every
    regular file directly inside it is used -- e.g. "/data/audio_clips" to
    pull in a whole folder); a glob pattern (e.g. "/data/audio_clips/*.wav");
    or several of these '+'-joined in one string (e.g.
    "clip_a.wav+clip_b.wav"), or an actual list/tuple of specs.

    Each resolved path is treated as an independent SOURCE by the
    build_*_kv_pool() functions below: its own token stream/bigram counts
    are folded into the SAME shared pool, but never stitched across a file
    boundary the way within-file chunking is (see _accumulate_bigram_chunk)
    -- two unrelated files (e.g. 5 audio clips with different frequency
    content) have no real adjacency between them, so treating their
    boundary as a genuine bigram would inject spurious facts. This is the
    single place multi-file/"pass a whole folder" support lives; every
    caller (text/audio/video real-data pools, and data/common/graph_io.py's
    multi-file edge loading) goes through this function, so no per-modality
    globbing logic is duplicated.

    Sorted (not glob's arbitrary OS order) so the resulting pool is
    reproducible across runs/machines for the same directory/pattern.
    """
    if isinstance(link, (list, tuple)):
        specs = list(link)
    else:
        specs = str(link).split("+")

    paths: List[str] = []
    for spec in specs:
        spec = spec.strip()
        if not spec:
            continue
        if any(ch in spec for ch in "*?["):
            matches = sorted(p for p in _glob.glob(spec) if os.path.isfile(p))
            if not matches:
                raise FileNotFoundError(f"resolve_link_paths: glob {spec!r} matched no files")
            paths.extend(matches)
        elif os.path.isdir(spec):
            matches = sorted(p for p in _glob.glob(os.path.join(spec, "*")) if os.path.isfile(p))
            if not matches:
                raise FileNotFoundError(f"resolve_link_paths: directory {spec!r} contains no files")
            paths.extend(matches)
        elif os.path.isfile(spec):
            paths.append(spec)
        else:
            raise FileNotFoundError(f"resolve_link_paths: {spec!r} is not a file, directory, or glob match")

    seen, deduped = set(), []
    for p in paths:
        if p not in seen:
            seen.add(p)
            deduped.append(p)
    if not deduped:
        raise FileNotFoundError(f"resolve_link_paths: {link!r} resolved to no files")
    return deduped


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


def _text_token_stream_hf_chunks(path: str, vocab_size: int):
    """Same train step as _text_token_stream_hf, but YIELDS each encoded
    chunk instead of concatenating every chunk into one corpus-length array
    -- this is the actual fix for the 2GB-corpus OOM (see module docstring):
    the old function's np.concatenate materialized the whole token stream
    before build_kv_pool_from_tokens ever ran, on top of THAT function's own
    3-4x blowup (now also fixed, see build_kv_pool_from_tokens)."""
    tok = _HFTokenizer(_HFBPEModel(unk_token=None))
    tok.pre_tokenizer = _HFByteLevel(add_prefix_space=False)
    trainer = _HFBpeTrainer(vocab_size=vocab_size, min_frequency=2, show_progress=False)
    tok.train([path], trainer=trainer)  # streams the file itself; no full-corpus string needed

    for chunk_text in _iter_text_chunks(path):
        ids = np.asarray(tok.encode(chunk_text).ids, dtype=np.int64)
        bad = (ids < 0) | (ids >= vocab_size)
        if bad.any():
            raise ValueError(
                f"_text_token_stream_hf_chunks: {int(bad.sum())} token id(s) fell outside "
                f"[0, {vocab_size}) -- tokenizer vocab is misconfigured; fix the "
                "trainer's vocab_size instead of folding ids with modulo."
            )
        yield ids


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

def build_text_kv_pool(path, label_range: int = LABEL_RANGE) -> List[Tuple[int, int]]:
    """Streaming AND multi-file: `path` may be a single text file, or a
    directory / glob pattern / '+'-joined list of these (see
    resolve_link_paths) -- e.g. an entire folder of text files, each
    contributing its own bigram-derived facts to one shared pool. Each
    resolved file is trained/encoded independently (the HF path streams +
    trains its own BPE vocabulary per file rather than assuming unrelated
    files share one byte-pair vocabulary) and folded into the same fixed
    count matrix; a chunk's cross-boundary bigram is only ever stitched
    WITHIN one file (see _accumulate_bigram_chunk), never across files.
    Only the `tokenizers`-unavailable fallback still fully materializes
    each file in memory (BPETokenizer.train() itself caps at
    _BPE_TRAIN_CHARS_CAP; see text_token_stream's own docstring)."""
    paths = resolve_link_paths(path)
    matrix = np.zeros(label_range * label_range, dtype=np.int64)
    for p in paths:
        if _HF_TOKENIZERS_AVAILABLE:
            with open(p, encoding="utf-8", errors="ignore") as f:
                if not f.read(1):
                    raise ValueError(f"{p}: empty text file")
            prev_last = None
            for chunk in _text_token_stream_hf_chunks(p, label_range):
                prev_last = _accumulate_bigram_chunk(matrix, chunk, label_range, prev_last)
        else:
            tokens = np.asarray(text_token_stream(p, label_range), dtype=np.int64)
            _accumulate_bigram_chunk(matrix, tokens, label_range, None)
    return _pool_from_bigram_matrix(matrix, label_range)

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
    try:
        # Works on a plain audio file OR a video container's audio track --
        # e.g. the same path used for the "video" modality in a
        # "video:clip.mp4+audio:clip.mp4" multimodal spec (see
        # data/multimodal/multimodal_dataset.py) -- as long as torchaudio's
        # backend can demux it.
        waveform, sample_rate = torchaudio.load(path)
    except Exception as e:
        raise RuntimeError(
            f"audio real-data loading: torchaudio could not decode an audio stream from "
            f"'{path}'. If this is a video container (mp4/mkv/...) with an embedded audio "
            "track, torchaudio needs an ffmpeg-capable backend to demux it -- a system "
            "ffmpeg install alongside `pip install torchaudio` usually fixes this. "
            f"Original error: {e}"
        ) from e
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


def build_audio_kv_pool(path, label_range: int = LABEL_RANGE) -> List[Tuple[int, int]]:
    """Multi-file counterpart of build_text_kv_pool/build_video_kv_pool for
    audio: `path` may be a single audio file, or a directory / glob
    pattern / '+'-joined list of these (see resolve_link_paths) -- e.g. an
    entire folder of clips with different frequency/amplitude profiles,
    each contributing its own facts to one shared pool instead of only the
    first file being read. Each file's dominant-frequency-bin token stream
    is folded into the same fixed count matrix independently -- no bigram
    is stitched across a file boundary (two unrelated clips have no real
    temporal adjacency)."""
    paths = resolve_link_paths(path)
    matrix = np.zeros(label_range * label_range, dtype=np.int64)
    for p in paths:
        tokens = np.asarray(audio_token_stream(p, label_range), dtype=np.int64)
        _accumulate_bigram_chunk(matrix, tokens, label_range, None)
    return _pool_from_bigram_matrix(matrix, label_range)


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

def build_video_kv_pool(path, label_range: int = LABEL_RANGE, grid: int = 8,
                        chunk_frames: int = 200_000) -> List[Tuple[int, int]]:
    """Streaming AND multi-file: `path` may be a single video file, or a
    directory / glob pattern / '+'-joined list of these (see
    resolve_link_paths) -- e.g. an entire folder of clips. Quantizes and
    folds frames into the bigram count matrix chunk_frames at a time (never
    building one `means` list for a whole video), and resets the
    within-file stitching state (prev_last) at the start of EACH file, so
    no bigram is stitched across a file boundary."""
    paths = resolve_link_paths(path)
    matrix = np.zeros(label_range * label_range, dtype=np.int64)
    def _flush(buf):
        arr = np.asarray(buf, dtype=np.float32)
        return np.clip((arr / 255.0 * (label_range - 1)).round(), 0, label_range - 1).astype(np.int64)
    for p in paths:
        prev_last = None
        buf = []
        for frame, cv2 in _iter_video_frames(p):
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            small = cv2.resize(gray, (grid, grid), interpolation=cv2.INTER_AREA)
            buf.append(float(small.mean()))
            if len(buf) >= chunk_frames:
                prev_last = _accumulate_bigram_chunk(matrix, _flush(buf), label_range, prev_last)
                buf = []
        if buf:
            prev_last = _accumulate_bigram_chunk(matrix, _flush(buf), label_range, prev_last)
    return _pool_from_bigram_matrix(matrix, label_range)

# ==========================================================================
# shared: token stream -> fact pool -> train/test split
# ==========================================================================
def build_kv_pool_from_tokens(token_ids, label_range: int = LABEL_RANGE) -> List[Tuple[int, int]]:
    """Consecutive-pair (bigram) counts -> one (key, most_common_value) fact
    per distinct key seen. Accumulates into a FIXED label_range x label_range
    int64 count matrix (8 MB at label_range=1000, independent of corpus
    length) instead of the old np.unique/lexsort pipeline, which built 3-4
    additional corpus-length arrays on top of `token_ids` itself (the modulo
    copy, the packed-pair array, np.unique's own sort buffer) -- exactly
    what turned a multi-GB token stream into a 10-20x-larger peak RSS and
    OOM'd. `token_ids` is consumed in bounded-size slices, not concatenated
    into another extra array, so even a pre-materialized huge list/array
    here doesn't multiply memory the way the old pipeline did.
    IDs are expected to already be valid -- out-of-range ids raise instead
    of being silently folded in via modulo (a silent fold can collide two
    genuinely distinct tokens into the same id).
    NOTE: ties resolve to whichever value the count matrix's argmax picks
    (lowest value index on a tie) -- harmless, this only seeds a coarse
    synthetic/real fact pool."""
    n = len(token_ids)
    if n < 2:
        raise ValueError("real-data source produced 0 distinct facts "
                         "(need at least 4) -- source is too short/uniform to train on.")
    matrix = np.zeros(label_range * label_range, dtype=np.int64)
    prev_last = None
    slice_size = 5_000_000
    for start in range(0, n, slice_size):
        chunk = np.asarray(token_ids[start:start + slice_size], dtype=np.int64)
        prev_last = _accumulate_bigram_chunk(matrix, chunk, label_range, prev_last)
    return _pool_from_bigram_matrix(matrix, label_range)


def _accumulate_bigram_chunk(matrix_flat: np.ndarray, chunk: np.ndarray, label_range: int, prev_last):
    """Folds one chunk's consecutive-pair counts into `matrix_flat` (a flat,
    length label_range**2 int64 array) in place, and returns this chunk's
    last token id so the NEXT chunk's caller can pass it back in as
    `prev_last` (preserves the cross-chunk-boundary bigram that would
    otherwise be lost by chunking). `chunk` must already contain only ids
    in [0, label_range) -- raises rather than silently wrapping out-of-range
    ids with modulo."""
    if chunk.size == 0:
        return prev_last
    bad = (chunk < 0) | (chunk >= label_range)
    if bad.any():
        raise ValueError(f"_accumulate_bigram_chunk: {int(bad.sum())} token id(s) fell "
                         f"outside [0, {label_range}) -- upstream tokenizer/quantizer is "
                         "misconfigured; ids must already be valid.")
    if prev_last is not None:
        matrix_flat[prev_last * label_range + int(chunk[0])] += 1
    if chunk.size >= 2:
        idx = chunk[:-1] * label_range + chunk[1:]
        matrix_flat += np.bincount(idx, minlength=label_range * label_range)
    return int(chunk[-1])


def _pool_from_bigram_matrix(matrix_flat: np.ndarray, label_range: int) -> List[Tuple[int, int]]:
    matrix = matrix_flat.reshape(label_range, label_range)
    row_sums = matrix.sum(axis=1)
    keys_present = np.nonzero(row_sums)[0]
    if keys_present.size < 4:
        raise ValueError(f"real-data source produced only {keys_present.size} distinct facts "
                         "(need at least 4) -- source is too short/uniform to train on.")
    vals = matrix[keys_present].argmax(axis=1)
    return list(zip(keys_present.tolist(), vals.tolist()))


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