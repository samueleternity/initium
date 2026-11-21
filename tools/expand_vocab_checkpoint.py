#!/usr/bin/env python
"""Expand a 320-token (Stage 3) checkpoint to the Stage 4 vocabulary, for `--init-from`.

Legacy rows/columns are copied unchanged; new input columns are N(0, std(old columns)); new output and
slot-decoder rows are zero with bias = min(old bias) - gap, so legacy logits are untouched and new tokens
start improbable. Optional --word-vectors initialises text-piece embeddings (PCA to hidden size) and the
matching output/decoder rows. Optimizer/scaler/RNG state is dropped (this file is for --init-from only).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import torch

sys.path.insert(0, "src")
from initium.data.workspace_capability.episode import TEXT_BASE, VOCAB_SIZE  # noqa: E402
from initium.data.workspace_capability.tracks import _lang_text as lt  # noqa: E402

_IN_ADAPTER = re.compile(r"^backbone(?:\.\d+)?\.in_adapter\.weight$")
_DECODER = "workspace.semantic_value_decoder"
_DROP = ("optimizer_state_dict", "scaler_state_dict", "rng_state", "ood_rng_state")


def grow_columns(w, total, gen):
    out = w.new_empty(w.shape[0], total)
    out[:, : w.shape[1]] = w
    out[:, w.shape[1] :].normal_(0.0, float(w.float().std()), generator=gen)
    return out


def grow_rows(w, b, total, gap):
    weight = w.new_zeros(total, w.shape[1])
    weight[: w.shape[0]] = w
    bias = b.new_full((total,), float(b.min()) - gap)
    bias[: b.shape[0]] = b
    return weight, bias


def piece_vectors(model_path, vectors_path, dim_out, std):
    """{token_id: vector(dim_out)} from a text vector file (`word v1 v2 ...`), PCA-projected and rescaled."""
    import sentencepiece as spm

    sp = spm.SentencePieceProcessor(model_file=model_path)
    wanted = {}
    for i in range(sp.get_piece_size()):
        word = sp.id_to_piece(i).lstrip("\u2581").lower()
        if word.isalpha() and not sp.is_unknown(i) and not sp.is_byte(i):
            wanted.setdefault(word, []).append(i)
    found, rows = [], []
    with open(vectors_path, encoding="utf-8", errors="ignore") as f:
        for line in f:
            word, _, rest = line.partition(" ")
            if word in wanted:
                values = rest.split()
                if len(values) > 8:
                    found.append(word)
                    rows.append(torch.tensor([float(v) for v in values]))
    if not rows or len({r.numel() for r in rows}) != 1:
        raise ValueError("word-vector file yielded no consistent vectors for the tokenizer pieces")
    X = torch.stack(rows)
    X = X - X.mean(0)
    q = min(dim_out, *X.shape)
    _u, _s, v = torch.pca_lowrank(X, q=q, center=False)
    Z = X @ v[:, :q]
    Z = torch.nn.functional.pad(Z, (0, dim_out - q)) * (std / float(Z.std().clamp_min(1e-8)))
    return {TEXT_BASE + i: Z[k] for k, word in enumerate(found) for i in wanted[word]}, len(found)


def expand(ckpt, total, gap, seed, vectors=None, model_path=None):
    cfg = ckpt["model_config"]
    old = int(cfg["input_size"])
    if old != VOCAB_SIZE:
        raise ValueError(f"checkpoint input_size is {old}; expected the legacy {VOCAB_SIZE}")
    gen = torch.Generator().manual_seed(seed)
    rnn, out = dict(ckpt["rnn_state_dict"]), dict(ckpt["output_proj_state_dict"])
    adapters = [k for k, v in rnn.items() if _IN_ADAPTER.match(k) and v.shape[1] == old]
    if len(adapters) != 1:
        raise ValueError(f"expected exactly one backbone input adapter, found {adapters}")
    legacy = {k: rnn[k].clone() for k in (adapters[0], f"{_DECODER}.weight") if k in rnn}
    rnn[adapters[0]] = grow_columns(rnn[adapters[0]], total, gen)
    out["weight"], out["bias"] = grow_rows(out["weight"], out["bias"], total, gap)
    if f"{_DECODER}.weight" in rnn:
        rnn[f"{_DECODER}.weight"], rnn[f"{_DECODER}.bias"] = grow_rows(
            rnn[f"{_DECODER}.weight"], rnn[f"{_DECODER}.bias"], total, gap
        )
    if vectors:
        adapter = rnn[adapters[0]]
        table, n = piece_vectors(
            model_path, vectors, adapter.shape[0], float(legacy[adapters[0]].std())
        )
        for token, vec in table.items():
            adapter[:, token] = vec
            out["weight"][token] = vec
            if f"{_DECODER}.weight" in rnn:
                rnn[f"{_DECODER}.weight"][token] = vec
        print(f"initialised {len(table)} text-piece embeddings from {n} vectors")
    # invariants: legacy slices untouched
    assert torch.equal(rnn[adapters[0]][:, :old], legacy[adapters[0]])
    for (
        key,
        value,
    ) in rnn.items():  # tensors still carrying the old vocab size would break load_state_dict
        if (
            key not in (*adapters, f"{_DECODER}.weight", f"{_DECODER}.bias")
            and value.ndim
            and old in value.shape[:2]
        ):
            print(
                f"WARNING: {key} {tuple(value.shape)} still has a legacy-vocabulary dimension (unchanged)"
            )
    result = {k: v for k, v in ckpt.items() if k not in _DROP}
    cfg = dict(
        cfg,
        input_size=total,
        input_dim=total,
        output_dim=total,
        workspace_lang_manifest_sha256=lt.manifest_sha256(),
        workspace_lang_tokenizer_sha256=lt.tokenizer_sha256(),
        workspace_lang_tokenizer_vocab_size=lt.tokenizer_vocab_size(),
    )
    result.update(
        model_config=cfg,
        rnn_state_dict=rnn,
        output_proj_state_dict=out,
        expanded_from={"input_size": old, "total_vocab": total},
    )
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint")
    ap.add_argument("output")
    ap.add_argument("--new-logit-gap", type=float, default=10.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--word-vectors", default=None, help="text file `word v1 v2 ...` (optional embedding init)"
    )
    a = ap.parse_args()
    manifest = json.loads(lt._path("manifest.json").read_text(encoding="utf-8"))
    total = TEXT_BASE + int(manifest["text_vocab_size"])
    ckpt = torch.load(a.checkpoint, map_location="cpu", weights_only=False)
    result = expand(
        ckpt, total, a.new_logit_gap, a.seed, a.word_vectors, str(lt._path("spm.model"))
    )
    Path(a.output).parent.mkdir(parents=True, exist_ok=True)
    torch.save(result, a.output)
    print(f"expanded {VOCAB_SIZE} -> {total} tokens: {a.output}")


if __name__ == "__main__":
    main()
