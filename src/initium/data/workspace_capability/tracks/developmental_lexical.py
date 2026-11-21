"""Stage 3a (real-word grounding via Stage 2 world + stable lexicon) and 3b (senses)."""

from __future__ import annotations

from contextlib import contextmanager, nullcontext

from initium.data.workspace_capability.episode import (
    BOS, EOS, EV_KV, QUERY, SEP, TaskEpisode, count_token, entity_token as e, fact_token, scalar_token,
)
from initium.data.workspace_capability.tracks import _dev_lexicon as dl
from initium.data.workspace_capability.tracks import _dev_semantics as ds
from initium.data.workspace_capability.tracks import developmental, production

TRACK = "developmental_lexical"


@contextmanager
def _stable_words():
    saved = developmental._lexicon, production._lexicon
    developmental._lexicon = production._lexicon = dl.stable_lexicon
    try:
        yield
    finally:
        developmental._lexicon, production._lexicon = saved


def generate(difficulty, rng, *, ood=False):
    grp = rng.choice(tuple(difficulty["groups"]))
    if grp.startswith("s_"):
        return _sense_episode(difficulty, rng, grp, ood)
    stable = dl.settings["lexicon_mode"] == "stable"
    keyed = not dl.settings["unkeyed"]
    cfg = {**difficulty, "groups": (grp,), "families": (grp,)}
    with (_stable_words() if stable else nullcontext()):
        if grp.startswith("d"):
            ep = developmental.generate(cfg, rng, ood=ood)
        else:
            ep = production.generate(cfg, rng, ood=ood, keyed=keyed, track=TRACK)
    md = {**ep.metadata, "track": TRACK, "stage3": True, "stage3_words": stable,
          "lexicon_mode": dl.settings["lexicon_mode"],
          "band": "B" if ep.metadata.get("heldout_words") else "A", "counterfactual": False}
    if not keyed:
        md.pop("workspace_output_keys", None)
    return TaskEpisode(TRACK, ep.inputs, ep.answer, md)


def _sense_episode(d, rng, grp, ood):
    """Sense records: word record at key k (EV_KV word), category record at key 32+k (EV_KV COUNT token).
    s_disambig answers with the category-record entity (32+k): the slot the argmatch program matches."""
    L = dl.load()
    held = rng.random() < float(d["heldout"])
    band = "B" if held else "A"
    cand = [w for w in L.bands[band] if w in L.senses]
    poly = [w for w in cand if len(L.senses[w]) == 2]
    pw = rng.choice(poly) if poly and (grp == "s_disambig" or rng.random() < float(d["polysemy"])) else None
    if grp == "s_disambig" and pw is None:
        raise ValueError("band has no polysemous words with two mapped categories")
    budget = max(3, int(d["senses"]))
    words, mono, picks = [], [], []
    if pw is not None:
        words.append(pw)
        picks += [(pw, c) for c, _ in L.senses[pw]]
    pool = [w for w in cand if w != pw]
    rng.shuffle(pool)
    while pool and (len(picks) < budget or not mono):  # always >=1 monosemous word
        w = pool.pop()
        words.append(w)
        mono.append(w)
        picks.append((w, L.senses[w][0][0]))
    if dl.settings["lexicon_mode"] == "episodic":
        tok = dict(zip(words, rng.sample(range(64, 128) if held else range(64), len(words))))
    else:
        tok = {w: w for w in words}
    recs = [(k, w, c) for k, (w, c) in zip(rng.sample(range(24), len(picks)), picks)]
    ev = [[EV_KV, e(k), fact_token(tok[w]), SEP] for k, w, c in recs]
    ev += [[EV_KV, e(32 + k), count_token(c), SEP] for k, w, c in recs]
    rng.shuffle(ev)
    stream = [BOS] + [t for x in ev for t in x]
    if grp == "s_s2w":
        k, w, c = rng.choice(recs)
        stream += [QUERY, e(k), scalar_token(0), SEP]
        ans, extra, base = (fact_token(tok[w]), EOS), {"workspace_output_keys": (k, -1)}, 100.0 / len(words)
    elif grp == "s_s2c":
        k, w, c = rng.choice(recs)
        stream += [QUERY, e(k), scalar_token(1), SEP]
        ans, extra = (count_token(c), EOS), {"workspace_output_keys": (32 + k, -1)}
        base = 100.0 / len({r[2] for r in recs})
    elif grp == "s_w2s":
        k, w, c = rng.choice([r for r in recs if r[1] in mono])
        stream += [QUERY, fact_token(tok[w]), SEP]
        ans, base = (e(k), EOS), 100.0 / len(recs)
        extra = {"workspace_output_programs": ((0, "argmatch", tuple(r[0] for r in recs),
                                                tuple(fact_token(tok[r[1]]) for r in recs)),)}
    else:  # s_disambig
        pr = [r for r in recs if r[1] == pw]
        k, w, c = rng.choice(pr)
        stream += [QUERY, fact_token(tok[pw]), count_token(c), SEP]
        ans, base = (e(32 + k), EOS), 50.0
        extra = {"workspace_output_programs": ((0, "argmatch", tuple(32 + r[0] for r in pr),
                                                tuple(count_token(r[2]) for r in pr)),)}
    if dl.settings["unkeyed"]:
        extra.pop("workspace_output_keys", None)
    R = {k: tok[w] for k, w, c in recs}
    reads = tuple((len(stream) - 1, 32 + k, count_token(c)) for k, w, c in recs)
    qtype = f"{grp}_{'poly' if pw is not None else 'mono'}_{'heldout' if held else 'train'}"
    return TaskEpisode(TRACK, tuple(stream), ans,
                       ds.meta(TRACK, stream, grp, qtype, base, held, ood, R, "direct", band=band,
                               reads=reads, polysemous=pw is not None, **extra))