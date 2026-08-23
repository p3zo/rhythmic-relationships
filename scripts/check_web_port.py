"""Reference values for the JavaScript in docs/index.html to be checked against.

The static build reimplements four things that already exist in Python: the sampling loop, the
hits vocabulary, the paired descriptors, and the relationship distance the mashup search ranks by.
Two implementations of one definition drift, so this emits what the Python says for a fixed set of
cases and the browser is asked the same questions.

The mashup transposition is the exception: it lives only in the page, so `key_shift` below is a
second implementation written from the same definition rather than a call into shipped code. It
is here to disagree if the JavaScript ever changes meaning.
Run it, then `node scripts/check_web_port.mjs` with the page served.
"""

import argparse
import json
import os

import numpy as np
import onnxruntime as ort
import torch

from pair_descriptors import get_antiphony, get_onset_balance
from rhythmic_relationships.data import get_hits_from_hits_seq, tokenize_hits
from rhythmic_relationships.interlock import interlock_features, relationship_distance
from rhythmic_relationships.vocab import START_IX, get_hits_vocab

# Deliberately awkward: empty, full, single onset, and uneven densities
PATTERNS = {
    "quarters": [1.0 if i % 4 == 0 else 0.0 for i in range(32)],
    "eighths": [1.0 if i % 2 == 0 else 0.0 for i in range(32)],
    "sparse": [1.0 if i in (0, 7, 20) else 0.0 for i in range(32)],
    "front": [1.0 if i < 8 else 0.0 for i in range(32)],
    "back": [1.0 if i >= 24 else 0.0 for i in range(32)],
    "mixed": [(0.25 * ((i * 3) % 5)) if i % 3 == 0 else 0.0 for i in range(32)],
    "empty": [0.0] * 32,
    "full": [1.0] * 32,
}


CLASH_PENALTY = 2


def interval_class(a, b):
    d = abs(a - b) % 12
    return min(d, 12 - d)


def key_shift(heard, seg_notes):
    """Semitones to move seg_notes so it sits with the notes in `heard`.

    Both are [step, pitch, velocity]. Reference for the page's keyShift.
    """
    if not seg_notes or not heard:
        return 0

    share = [0.0] * 12
    for _, pitch, vel in heard:
        share[pitch % 12] += vel
    total = sum(share)
    if not total:
        return 0

    by_step = {}
    for step, pitch, vel in heard:
        by_step.setdefault(step, []).append((pitch, vel))

    best, best_score = 0, float("-inf")
    for t in range(-6, 6):
        score = 0.0
        for step, pitch, vel in seg_notes:
            pc = (pitch + t) % 12
            score += vel * share[pc] / total
            for other, other_vel in by_step.get(step, []):
                if interval_class(pc, other % 12) in (1, 6):
                    score -= CLASH_PENALTY * vel * other_vel
        if score > best_score + 1e-9 or (abs(score - best_score) < 1e-9 and abs(t) < abs(best)):
            best, best_score = t, score
    return best


def greedy_from_onnx(model_dir, hits, n_steps, part_2):
    """The same fixed-length-buffer generation the page does, in Python.

    Also the onset probability the page draws its P(onset) row from, which is the mass the step's
    own distribution puts on the tokens that decode to an onset.
    """
    encoder = ort.InferenceSession(os.path.join(model_dir, "encoder.onnx"))
    decoder = ort.InferenceSession(os.path.join(model_dir, "decoder.onnx"))
    onset_tokens = [k for k, v in get_hits_vocab().items()
                    if not isinstance(v, str) and v > 0]

    src = np.array([tokenize_hits(np.array(hits), block_size=1)], dtype=np.int64)
    enc = encoder.run(None, {"src": src})[0]

    tgt = np.zeros((1, n_steps), dtype=np.int64)
    seq, onset_probs = [START_IX], []
    for t in range(n_steps):
        tgt[0, t] = seq[t]
        logits = decoder.run(None, {"tgt": tgt, "enc": enc})[0]
        row = logits[0, t]
        probs = np.exp(row - row.max())
        probs /= probs.sum()
        onset_probs.append(round(float(probs[onset_tokens].sum()), 6))
        seq.append(int(row.argmax()))
    return {
        "hits": get_hits_from_hits_seq(np.array(seq[1:]), part=part_2, block_size=1),
        "onset_probs": onset_probs,
    }


def key_shift_cases(data_dir, drawn_pitch, part_1, part_2):
    """Real input/segment pairs, plus the cases that are easy to get wrong."""
    with open(os.path.join(data_dir, "parts", part_1, "examples.json")) as f:
        examples = [e for e in json.load(f) if e["p"]]
    with open(os.path.join(data_dir, "parts", part_2, "index.json")) as f:
        index = [seg for seg in json.load(f) if seg["p"]]

    def notes(raw):
        return [[step, pitch, vel / 127] for step, pitch, vel in raw]

    pairs = []
    # Spread across both files rather than taking a contiguous run of either
    for i in range(20):
        pairs.append((notes(examples[i * 7]["p"]), notes(index[i * 137]["p"])))

    heard = notes(examples[0]["p"])
    drawn = [[step, drawn_pitch, 1.0] for step in (0, 4, 8, 12)]
    pairs += [
        (heard, []),
        ([], notes(index[0]["p"])),
        (heard, heard),                                                   # already fits
        (heard, [[s, p + 1, v] for s, p, v in heard]),                     # a semitone off
        (heard, [[s, p + 6, v] for s, p, v in heard]),                     # a tritone off
        (drawn, notes(index[0]["p"])),                                     # drawn grid, one pitch
        (drawn, [[s, drawn_pitch + 1, 1.0] for s in (0, 4, 8, 12)]),       # clashes on every step
    ]
    return [{"heard": h, "seg": g, "shift": key_shift(h, g)} for h, g in pairs]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="docs/data")
    parser.add_argument("--outfile", type=str, default="docs/data/.port_check.json")
    args = parser.parse_args()

    with open(os.path.join(args.data_dir, "meta.json")) as f:
        meta = json.load(f)
    n_steps = meta["n_steps"]

    cases = {"tokenize": {}, "greedy": {}, "paired": {}, "relationship": {}}

    for name, hits in PATTERNS.items():
        cases["tokenize"][name] = tokenize_hits(np.array(hits), block_size=1)

    # Every model on the page, since each has its own exported weights to disagree with
    for model in meta["models"]:
        model_dir = os.path.join(args.data_dir, "models", model["id"])
        for name in ["quarters", "eighths", "sparse", "mixed"]:
            cases["greedy"][f"{model['id']}|{name}"] = greedy_from_onnx(
                model_dir, PATTERNS[name], n_steps, model["part_2"]
            )

    # Per model, because each ships its own targets and covariance for the page to measure
    # against. Silent patterns are left out: the page skips them, as usable() does in the Python.
    for model in meta["models"]:
        precision = np.array(model["relationships"]["precision"])
        # Three of the shipped targets rather than all of them; the arithmetic is the same one
        for t, target in enumerate(np.array(model["relationships"]["targets"][:3])):
            for a, b in ((a, b) for a in PATTERNS for b in PATTERNS
                         if any(PATTERNS[a]) and any(PATTERNS[b])):
                features = interlock_features(np.array([PATTERNS[a]]), np.array([PATTERNS[b]]))
                cases["relationship"][f"{model['id']}|{t}|{a}|{b}"] = round(
                    float(relationship_distance(features, target, precision)[0]), 6
                )

    for a in PATTERNS:
        for b in PATTERNS:
            ta = (torch.tensor([PATTERNS[a]]) > 0).to(int)
            tb = (torch.tensor([PATTERNS[b]]) > 0).to(int)
            if int(ta.sum()) == 0 or int(tb.sum()) == 0:
                cases["paired"][f"{a}|{b}"] = [None, None]
                continue
            cases["paired"][f"{a}|{b}"] = [
                round(float(get_onset_balance(ta, tb)[0]), 6),
                round(float(get_antiphony(ta, tb)[0]), 6),
            ]

    first = meta["models"][0]
    cases["key_shift"] = key_shift_cases(
        args.data_dir, meta["input_pitch"], first["part_1"], first["part_2"]
    )

    # The page has no model menu - the two part dropdowns are the chooser - so the check needs
    # to know which pair each model is
    models = [{"id": m["id"], "part_1": m["part_1"], "part_2": m["part_2"]} for m in meta["models"]]

    with open(args.outfile, "w") as f:
        json.dump({"patterns": PATTERNS, "cases": cases, "models": models}, f, indent=1)
    print(f"Saved {args.outfile}")
    print(f"  {len(cases['tokenize'])} tokenizations, {len(cases['greedy'])} greedy generations, "
          f"{len(cases['paired'])} descriptor pairs, "
          f"{len(cases['relationship'])} relationship distances, "
          f"{len(cases['key_shift'])} key shifts")


if __name__ == "__main__":
    main()
