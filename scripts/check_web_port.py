"""Reference values for the JavaScript in docs/index.html to be checked against.

The static build reimplements three things that already exist in Python: the sampling loop, the
hits vocabulary, and the paired descriptors. Two implementations of one definition drift, so this
emits what the Python says for a fixed set of cases and the browser is asked the same questions.
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
from rhythmic_relationships.vocab import START_IX

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


def greedy_from_onnx(data_dir, hits, n_steps, part_2):
    """The same fixed-length-buffer generation the page does, in Python."""
    encoder = ort.InferenceSession(os.path.join(data_dir, "encoder.onnx"))
    decoder = ort.InferenceSession(os.path.join(data_dir, "decoder.onnx"))

    src = np.array([tokenize_hits(np.array(hits), block_size=1)], dtype=np.int64)
    enc = encoder.run(None, {"src": src})[0]

    tgt = np.zeros((1, n_steps), dtype=np.int64)
    seq = [START_IX]
    for t in range(n_steps):
        tgt[0, t] = seq[t]
        logits = decoder.run(None, {"tgt": tgt, "enc": enc})[0]
        seq.append(int(logits[0, t].argmax()))
    return get_hits_from_hits_seq(np.array(seq[1:]), part=part_2, block_size=1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="docs/data")
    parser.add_argument("--outfile", type=str, default="docs/data/.port_check.json")
    args = parser.parse_args()

    with open(os.path.join(args.data_dir, "meta.json")) as f:
        meta = json.load(f)
    n_steps, part_2 = meta["n_steps"], meta["part_2"]

    cases = {"tokenize": {}, "greedy": {}, "paired": {}}

    for name, hits in PATTERNS.items():
        cases["tokenize"][name] = tokenize_hits(np.array(hits), block_size=1)

    for name in ["quarters", "eighths", "sparse", "mixed"]:
        cases["greedy"][name] = greedy_from_onnx(args.data_dir, PATTERNS[name], n_steps, part_2)

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

    with open(args.outfile, "w") as f:
        json.dump({"patterns": PATTERNS, "cases": cases}, f, indent=1)
    print(f"Saved {args.outfile}")
    print(f"  {len(cases['tokenize'])} tokenizations, {len(cases['greedy'])} greedy generations, "
          f"{len(cases['paired'])} descriptor pairs")


if __name__ == "__main__":
    main()
