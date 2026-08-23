"""Retrieve a partner by how the pair sits together, with no model in the loop.

The ablation showed the generated query adds nothing to a mashup: conditioning it on the melody
or on someone else's melody produced relationships equally close to real ones, and a segment
picked at random did as well. So this asks the question directly - given a melody, which real
segment of the other part makes the most plausible pair? - and scores candidates on the
relationship itself.

Which features to score on is not obvious, and it is not the two paired descriptors: they prefer
a real partner over an imposter 0.392 of the time, which is below chance because a direction
fitted on them does not generalise, so retrieval built on them alone would be worse than random
by construction. Every feature set here is therefore measured the same way before being used to
retrieve:

  balance     onset_balance and antiphony - the thesis pair
  interlock   where the two parts' onsets fall relative to each other: both, only one, neither,
              and how much of each part lands on the beat
  combined    both of the above

The headline is the lineup accuracy - shown a melody, its real partner and one imposter, how
often the score prefers the real one. A feature set at 0.5 cannot drive retrieval whatever its
distributions look like, and saying so is the point of measuring it first. The scorer is fitted on
one half of the real pairs and measured on the other, and imposters are other melodies' real
partners rather than arbitrary segments, so neither the fit nor the population can flatter it.
"""

import argparse
import json
import os

import numpy as np
import pandas as pd
import torch
from export_web import collect_segments
from pair_descriptors import get_antiphony, get_onset_balance
from rhythmic_relationships import MODELS_DIR
from rhythmic_relationships.data import load_co_occurring_hits
from rhythmic_relationships.evaluate import compute_oa_and_kld, compute_oa_kld_dists
from rhythmic_relationships.interlock import (
    PairScore,
    interlock_features,
    lineup_accuracy,
    usable,
)


def balance_features(a, b):
    """The thesis pair: how evenly onsets are shared, and how far apart they sit in time."""
    ta, tb = torch.tensor(a > 0).to(int), torch.tensor(b > 0).to(int)
    return np.stack([
        get_onset_balance(ta, tb).numpy(),
        get_antiphony(ta, tb).numpy(),
    ], axis=1)


FEATURES = {
    "balance": balance_features,
    "interlock": interlock_features,
    "combined": lambda a, b: np.concatenate(
        [balance_features(a, b), interlock_features(a, b)], axis=1
    ),
}


def retrieve(fn, model, inputs, pool, rng, top_k=1, chunk=4000):
    """A plausible partner in the pool for each input, scored pair by pair.

    `top_k` is the same knob as nucleus sampling against greedy. Always taking the single most
    plausible candidate walks every input toward the middle of the real-pair distribution, which
    is the most typical relationship rather than the right one; drawing from the top of the
    ranking keeps the spread.
    """
    best = np.zeros(len(inputs), dtype=int)
    for i in range(len(inputs)):
        row = np.repeat(inputs[i : i + 1], len(pool), axis=0)
        scores = np.full(len(pool), -np.inf)
        for start in range(0, len(pool), chunk):
            stop = min(start + chunk, len(pool))
            keep = usable(row[start:stop], pool[start:stop])
            if keep.any():
                idx = np.arange(start, stop)[keep]
                scores[idx] = model.score(fn(row[idx], pool[idx]))
        if top_k <= 1:
            best[i] = int(scores.argmax())
        else:
            top = np.argpartition(-scores, top_k - 1)[:top_k]
            best[i] = int(rng.choice(top))
    return best


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="lmdc_17243_2bar_4res")
    parser.add_argument("--part_1", type=str, default="Melody")
    parser.add_argument("--part_2", type=str, default="Bass")
    parser.add_argument("--n_seqs", type=int, default=500)
    parser.add_argument("--pool", type=int, default=8000)
    parser.add_argument("--n_retrieve", type=int, default=150,
                        help="How many inputs to actually retrieve for; scoring every input "
                             "against every candidate is the expensive part")
    parser.add_argument("--imposters", type=int, default=40)
    parser.add_argument("--top_k", type=int, nargs="+", default=[1, 20, 200])
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--outdir", type=str,
                        default=os.path.join(MODELS_DIR, "hits_encdec", "relationship_retrieval"))
    args = parser.parse_args()

    segments = load_co_occurring_hits(args.dataset, [args.part_1, args.part_2], args.n_seqs, args.seed)
    inputs, real = segments[args.part_1], segments[args.part_2]
    pool_segs = collect_segments(args.dataset, args.part_2, args.pool, args.seed + 1, 4,
                                 with_pitches=False)
    pool = np.array([[v / 4 for v in s["h"]] for s in pool_segs], dtype=np.float32)
    print(f"{args.part_1} -> {args.part_2}: {len(inputs)} real pairs, {len(pool)} candidates\n")

    # Fitted on one half of the real pairs and scored on the other. Fitting and testing on the
    # same pairs would flatter every feature set, and the wider ones most of all.
    half = len(inputs) // 2
    fit_in, fit_real = inputs[:half], real[:half]
    test_in, test_real = inputs[half:], real[half:]

    fit_rng = np.random.default_rng(args.seed + 1)
    fitted = {}
    print(f"{'features':<12}{'dims':>6}{'prefers the real partner':>26}{'95% ci':>10}")
    for name, fn in FEATURES.items():
        keep = usable(fit_in, fit_real)
        a, b = fit_in[keep], fit_real[keep]
        shuffled = b[fit_rng.permutation(len(b))]
        model = PairScore(fn(a, b), fn(a, shuffled))
        acc, n = lineup_accuracy(model, test_in, test_real,
                                 np.random.default_rng(args.seed + 2), args.imposters,
                                 features=fn)
        fitted[name] = (fn, model, acc, n)
        dims = fn(inputs[:2], real[:2]).shape[1]
        print(f"{name:<12}{dims:>6}{acc:>26.3f}{1.96 * (0.25 / max(n, 1)) ** 0.5:>10.3f}")

    best_name = max(fitted, key=lambda k: fitted[k][2])
    fn, model, acc, _ = fitted[best_name]
    print(f"\nretrieving with '{best_name}' for {args.n_retrieve} inputs")

    sub = slice(half, half + args.n_retrieve)
    rng = np.random.default_rng(args.seed + 3)
    arms = {}
    for k in args.top_k:
        arms[f"top {k}" if k > 1 else "most plausible"] = pool[
            retrieve(fn, model, inputs[sub], pool, rng, top_k=k)
        ]
    arms["random"] = pool[rng.integers(0, len(pool), size=args.n_retrieve)]
    ref_keep = usable(fit_in, fit_real)
    reference = pd.DataFrame(balance_features(fit_in[ref_keep], fit_real[ref_keep]),
                             columns=["onset_balance", "antiphony"])

    print(f"\n{'arm':<16}{'OA vs real pairs':>18}{'KLD':>9}{'distinct':>10}")
    rows = []
    for name, cand in arms.items():
        keep = usable(inputs[sub], cand)
        df = pd.DataFrame(balance_features(inputs[sub][keep], cand[keep]),
                          columns=["onset_balance", "antiphony"])
        oa_kld = compute_oa_and_kld(compute_oa_kld_dists(gen_df=df, ref_df=reference))
        distinct = len({tuple(r) for r in (cand > 0).astype(int)})
        rows.append({"arm": name, "oa": oa_kld["ref_gen_oa"], "kld": oa_kld["ref_gen_kld"],
                     "distinct": distinct})
        print(f"{name:<16}{rows[-1]['oa']:>18.3f}{rows[-1]['kld']:>9.4f}{distinct:>10}")

    os.makedirs(args.outdir, exist_ok=True)
    out = os.path.join(args.outdir, f"{args.part_1}_{args.part_2}.json")
    with open(out, "w") as f:
        json.dump({"parts": [args.part_1, args.part_2], "pool": len(pool),
                   "lineups": {k: {"accuracy": v[2], "n": v[3]} for k, v in fitted.items()},
                   "chosen": best_name, "arms": rows}, f, indent=2)
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
