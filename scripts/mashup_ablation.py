"""Does the model contribute anything to the mashups, or is the search doing the work?

The mashup layer is retrieval: generate a rhythm for the target part, then find the real segments
closest to it. Cycle consistency showed these models barely carry their input forward, which
raises the question of what the generated query is adding. If the query is effectively "a typical
rhythm for this part", the segments retrieved for your melody are the segments that would be
retrieved for anyone's.

Four arms, all retrieving from the same pool and all scored as partners for the *same* melody:

  real       the Bass actually recorded with that melody - the ceiling
  matched    query = the model's answer to this melody
  mismatched query = the model's answer to a different melody - kills the conditioning, keeps
             everything else, so matched vs mismatched isolates what conditioning is worth
  random     a real segment picked without reference to the melody at all - the floor

Scored on the relationship rather than on step agreement, because step agreement over 32 sparse
steps is dominated by both parts being mostly empty. The pair descriptors from the thesis
(onset_balance, antiphony) go through the same OA/KLD comparison the model evals use: how closely
does each arm's distribution of relationships match the distribution real pairs have?

That distributional score turns out to be nearly marginal-determined - a random real segment
scores as well as a real partner - so it cannot on its own tell pairing from plausibility. The
run therefore also reports a conditional test: shown one melody, its real partner and one
imposter, how often do the descriptors prefer the real partner? At 0.5 they carry no information
about who plays with whom, whatever their distribution looks like.
"""

import argparse
import json
import os

import numpy as np
import pandas as pd
import torch
from chain_vs_direct import load_triples, run
from export_web import collect_segments
from pair_descriptors import get_antiphony, get_onset_balance
from rhythmic_relationships import MODELS_DIR
from rhythmic_relationships.evaluate import compute_oa_and_kld, compute_oa_kld_dists
from rhythmic_relationships.model_utils import load_model
from rhythmic_relationships.models.hits_encdec import TransformerEncoderDecoder

PAIRED = ["onset_balance", "antiphony"]


def describe_pairs(a, b):
    """The two paired descriptors for each row of two aligned batches of hits."""
    ta = (torch.tensor(a, dtype=torch.float32) > 0).to(int)
    tb = (torch.tensor(b, dtype=torch.float32) > 0).to(int)
    # Both descriptors divide by an onset count, so a silent part has no relationship to report
    keep = (ta.sum(axis=1) > 0) & (tb.sum(axis=1) > 0)
    if not bool(keep.any()):
        return pd.DataFrame(columns=PAIRED), 0
    ta, tb = ta[keep], tb[keep]
    return (
        pd.DataFrame({
            "onset_balance": get_onset_balance(ta, tb).numpy(),
            "antiphony": get_antiphony(ta, tb).numpy(),
        }),
        int((~keep).sum()),
    )


def nearest(queries, pool_hits, pool_files):
    """Index of the closest pool segment to each query.

    Mirrors the rule the interface uses: rank by how many of the 32 steps disagree about having
    an onset, break ties on velocity distance. One per source file, so a query cannot be answered
    by ten takes of the same piece - here only the top hit is kept, so that reduces to skipping
    nothing, but the ranking has to match what the page would show first.
    """
    q_on = queries > 0
    p_on = pool_hits > 0
    picks = []
    for i in range(len(queries)):
        apart = (p_on != q_on[i]).sum(axis=1)
        vel = np.abs(pool_hits - queries[i]).sum(axis=1)
        picks.append(int(np.lexsort((vel, apart))[0]))
    return np.array(picks)


def discrimination(inputs, real, pool_hits, rng, n_imposters=1):
    """Can the pair descriptors pick a real partner out of a lineup?

    A 2D Gaussian on the real pairs' (onset_balance, antiphony) stands in for "this is what a
    real relationship looks like". Each melody's real partner is scored against imposters drawn
    from the pool, and the statistic is how often the real one wins - 0.5 being chance.
    """
    reference, _ = describe_pairs(inputs, real)
    mu = reference[PAIRED].values.mean(axis=0)
    cov = np.cov(reference[PAIRED].values, rowvar=False)
    inv = np.linalg.inv(cov)

    def plausibility(a, b):
        df, keep = describe_pairs_keep(a, b)
        d = df[PAIRED].values - mu
        # Negative Mahalanobis distance: higher is more like a real pair
        return -np.einsum("ij,jk,ik->i", d, inv, d), keep

    real_score, real_keep = plausibility(inputs, real)
    wins, total = 0.0, 0
    for _ in range(n_imposters):
        picks = rng.integers(0, len(pool_hits), size=len(inputs))
        imp_score, imp_keep = plausibility(inputs, pool_hits[picks])
        both = real_keep & imp_keep
        rs = np.full(len(inputs), np.nan)
        rs[real_keep] = real_score
        isc = np.full(len(inputs), np.nan)
        isc[imp_keep] = imp_score
        cmp = rs[both] - isc[both]
        wins += float((cmp > 0).sum() + 0.5 * (cmp == 0).sum())
        total += int(both.sum())
    return wins / max(total, 1), total


def describe_pairs_keep(a, b):
    """`describe_pairs` but returning which rows survived, so scores can be aligned."""
    ta = (torch.tensor(a, dtype=torch.float32) > 0).to(int)
    tb = (torch.tensor(b, dtype=torch.float32) > 0).to(int)
    keep = ((ta.sum(axis=1) > 0) & (tb.sum(axis=1) > 0)).numpy()
    ta, tb = ta[keep], tb[keep]
    return pd.DataFrame({
        "onset_balance": get_onset_balance(ta, tb).numpy(),
        "antiphony": get_antiphony(ta, tb).numpy(),
    }), keep


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True, help="the A -> B model")
    parser.add_argument("--n_seqs", type=int, default=500)
    parser.add_argument("--pool", type=int, default=30000)
    parser.add_argument("--sampler", type=str, default="nucleus")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seed", type=int, default=13)
    # Each melody faces this many imposters. One lineup per melody leaves the win rate too noisy
    # to separate a small effect from chance.
    parser.add_argument("--imposters", type=int, default=20)
    parser.add_argument("--outdir", type=str,
                        default=os.path.join(MODELS_DIR, "hits_encdec", "mashup_ablation"))
    args = parser.parse_args()

    model, config = load_model(args.model_path, TransformerEncoderDecoder)
    model.eval()
    model.to(args.device)
    part_1, part_2 = config["data"]["part_1"], config["data"]["part_2"]
    dataset_name = config["data"]["dataset_name"]
    n_steps = config["model"]["context_len"]
    print(f"{part_1} -> {part_2}, {os.path.basename(os.path.dirname(args.model_path))}")

    segments = load_triples(dataset_name, [part_1, part_2], args.n_seqs, args.seed)
    inputs, real = segments[part_1], segments[part_2]
    n = len(inputs)

    pool = collect_segments(dataset_name, part_2, args.pool, args.seed + 1, 4, with_pitches=False)
    pool_hits = np.array([[v / 4 for v in seg["h"]] for seg in pool], dtype=np.float32)
    pool_files = [seg["n"].split(" seg ")[0] for seg in pool]
    print(f"{n} real {part_1}/{part_2} pairs, {len(pool_hits)} {part_2} segments to search\n")

    generated = run(model, inputs, part_2, n_steps, args.sampler, args.device)
    rng = np.random.default_rng(args.seed)

    # Rolling by one keeps every query a genuine model output while detaching it from its melody
    arms = {
        "matched": nearest(generated, pool_hits, pool_files),
        "mismatched": nearest(np.roll(generated, 1, axis=0), pool_hits, pool_files),
        "random": rng.integers(0, len(pool_hits), size=n),
    }

    reference, dropped = describe_pairs(inputs, real)
    print(f"{'arm':<12}{'OA vs real pairs':>18}{'KLD':>9}{'onset_balance':>15}{'antiphony':>11}"
          f"{'same pick as matched':>22}")
    rows = [{"arm": "real", "oa": 1.0, "kld": 0.0,
             "onset_balance": float(reference["onset_balance"].mean()),
             "antiphony": float(reference["antiphony"].mean()), "overlap": None}]
    print(f"{'real':<12}{'-':>18}{'-':>9}{rows[0]['onset_balance']:>15.3f}"
          f"{rows[0]['antiphony']:>11.3f}{'-':>22}")

    for name, picks in arms.items():
        got = pool_hits[picks]
        df, _ = describe_pairs(inputs, got)
        oa_kld = compute_oa_and_kld(compute_oa_kld_dists(gen_df=df, ref_df=reference))
        overlap = float((picks == arms["matched"]).mean())
        rows.append({
            "arm": name,
            "oa": oa_kld["ref_gen_oa"],
            "kld": oa_kld["ref_gen_kld"],
            "onset_balance": float(df["onset_balance"].mean()),
            "antiphony": float(df["antiphony"].mean()),
            "overlap": overlap,
            "distinct_picks": int(len(set(picks.tolist()))),
        })
        oa, kld = oa_kld["ref_gen_oa"], oa_kld["ref_gen_kld"]
        print(f"{name:<12}{oa:>18.3f}{kld:>9.4f}{rows[-1]['onset_balance']:>15.3f}"
              f"{rows[-1]['antiphony']:>11.3f}{overlap:>22.3f}")

    print(f"\ndistinct segments retrieved, out of {n} melodies:")
    for r in rows[1:]:
        print(f"  {r['arm']:<12}{r['distinct_picks']:>6}")
    auc, compared = discrimination(inputs, real, pool_hits, np.random.default_rng(args.seed + 2),
                                   n_imposters=args.imposters)
    se = (0.25 / max(compared, 1)) ** 0.5
    print(f"\nreal partner preferred over an imposter: {auc:.3f} of {compared} lineups "
          f"(0.5 is chance, +/-{1.96 * se:.3f} at 95%)")
    rows.append({"arm": "discrimination", "real_vs_imposter": auc, "n": compared,
                 "ci95": 1.96 * se})

    if dropped:
        print(f"\n{dropped} pairs left out: one of the two parts is silent, so the descriptors "
              f"have no relationship to report")

    os.makedirs(args.outdir, exist_ok=True)
    out = os.path.join(args.outdir, f"{part_1}_{part_2}_{args.sampler}.json")
    with open(out, "w") as f:
        json.dump({"parts": [part_1, part_2], "sampler": args.sampler, "n_seqs": n,
                   "pool": len(pool_hits), "rows": rows}, f, indent=2)
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
