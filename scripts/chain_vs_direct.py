"""Does going A -> B -> C agree with going A -> C directly?

With a model per directed part pair, an arrangement can be built either way. If pairwise
conditioning were sufficient, the two routes would land in the same place. Where they do not is
where a part depends on more than the one part it was conditioned on - an argument, measured
rather than asserted, for modelling the parts jointly.

Both routes are scored against the real C from the same segment, so "the chain disagrees with
the direct model" can be separated from "the chain is worse".

Two controls, because agreement over 32 sparse steps is high by default and because a decoder
that ignores its input would make the two routes agree perfectly for the wrong reason:

  copy A    predicting C by reusing A's rhythm unchanged - the bar any route has to clear
  distinct  how many different patterns the 200 inputs produced. If this is near 1 the route is
            answering the same way regardless of its input, and its agreement scores say nothing
            about conditioning.
"""

import argparse
import glob
import itertools
import json
import os

import numpy as np
import torch
from rhythmic_relationships import (
    MODELS_DIR,
)
from rhythmic_relationships.data import (
    get_hits_from_hits_seq,
    load_co_occurring_hits,
    tokenize_hits,
)
from rhythmic_relationships.evaluate import hits_inference
from rhythmic_relationships.model_utils import load_model
from rhythmic_relationships.models.hits_encdec import TransformerEncoderDecoder

PARTS = ["Drums", "Bass", "Melody", "Harmony"]


def find_models(run_filter):
    """The newest model.pt for each directed part pair under hits_encdec."""
    models = {}
    for path in sorted(glob.glob(os.path.join(MODELS_DIR, "hits_encdec", "*", "*", "model.pt"))):
        run = os.path.basename(os.path.dirname(path))
        pair = os.path.basename(os.path.dirname(os.path.dirname(path)))
        if run_filter and run_filter not in run:
            continue
        if "_" not in pair:
            continue
        a, b = pair.split("_", 1)
        if a not in PARTS or b not in PARTS:
            continue
        if pair not in models or os.path.getmtime(path) > os.path.getmtime(models[pair]):
            models[pair] = path
    return models


def run(model, hits_batch, part_out, n_steps, sampler, device):
    src = torch.tensor(
        np.stack([tokenize_hits(h, block_size=1) for h in hits_batch]),
        dtype=torch.long,
        device=device,
    )
    seqs = hits_inference(
        model=model, src=src, n_tokens=n_steps, temperature=1.0,
        device=device, sampler=sampler, nucleus_p=0.92,
    )
    return np.stack([
        np.asarray(get_hits_from_hits_seq(r.cpu().numpy(), part=part_out, block_size=1),
                   dtype=np.float32)
        for r in seqs
    ])


def agreement(a, b):
    return float(((a > 0) == (b > 0)).mean())


def distinct(patterns):
    return len({tuple((p > 0).astype(int)) for p in patterns})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_seqs", type=int, default=200)
    parser.add_argument("--sampler", type=str, default="greedy")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--run_filter", type=str, default=None,
                        help="only use runs whose name contains this")
    parser.add_argument("--outdir", type=str,
                        default=os.path.join(MODELS_DIR, "hits_encdec", "chain_vs_direct"))
    args = parser.parse_args()

    paths = find_models(args.run_filter)

    # A triple needs three models: A->B, B->C and the A->C it is compared against. Run the ones
    # that are covered rather than demanding all twelve, and name what each missing one costs -
    # a quietly shorter table would read as a complete result.
    triples, skipped = [], []
    for a, b, c in itertools.permutations(PARTS, 3):
        needed = [f"{a}_{b}", f"{b}_{c}", f"{a}_{c}"]
        absent = [pair for pair in needed if pair not in paths]
        (skipped if absent else triples).append(
            (a, b, c) if not absent else ((a, b, c), absent)
        )
    if not triples:
        raise SystemExit(
            f"No triple is covered. Trained pairs: {', '.join(sorted(paths)) or 'none'}"
        )
    print(f"{len(triples)} of {len(triples) + len(skipped)} triples covered")
    for (a, b, c), absent in skipped:
        print(f"  skipping {a}→{b}→{c}: no {', '.join(absent)}")

    used = sorted({pair for a, b, c in triples for pair in (f"{a}_{b}", f"{b}_{c}", f"{a}_{c}")})
    paths = {pair: paths[pair] for pair in used}
    parts_used = sorted({p for a, b, c in triples for p in (a, b, c)})

    loaded, dataset_name, n_steps = {}, None, None
    for pair, path in paths.items():
        model, config = load_model(path, TransformerEncoderDecoder)
        model.eval()
        model.to(args.device)
        loaded[pair] = model
        dataset_name = dataset_name or config["data"]["dataset_name"]
        n_steps = n_steps or config["model"]["context_len"]
        print(f"  {pair:<16} {os.path.basename(os.path.dirname(path))}")

    segments = load_co_occurring_hits(dataset_name, parts_used, args.n_seqs, args.seed,
                                      per_file=1)
    n = len(segments[parts_used[0]])
    print(f"\n{n} segments carrying {', '.join(parts_used)}, sampler={args.sampler}\n")

    header = (f"{'route':<26}{'direct vs true':>15}{'chain vs true':>14}"
              f"{'chain vs direct':>16}{'copy A':>8}{'distinct direct':>16}"
              f"{'distinct chain':>15}")
    print(header)
    print("-" * len(header))

    rows = []
    for a, b, c in triples:
        direct = run(loaded[f"{a}_{c}"], segments[a], c, n_steps, args.sampler, args.device)
        middle = run(loaded[f"{a}_{b}"], segments[a], b, n_steps, args.sampler, args.device)
        chained = run(loaded[f"{b}_{c}"], middle, c, n_steps, args.sampler, args.device)
        truth = segments[c]

        row = {
            "a": a, "b": b, "c": c,
            "direct_vs_true": agreement(direct, truth),
            "chain_vs_true": agreement(chained, truth),
            "chain_vs_direct": agreement(chained, direct),
            # Predicting C by just copying A's rhythm - the bar any route has to clear
            "copy_vs_true": agreement(segments[a], truth),
            "distinct_direct": distinct(direct),
            "distinct_chain": distinct(chained),
        }
        rows.append(row)
        print(f"{a[:2]}→{b[:2]}→{c[:2]}  (vs {a[:2]}→{c[:2]}){'':<7}"
              f"{row['direct_vs_true']:>15.3f}{row['chain_vs_true']:>14.3f}"
              f"{row['chain_vs_direct']:>16.3f}{row['copy_vs_true']:>8.3f}"
              f"{row['distinct_direct']:>16}{row['distinct_chain']:>15}")

    print("-" * len(header))
    print(f"{'mean':<26}{np.mean([r['direct_vs_true'] for r in rows]):>15.3f}"
          f"{np.mean([r['chain_vs_true'] for r in rows]):>14.3f}"
          f"{np.mean([r['chain_vs_direct'] for r in rows]):>16.3f}"
          f"{np.mean([r['copy_vs_true'] for r in rows]):>8.3f}"
          f"{np.mean([r['distinct_direct'] for r in rows]):>16.0f}"
          f"{np.mean([r['distinct_chain'] for r in rows]):>15.0f}")
    print(f"\n{n} inputs went in. A route whose 'distinct' is near 1 is not conditioning on "
          f"them,\nand its agreement columns cannot be read as evidence about conditioning.")

    os.makedirs(args.outdir, exist_ok=True)
    out = os.path.join(args.outdir, f"chain_vs_direct_{args.sampler}.json")
    with open(out, "w") as f:
        json.dump({"dataset": dataset_name, "n_seqs": n, "sampler": args.sampler,
                   "models": {k: os.path.dirname(v) for k, v in paths.items()},
                   "rows": rows}, f, indent=2)
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
