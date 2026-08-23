"""Feed a pair of opposite models their own output and see where the loop goes.

Melody -> Bass -> Melody -> ... If the two models have learned a relationship rather than a
marginal distribution over parts, the loop should stay near where it started and stay varied.
Two failure modes are worth telling apart, and neither shows up in a loss curve:

  collapse   every chain converges to the same pattern, so the conditioning is being ignored
  drift      each chain wanders away from its starting point and never settles

Both are measured per iteration: agreement with the previous iterate (has it settled), agreement
with the starting melody (how far it has gone), and how many distinct patterns survive across
chains (has it collapsed).

Agreement over 32 steps is high by default - two unrelated sparse patterns agree about most
steps simply by both being mostly empty - so every run also pairs each output with a *different*
chain's seed. That control is the only thing that makes "agrees with start" readable: what
matters is the gap between the two, not the absolute number.
"""

import argparse
import glob
import json
import os
import random

import matplotlib.pyplot as plt
import numpy as np
import torch
from rhythmic_relationships import DATASETS_DIR, REPRESENTATIONS_DIRNAME, REPRESENTATIONS_FILENAME
from rhythmic_relationships.data import get_hits_from_hits_seq, tokenize_hits
from rhythmic_relationships.evaluate import hits_inference
from rhythmic_relationships.model_utils import load_model
from rhythmic_relationships.models.hits_encdec import TransformerEncoderDecoder


def load_seed_hits(dataset_name, part, n_seqs, seed):
    """Real segments of `part` to start the chains from."""
    dataset_dir = os.path.join(DATASETS_DIR, dataset_name)
    with open(os.path.join(dataset_dir, REPRESENTATIONS_FILENAME)) as f:
        hits_ix = f.read().split(",").index("hits")

    paths = glob.glob(
        os.path.join(dataset_dir, REPRESENTATIONS_DIRNAME, "**", "*.npz"), recursive=True
    )
    random.Random(seed).shuffle(paths)

    seeds = []
    for path in paths:
        with np.load(path, allow_pickle=True) as npz:
            for key in [k for k in npz.files if k.endswith(f"_{part}")]:
                seeds.append(np.asarray(npz[key][0][hits_ix], dtype=np.float32))
                if len(seeds) >= n_seqs:
                    return np.stack(seeds)
    if not seeds:
        raise SystemExit(f"No {part} segments in {dataset_dir}")
    return np.stack(seeds)


def step(model, hits_batch, n_steps, sampler, temperature, nucleus_p, device, part_out):
    """One pass through a model for a whole batch of inputs."""
    src = torch.tensor(
        np.stack([tokenize_hits(h, block_size=1) for h in hits_batch]),
        dtype=torch.long,
        device=device,
    )
    seqs = hits_inference(
        model=model,
        src=src,
        n_tokens=n_steps,
        temperature=temperature,
        device=device,
        sampler=sampler,
        nucleus_p=nucleus_p,
    )
    return np.stack(
        [
            np.asarray(
                get_hits_from_hits_seq(row.cpu().numpy(), part=part_out, block_size=1),
                dtype=np.float32,
            )
            for row in seqs
        ]
    )


def agreement(a, b):
    """Fraction of steps where two batches of patterns agree about carrying an onset."""
    return ((a > 0) == (b > 0)).mean(axis=1)


def distinct(patterns):
    return len({tuple((p > 0).astype(int)) for p in patterns})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_ab", type=str, required=True, help="A -> B checkpoint")
    parser.add_argument("--model_ba", type=str, required=True, help="B -> A checkpoint")
    parser.add_argument("--n_seqs", type=int, default=200)
    parser.add_argument("--iters", type=int, default=8)
    parser.add_argument("--sampler", type=str, default="greedy")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--nucleus_p", type=float, default=0.92)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--outdir", type=str, default=None)
    args = parser.parse_args()

    model_ab, config_ab = load_model(args.model_ab, TransformerEncoderDecoder)
    model_ba, config_ba = load_model(args.model_ba, TransformerEncoderDecoder)
    for model in (model_ab, model_ba):
        model.eval()
        model.to(args.device)

    part_a = config_ab["data"]["part_1"]
    part_b = config_ab["data"]["part_2"]
    if (config_ba["data"]["part_1"], config_ba["data"]["part_2"]) != (part_b, part_a):
        raise SystemExit(
            f"{args.model_ba} is {config_ba['data']['part_1']} -> "
            f"{config_ba['data']['part_2']}, not {part_b} -> {part_a}"
        )
    n_steps = config_ab["model"]["context_len"]
    if config_ba["model"]["context_len"] != n_steps:
        raise SystemExit("The two models disagree about how many steps a segment has")

    torch.manual_seed(args.seed)
    start = load_seed_hits(config_ab["data"]["dataset_name"], part_a, args.n_seqs, args.seed)
    print(f"{part_a} -> {part_b} -> {part_a}, {len(start)} chains, "
          f"{args.iters} round trips, sampler={args.sampler}")

    kwargs = dict(
        n_steps=n_steps, sampler=args.sampler, temperature=args.temperature,
        nucleus_p=args.nucleus_p, device=args.device,
    )

    current = start
    rows = []
    print(f"\n{'iter':>4}  {'vs previous':>12}  {'vs start':>10}  {'vs other start':>15}  "
          f"{'kept':>6}  {'onsets':>7}  {'distinct':>9}  {'between chains':>14}")
    print(f"{0:>4}  {'-':>12}  {1.0:>10.3f}  "
          f"{float(agreement(start, start[::-1]).mean()):>15.3f}  {'-':>6}  "
          f"{float((start > 0).sum(1).mean()):>7.1f}  "
          f"{distinct(start):>9}  {float(1 - agreement(start, start[::-1]).mean()):>14.3f}")

    for i in range(1, args.iters + 1):
        middle = step(model_ab, current, part_out=part_b, **kwargs)
        nxt = step(model_ba, middle, part_out=part_a, **kwargs)

        vs_start = float(agreement(nxt, start).mean())
        # The same output against someone else's seed. Anything above this is what the loop has
        # actually retained; the rest is two sparse patterns agreeing about empty steps.
        vs_other = float(agreement(nxt, start[::-1]).mean())
        row = {
            "iter": i,
            "vs_previous": float(agreement(nxt, current).mean()),
            "vs_start": vs_start,
            "vs_other_start": vs_other,
            "kept": vs_start - vs_other,
            "onsets": float((nxt > 0).sum(1).mean()),
            "distinct": distinct(nxt),
            # Mean disagreement between different chains: 0 means they have all become one pattern
            "between_chains": float(1 - agreement(nxt, nxt[::-1]).mean()),
        }
        rows.append(row)
        print(f"{i:>4}  {row['vs_previous']:>12.3f}  {row['vs_start']:>10.3f}  "
              f"{row['vs_other_start']:>15.3f}  {row['kept']:>+6.3f}  "
              f"{row['onsets']:>7.1f}  {row['distinct']:>9}  {row['between_chains']:>14.3f}")
        current = nxt

    outdir = args.outdir or os.path.join(
        os.path.dirname(args.model_ab), f"cycle_{part_a}_{part_b}_{args.sampler}"
    )
    os.makedirs(outdir, exist_ok=True)
    with open(os.path.join(outdir, "cycle.json"), "w") as f:
        json.dump({"parts": [part_a, part_b], "sampler": args.sampler,
                   "n_seqs": len(start), "rows": rows}, f, indent=2)

    iters = [r["iter"] for r in rows]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(iters, [r["vs_previous"] for r in rows], marker="o", label="agrees with previous")
    ax.plot(iters, [r["vs_start"] for r in rows], marker="o", label="agrees with start")
    ax.plot(iters, [r["vs_other_start"] for r in rows], marker="o", ls="--",
            label="agrees with another chain's start")
    ax.plot(iters, [r["between_chains"] for r in rows], marker="o", label="differs between chains")
    ax.set_xlabel(f"{part_a} → {part_b} → {part_a} round trips")
    ax.set_ylim(0, 1)
    ax.legend()
    ax.set_title(f"{part_a}/{part_b} cycle, {len(start)} chains, {args.sampler}")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "cycle.png"), dpi=140)
    print(f"\nSaved {outdir}/cycle.json and cycle.png")


if __name__ == "__main__":
    main()
