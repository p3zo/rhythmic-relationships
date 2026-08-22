import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from scipy import integrate, stats
from scipy.spatial import distance_matrix
from matplotlib import rcParams
from rhythmic_relationships.vocab import get_hits_vocab

sns.set_style("white")
sns.set_context("paper")

rcParams["figure.figsize"] = 11.7, 8.27  # fig size in inches


def temperatured_softmax(logits, temperature):
    """Adapted from https://github.com/YatingMusic/MuseMorphose/blob/069570279db65ffb3914ca9aacab8061badfacb3/generate.py#L41-L50"""
    scaled = logits / temperature

    # Subtracting the max leaves the result unchanged but keeps exp() from overflowing
    scaled = scaled - scaled.max(axis=-1, keepdims=True)

    exp = np.exp(scaled)
    probs = exp / exp.sum(axis=-1, keepdims=True)

    if np.count_nonzero(np.isnan(probs)) > 0:
        raise ValueError(f"Softmax produced NaNs for {temperature=}")

    return probs


def nucleus(probs, p):
    """Adapted from https://github.com/YatingMusic/MuseMorphose/blob/069570279db65ffb3914ca9aacab8061badfacb3/generate.py#L52-L66"""
    probs /= sum(probs)
    sorted_probs = np.sort(probs)[::-1]
    sorted_ixs = np.argsort(probs)[::-1]
    cumsum_sorted_probs = np.cumsum(sorted_probs)
    after_thresh = cumsum_sorted_probs > p
    if after_thresh.sum() > 0:
        # The first index above the threshold is the smallest pool whose mass covers p. Taking
        # the last one instead keeps all but the least likely token, which is no truncation.
        first_index = np.where(after_thresh)[0][0]
        candidate_ixs = sorted_ixs[: first_index + 1]
    else:
        # just assign a value
        candidate_ixs = sorted_ixs[:3]
    candidate_probs = np.array([probs[i] for i in candidate_ixs], dtype=np.float64)
    candidate_probs /= sum(candidate_probs)
    return np.random.choice(candidate_ixs, size=1, p=candidate_probs)[0]


def hits_inference(
    model,
    src,
    n_tokens,
    temperature,
    device,
    sampler="multinomial",
    nucleus_p=0.92,
):
    # TODO: move to a new hits module in models dir
    if sampler not in ["multinomial", "nucleus", "greedy"]:
        raise ValueError(f"Unsupported {sampler}: sampler")

    hits_vocab = get_hits_vocab()
    ttoi = {v: k for k, v in hits_vocab.items()}
    start_ix = ttoi["start"]

    y = torch.tensor(
        [[start_ix]],
        dtype=torch.long,
        requires_grad=False,
        device=device,
    )

    for ix in range(n_tokens):
        # Get the predictions
        with torch.no_grad():
            logits = model(src=src, tgt=y)

        # Take the logits for the last tokens
        logits = logits[:, -1, :]

        # Apply softmax to get probabilities
        probs = temperatured_softmax(logits.cpu().numpy(), temperature)

        if sampler == "nucleus":
            y_next = []
            for j in range(probs.shape[0]):
                yn = nucleus(probs[j], p=nucleus_p)
                y_next.append(yn)
            y_next = torch.tensor(y_next, dtype=torch.long, device=device).unsqueeze(1)
        elif sampler == "multinomial":
            y_next = torch.multinomial(
                torch.tensor(probs, dtype=torch.float32, device=device),
                num_samples=1,
            )
        else:
            y_next = torch.tensor(
                [probs.argmax()], dtype=torch.long, device=device
            ).unsqueeze(1)

        y = torch.cat([y, y_next], dim=1)

    return y.squeeze(0)[1:]


def get_flat_nonzero_dissimilarity_matrix(x, y=None):
    if y is not None:
        x = np.concatenate((x, y))

    # Add a dummy axis
    if len(x.shape) == 1:
        x = x[..., np.newaxis]

    dists = distance_matrix(x, x, p=2)
    flat = dists.flatten()
    return np.delete(flat, np.arange(0, len(flat), len(x) + 1))


def compute_oa(A, B):
    # obtain density functions from data series
    pdf_A = stats.gaussian_kde(A)
    pdf_B = stats.gaussian_kde(B)

    lower_limit = np.min((np.min(A), np.min(B)))  # min of both
    upper_limit = np.max((np.max(A), np.max(B)))  # max of both

    # TODO: subsample pdfs to speed up integration
    area = integrate.quad(lambda x: min(pdf_A(x), pdf_B(x)), lower_limit, upper_limit)
    return area[0]


def compute_kld(A, B, num_sample=1000):
    pdf_A = stats.gaussian_kde(A)
    pdf_B = stats.gaussian_kde(B)

    # Both densities have to be evaluated at the same points for the result to be a divergence
    # between them. Sampling each over its own range and zipping the two vectors together
    # compares unrelated points.
    lower_limit = np.min((np.min(A), np.min(B)))
    upper_limit = np.max((np.max(A), np.max(B)))
    grid = np.linspace(lower_limit, upper_limit, num_sample)

    return stats.entropy(pdf_A(grid), pdf_B(grid))


def compute_oa_kld_dists(gen_df, ref_df, train_df=None, train_dist=None):
    """Pairwise distance distributions for each reference set, with and without the generations.

    Returns a `{label}_dist` / `{label}_gen_dist` pair per reference set, which is the shape
    `compute_oa_and_kld` and `make_oa_kld_plot` consume.
    """
    dists = {
        "ref_dist": get_flat_nonzero_dissimilarity_matrix(ref_df.values),
        "ref_gen_dist": get_flat_nonzero_dissimilarity_matrix(ref_df.values, gen_df.values),
    }

    if train_df is not None:
        # The training distribution is expensive on a large dataset, so accept a precomputed one
        dists["train_dist"] = (
            train_dist
            if train_dist is not None
            else get_flat_nonzero_dissimilarity_matrix(train_df.values)
        )
        dists["train_gen_dist"] = get_flat_nonzero_dissimilarity_matrix(
            train_df.values,
            gen_df.values,
        )

    return dists


def compute_oa_and_kld(dists):
    """Overlapping area and KL divergence for every pair in a `compute_oa_kld_dists` result"""
    suffix = "_gen_dist"

    oa_klds = {}
    for key in dists:
        if not key.endswith(suffix):
            continue
        label = key[: -len(suffix)]
        oa_klds[f"{label}_gen_oa"] = compute_oa(dists[f"{label}_dist"], dists[key])
        oa_klds[f"{label}_gen_kld"] = compute_kld(dists[f"{label}_dist"], dists[key])

    return oa_klds


def get_oa_kld_dists(gen_df, ref_df):
    results = {}

    # For all descriptors together
    ref_dist = get_flat_nonzero_dissimilarity_matrix(ref_df.values)
    ref_gen_dist = get_flat_nonzero_dissimilarity_matrix(
        ref_df.values,
        gen_df.values,
    )
    results["all_descriptors"] = {
        "ref_dist": ref_dist,
        "ref_gen_dist": ref_gen_dist,
    }

    # For each descriptor separately
    common_descriptors = set(gen_df.columns).intersection(set(ref_df.columns))
    for descriptor in common_descriptors:
        ref_dist = get_flat_nonzero_dissimilarity_matrix(ref_df[descriptor].values)

        ref_gen_dist = get_flat_nonzero_dissimilarity_matrix(
            ref_df[descriptor].values,
            gen_df[descriptor].values,
        )

        results[descriptor] = {
            "ref_dist": ref_dist,
            "ref_gen_dist": ref_gen_dist,
        }

    return results


def make_oa_kld_plot(
    dist_1,
    dist_2,
    oa,
    kld,
    outdir,
    model_name,
    suffix=None,
    label="train",
    descriptor="all descriptors",
):
    sns.kdeplot(dist_1, color="blue", label=f"{label}", bw_adjust=5, cut=0)
    sns.kdeplot(dist_2, color="orange", label=f"{label} + gen", bw_adjust=5, cut=0)
    plt.ylabel("")
    plt.legend()
    plt.title(f"{model_name}\n{descriptor}\nOA={round(oa, 3)}, KLD={round(kld, 3)}")
    plt.tight_layout()
    outname = os.path.join(outdir, f"oa-kde-dists-{label}")
    if suffix:
        outname += f"-{suffix}"
    outpath = f"{outname}.png"
    plt.savefig(outpath)
    plt.clf()
    print(f"Saved {outpath}")


def mk_descriptor_dist_plot(
    gen_df,
    ref_df,
    model_name,
    outdir,
    label="Train",
    checkpoint_num=None,
    id_col="Gen",
    title_suffix=None,
    filename_suffix=None,
):
    if len(ref_df) <= 2:
        print("WARNING: n_eval_seqs must be > 1 for oa and kld computation")
        return 0, 1

    id_col = "Gen"

    # Work on copies: this used to add a column to the caller's frames and drop it again at the
    # end, so anything raised in between left them modified
    ref_df = ref_df.copy()
    gen_df = gen_df.copy()
    ref_df[id_col] = label
    gen_df[id_col] = "Generated"
    feature_cols = [c for c in ref_df.columns if c != id_col]

    # Combine the generated with the ground truth
    compare_df = pd.concat([ref_df, gen_df])

    # Scale the feature columns to [0, 1]
    compare_df_scaled = (compare_df[feature_cols] - compare_df[feature_cols].min()) / (
        compare_df[feature_cols].max() - compare_df[feature_cols].min()
    )
    compare_df_scaled[id_col] = compare_df[id_col]

    # Plot a comparison of distributions for all descriptors
    sns.boxplot(
        x="variable",
        y="value",
        hue=id_col,
        data=pd.melt(compare_df_scaled, id_vars=id_col),
    )
    plt.ylabel("")
    plt.xlabel("")
    plt.yticks([])
    plt.xticks()
    title = f"{model_name}"
    if checkpoint_num:
        title += f" checkpoint {checkpoint_num}"
    if title_suffix:
        title += title_suffix
    plt.title(title)
    plt.legend(
        loc="upper left",
        # bbox_to_anchor=(0.92, 1.2),
        fancybox=True,
        shadow=False,
        ncol=1,
    )
    plt.tight_layout()
    filename = "dist-comparison"
    if filename_suffix:
        filename += f"-{filename_suffix}"
    outpath = os.path.join(outdir, f"{filename}.png")
    plt.savefig(outpath)
    plt.clf()
    print(f"Saved {outpath}")

    # Asymmetrical violin plot of the same data
    sns.violinplot(
        data=pd.melt(compare_df_scaled, id_vars=id_col),
        x="variable",
        y="value",
        hue=id_col,
        split=True,
    )
    plt.ylabel("")
    plt.xlabel("")
    plt.yticks([])
    plt.xticks()
    title = f"{model_name}"
    if checkpoint_num:
        title += f" checkpoint {checkpoint_num}"
    if title_suffix:
        title += title_suffix
    plt.title(title)
    plt.legend(
        loc="upper left",
        #         bbox_to_anchor=(0.92, 1.2),
        fancybox=True,
        shadow=False,
        ncol=1,
    )
    plt.tight_layout()
    filename = "dist-comparison-violin"
    if filename_suffix:
        filename += f"-{filename_suffix}"
    outpath = os.path.join(outdir, f"{filename}.png")
    plt.savefig(outpath)
    plt.clf()
    print(f"Saved {outpath}")
