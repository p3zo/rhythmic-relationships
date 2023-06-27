import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import integrate, stats
from scipy.spatial import distance_matrix


def get_flat_nonzero_dissimilarity_matrix(x, y=None):
    if y is not None:
        x = np.concatenate((x, y))
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

    sample_A = np.linspace(np.min(A), np.max(A), num_sample)
    sample_B = np.linspace(np.min(B), np.max(B), num_sample)

    return stats.entropy(pdf_A(sample_A), pdf_B(sample_B))


def compute_oa_kld_dists(gen_df, ref_df, train_df=None, train_dist=None):
    results = {}
    results["ref_dist"] = get_flat_nonzero_dissimilarity_matrix(ref_df.values)

    ref_gen_dist = get_flat_nonzero_dissimilarity_matrix(ref_df.values, gen_df.values)
    results["ref_gen_dist"] = ref_gen_dist

    if train_df is not None:
        if train_dist is None:
            train_dist = get_flat_nonzero_dissimilarity_matrix(train_df.values)

        train_gen_dist = get_flat_nonzero_dissimilarity_matrix(
            train_df.values,
            gen_df.values,
        )
        results["train_dist"] = train_dist
        results["train_gen_dist"] = train_gen_dist

    return results


def compute_oa_and_kld(oa_kld_dists):
    results = {}

    print("Computing OA and KLD for ref_gen")
    ref_dist = oa_kld_dists["ref_dist"]
    ref_gen_dist = oa_kld_dists["ref_gen_dist"]
    results["ref_gen_oa"] = compute_oa(ref_dist, ref_gen_dist)
    results["ref_gen_kld"] = compute_kld(ref_dist, ref_gen_dist)

    if "train_dist" in oa_kld_dists:
        print("Computing OA and KLD for train_gen")
        train_dist = oa_kld_dists["train_dist"]
        train_gen_dist = oa_kld_dists["train_gen_dist"]
        results["train_gen_oa"] = compute_oa(train_dist, train_gen_dist)
        results["train_gen_kld"] = compute_kld(train_dist, train_gen_dist)

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
):
    sns.kdeplot(dist_1, color="blue", label=label, bw_adjust=5, cut=0)
    sns.kdeplot(dist_2, color="orange", label=f"{label} + gen", bw_adjust=5, cut=0)
    plt.ylabel("")
    plt.legend()
    plt.title(f"{model_name}\nOA={round(oa, 3)}, KLD={round(kld, 3)}")
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
    id_col="Generated",
    title_suffix=None,
    filename_suffix=None,
):
    if len(ref_df) <= 2:
        print("WARNING: n_eval_seqs must be > 1 for oa and kld computation")
        return 0, 1

    id_col = "Generated"
    ref_df[id_col] = f"{label} (n={len(ref_df)})"
    gen_df[id_col] = f"Generated (n={len(gen_df)})"
    feature_cols = [c for c in ref_df.columns if c != id_col]

    # Combine the generated with the ground truth
    compare_df = pd.concat([gen_df, ref_df])

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
    title = f"{model_name}"
    if checkpoint_num:
        title += f" checkpoint {checkpoint_num}"
    if title_suffix:
        title += title_suffix
    plt.title(title)
    plt.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.08),
        fancybox=True,
        shadow=False,
        ncol=2,
    )
    plt.tight_layout()
    filename = "dist-comparison"
    if filename_suffix:
        filename += f"-{filename_suffix}"
    outpath = os.path.join(outdir, f"{filename}.png")
    plt.savefig(outpath)
    plt.clf()
    print(f"Saved {outpath}")

    ref_df.drop(id_col, axis=1, inplace=True)
    gen_df.drop(id_col, axis=1, inplace=True)
