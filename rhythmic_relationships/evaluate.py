import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import integrate, stats
from scipy.spatial import distance_matrix


def get_flat_nonzero_dissimilarity_matrix(x):
    dists = distance_matrix(x, x, p=2)
    flat = dists.flatten()
    return np.delete(flat, np.arange(0, len(flat), len(x) + 1))


def compute_oa(A, B):
    print("Computing OA")

    # obtain density functions from data series
    pdf_A = stats.gaussian_kde(A)
    pdf_B = stats.gaussian_kde(B)

    lower_limit = np.min((np.min(A), np.min(B)))  # min of both
    upper_limit = np.max((np.max(A), np.max(B)))  # max of both

    # TODO: subsample pdfs to speed up integration
    area = integrate.quad(lambda x: min(pdf_A(x), pdf_B(x)), lower_limit, upper_limit)
    return area[0]


def compute_kld(A, B, num_sample=1000):
    print("Computing KLD")

    pdf_A = stats.gaussian_kde(A)
    pdf_B = stats.gaussian_kde(B)

    sample_A = np.linspace(np.min(A), np.max(A), num_sample)
    sample_B = np.linspace(np.min(B), np.max(B), num_sample)

    return stats.entropy(pdf_A(sample_A), pdf_B(sample_B))


def compute_oa_and_kld(train_dist, train_gen_dist):
    oa = compute_oa(train_dist, train_gen_dist)
    kld = compute_kld(train_dist, train_gen_dist)
    return oa, kld


def make_oa_kld_plot(
    train_dist, train_gen_dist, oa, kld, outdir, model_name, suffix=None
):
    # Plot train_dist vs train_gen_dist
    sns.kdeplot(train_dist, color="blue", label="dataset", bw_adjust=5, cut=0)
    sns.kdeplot(
        train_gen_dist, color="orange", label="dataset + gen", bw_adjust=5, cut=0
    )
    plt.ylabel("")
    plt.legend()
    plt.title(f"{model_name}\nOA={round(oa, 3)}, KLD={round(kld, 3)}")
    plt.tight_layout()
    outname = os.path.join(outdir, "oa-kde-dists")
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
    checkpoint_num=None,
    id_col="Generated",
    title_suffix=None,
    filename_suffix=None,
):
    if len(ref_df) <= 2:
        print("WARNING: n_eval_seqs must be > 1 for oa and kld computation")
        return 0, 1

    id_col = "Generated"
    ref_df[id_col] = f"Dataset (n={len(ref_df)})"
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
