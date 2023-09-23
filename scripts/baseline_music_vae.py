import glob
import os

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch
from matplotlib import rcParams

from rhythmtoolbox import pianoroll2descriptors, DESCRIPTOR_NAMES
from rhythmic_relationships import DATASETS_DIR, PLOTS_DIRNAME
from rhythmic_relationships.evaluate import (
    get_oa_kld_dists,
    compute_oa,
    compute_kld,
    mk_descriptor_dist_plot,
)
from rhythmic_relationships.io import load_midi_file, get_subdivisions
from rhythmic_relationships.representations import get_representations

from pair_descriptors import get_onset_balance, get_antiphony

sns.set_style("white")
sns.set_context("paper")

# fig size in inches
rcParams["figure.figsize"] = [6, 4]  # fig size in inches
rcParams['figure.dpi'] = 300
# rcParams['font.size'] = 22

# Size of the reference subsets to use when computing OA+KLD. Speeds up computation.
ref_subset = 100

# Load output samples from music vae
baseline_dir = os.path.join(DATASETS_DIR, "baseline")
samples_dir = os.path.join(baseline_dir, "music-vae-4bar-trios-samples")
sample_filepaths = glob.glob(os.path.join(samples_dir, "*.mid"))

# Slice into 2 bars and extract melody/bass
bass_hits = []
melody_hits = []
bass_descs = []
melody_descs = []

for fp in sample_filepaths:
    pmid = load_midi_file(fp)
    subdivisions = get_subdivisions(pmid, resolution=4)
    tracks = get_representations(pmid, subdivisions)
    for track in tracks:
        hits_4bar = track["hits"].tolist()
        hits_2bar = [hits_4bar[:32], hits_4bar[32:]]

        onset_roll_2bar_1 = track["onset_roll"][:32]
        onset_roll_2bar_2 = track["onset_roll"][32:]
        descs_1 = np.array(
            list(
                pianoroll2descriptors(
                    onset_roll_2bar_1,
                    resolution=4,
                    drums=track["name"] == "Drums",
                ).values()
            ),
            dtype=np.float32,
        )
        descs_2 = np.array(
            list(
                pianoroll2descriptors(
                    onset_roll_2bar_2,
                    resolution=4,
                    drums=track["name"] == "Drums",
                ).values()
            ),
            dtype=np.float32,
        )

        if track["name"] == "Bass":
            bass_hits.extend(hits_2bar)
            bass_descs.extend([descs_1, descs_2])
        elif track["name"] == "Melody":
            melody_hits.extend(hits_2bar)
            melody_descs.extend([descs_1, descs_2])

drop_cols = ["noi", "polyDensity", "syness"]
bass_desc_df = pd.DataFrame(np.vstack(bass_descs), columns=DESCRIPTOR_NAMES)
bass_desc_df.dropna(how="all", axis=0, inplace=True)
bass_desc_df.dropna(how="all", axis=1, inplace=True)
bass_desc_df.drop(drop_cols, axis=1, inplace=True)
bass_desc_df.to_csv(
    os.path.join(baseline_dir, "baseline_mvae_bass_descs.csv"), index=False
)
bass_desc_df.describe().to_csv(
    os.path.join(baseline_dir, "baseline_mvae_bass_descs_summary.csv")
)


melody_desc_df = pd.DataFrame(np.vstack(melody_descs), columns=DESCRIPTOR_NAMES)
melody_desc_df.dropna(how="all", axis=0, inplace=True)
melody_desc_df.dropna(how="all", axis=1, inplace=True)
melody_desc_df.drop(drop_cols, axis=1, inplace=True)
melody_desc_df.to_csv(
    os.path.join(baseline_dir, "baseline_mvae_melody_descs.csv"), index=False
)
melody_desc_df.describe().to_csv(
    os.path.join(baseline_dir, "baseline_mvae_melody_descs_summary.csv")
)


dataset_plots_path = os.path.join(DATASETS_DIR, "lmdc_3000_2bar_4res", PLOTS_DIRNAME)
ref_bass_df = pd.read_csv(os.path.join(dataset_plots_path, "bass_descriptors.csv"))
ref_mel_df = pd.read_csv(os.path.join(dataset_plots_path, "melody_descriptors.csv"))

cols = ['Step Density', 'Syncopation', 'Balance', 'Evenness']
ref_bass_df.columns = cols
ref_mel_df.columns = cols
bass_desc_df.columns = cols
melody_desc_df.columns = cols

mk_descriptor_dist_plot(
    gen_df=bass_desc_df,
    ref_df=ref_bass_df,
    model_name="MusicVAE",
    outdir=baseline_dir,
    title_suffix="",
    filename_suffix="baseline_bass_train_vs_gen",
)
mk_descriptor_dist_plot(
    gen_df=melody_desc_df,
    ref_df=ref_mel_df,
    model_name="MusicVAE",
    outdir=baseline_dir,
    title_suffix="",
    filename_suffix="baseline_melody_train_vs_gen",
)

# Compute OA KLD for individual descs
# Subset reference dfs to speed up computation
ref_bass_df_small = ref_bass_df.sample(ref_subset)
ref_mel_df_small = ref_mel_df.sample(ref_subset)

indiv_oa_klds = []

print(f"Computing oa and kld using a reference subset of {ref_subset} obs")
indv_oa_kld_dists = get_oa_kld_dists(gen_df=bass_desc_df, ref_df=ref_bass_df_small)
for descriptor in indv_oa_kld_dists:
    print(f"{descriptor=}")
    dist_1 = indv_oa_kld_dists[descriptor]["ref_dist"]
    dist_2 = indv_oa_kld_dists[descriptor]["ref_gen_dist"]

    oa = compute_oa(dist_1, dist_2)
    kld = compute_kld(dist_1, dist_2)
    print(f"{oa=} and {kld=}")

    sns.kdeplot(dist_1, color="blue", label="Train", bw_adjust=5, cut=0)
    sns.kdeplot(dist_2, color="orange", label="Train + gen", bw_adjust=5, cut=0)
    plt.ylabel("")
    plt.legend()
    plt.title(f"baseline\n{descriptor}\nOA={round(oa, 3)}, KLD={round(kld, 3)}")
    plt.tight_layout()
    outpath = os.path.join(baseline_dir, f"oa-kde-dists-train_{descriptor}.png")
    plt.savefig(outpath)
    plt.clf()
    print(f"Saved {outpath}")
    indiv_oa_klds.append([descriptor, round(oa, 3), round(kld, 3)])

indiv_oa_kld_df = pd.DataFrame(indiv_oa_klds)
indiv_oa_kld_df.to_csv(
    os.path.join(baseline_dir, "baseline_indiv_oa_klds.csv"), index=False, header=False
)

# Compute paired descriptors
bass_hits_tensor = (torch.tensor(bass_hits) > 0).to(int)
melody_hits_tensor = (torch.tensor(melody_hits) > 0).to(int)

onset_balance = get_onset_balance(bass_hits_tensor, melody_hits_tensor)
antiphony = get_antiphony(bass_hits_tensor, melody_hits_tensor)

columns = ["onset_balance", "antiphony"]
gen_df = pd.DataFrame(
    torch.stack([onset_balance, antiphony], axis=1),
    columns=columns,
)
gen_df.dropna(how="any", inplace=True)
gen_df.to_csv(
    os.path.join(baseline_dir, f"baseline_mvae_paired_descs.csv"), index=False
)
gen_df.describe().to_csv(
    os.path.join(baseline_dir, "baseline_gen_descs_paired_summary.csv")
)

# Plot descriptor distributions against LMD
dataset_plots_dir = os.path.join(DATASETS_DIR, "lmdc_3000_2bar_4res", "plots")
# NOTE: Just load one of these two because they represent the same data, even if they're not exactly equal due to random sampling
# mb_df = pd.read_csv(os.path.join(dataset_plots_dir, 'Melody_Bass', 'paired_desc_dists_10000.csv'))
ref_paired_df = pd.read_csv(
    os.path.join(dataset_plots_dir, "paired_melody_bass_descriptors.csv")
)

# First Melody-Bass
id_col = "Gen"
ref_paired_df[id_col] = "Train"
gen_df[id_col] = "Gen"
feature_cols = [c for c in ref_paired_df.columns if c != id_col]
compare_df = pd.concat([ref_paired_df, gen_df])

# Asymmetrical violin plot
sns.violinplot(
    data=pd.melt(compare_df, id_vars=id_col),
    x="variable",
    y="value",
    hue=id_col,
    split=True,
)
plt.ylabel("")
plt.xlabel("")
plt.yticks([])
# title = f"{model_name}"
# plt.title(title)
plt.legend(
    loc="upper left",
    # bbox_to_anchor=(0.92, 1.2),
    fancybox=True,
    shadow=False,
    ncol=1,
)
plt.tight_layout()
outpath = os.path.join(baseline_dir, f"pair-dist-comparison-violin.png")
plt.savefig(outpath)
plt.clf()
print(f"Saved {outpath}")

ref_paired_df.drop(id_col, axis=1, inplace=True)
gen_df.drop(id_col, axis=1, inplace=True)

# Compute OA+KLD for paired dscs
# Subset reference df to speed up computation
ref_paired_df_small = ref_paired_df.sample(ref_subset)

paired_oa_klds = []

print(f"Computing oa and kld using a reference subset of {ref_subset} obs")
oa_kld_dists = get_oa_kld_dists(gen_df=gen_df, ref_df=ref_paired_df_small)
for descriptor in oa_kld_dists:
    print(f"{descriptor=}")
    dist_1 = oa_kld_dists[descriptor]["ref_dist"]
    dist_2 = oa_kld_dists[descriptor]["ref_gen_dist"]

    oa = compute_oa(dist_1, dist_2)
    kld = compute_kld(dist_1, dist_2)
    print(f"{oa=} and {kld=}")

    sns.kdeplot(dist_1, color="blue", label="Train", bw_adjust=5, cut=0)
    sns.kdeplot(dist_2, color="orange", label="Train + gen", bw_adjust=5, cut=0)
    plt.ylabel("")
    plt.legend()
    plt.title(f"baseline\n{descriptor}\nOA={round(oa, 3)}, KLD={round(kld, 3)}")
    plt.tight_layout()
    outpath = os.path.join(baseline_dir, f"paired-oa-kde-dists-train_{descriptor}.png")
    plt.savefig(outpath)
    plt.clf()
    print(f"Saved {outpath}")

    paired_oa_klds.append([descriptor, round(oa, 3), round(kld, 3)])

paired_oa_kld_df = pd.DataFrame(paired_oa_klds)
paired_oa_kld_df.to_csv(
    os.path.join(baseline_dir, "baseline_paired_oa_klds.csv"), index=False, header=False
)
