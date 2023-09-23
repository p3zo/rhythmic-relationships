import glob
import os

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch
from matplotlib import rcParams

from rhythmic_relationships import DATASETS_DIR, PLOTS_DIRNAME, MODELS_DIR
from rhythmic_relationships.io import (
    load_midi_file,
    get_subdivisions,
)
from rhythmic_relationships.evaluate import (
    get_oa_kld_dists,
    compute_oa,
    compute_kld,
    mk_descriptor_dist_plot,
)
from rhythmic_relationships.representations import get_representations
from rhythmic_relationships.model_utils import load_model
from rhythmic_relationships.models.hits_encdec import TransformerEncoderDecoder
from rhythmtoolbox import pianoroll2descriptors, DESCRIPTOR_NAMES

from pair_descriptors import get_onset_balance, get_antiphony

sns.set_style("white")
sns.set_context("paper")

# fig size in inches
rcParams["figure.figsize"] = [6, 4]  # fig size in inches
rcParams["figure.dpi"] = 300
# rcParams['font.size'] = 22

# Load existing samples from a model and compute their paired descriptors
model_type = "hits_encdec"

# Melody -> Bass
model_name = "fragmental_2306210056"

# Bass -> Melody
# model_name = "literation_2307011858"
# model_name = "dematerialize_2307012124"


model_dir = os.path.join(MODELS_DIR, model_type, model_name)
model_eval_dir = os.path.join(model_dir, "eval")
samples_dir = os.path.join(model_eval_dir, "inference")

model_path = os.path.join(model_dir, "model.pt")

model, config = load_model(model_path, TransformerEncoderDecoder)

# Size of the reference subsets to use when computing OA+KLD. Speeds up computation.
ref_subset = 100

# Create the output directory
dataset_plots_dir = os.path.join(
    DATASETS_DIR, config["data"]["dataset_name"], PLOTS_DIRNAME
)
if not os.path.isdir(dataset_plots_dir):
    os.makedirs(dataset_plots_dir)


ref_bass_df = pd.read_csv(os.path.join(dataset_plots_dir, "bass_descriptors.csv"))
ref_mel_df = pd.read_csv(os.path.join(dataset_plots_dir, "melody_descriptors.csv"))
ref_paired_df = pd.read_csv(
    os.path.join(dataset_plots_dir, "paired_melody_bass_descriptors.csv")
)

sample_filepaths = glob.glob(os.path.join(samples_dir, "*.mid"))
for sampler in ["greedy", "multinomial", "nucleus"]:
    print(f"{sampler}")

    src_sample_filepaths = [i for i in sample_filepaths if f"{sampler}_src" in i]
    gen_sample_filepaths = [i for i in sample_filepaths if f"{sampler}_gen" in i]

    # Load samples as midi, compute descriptors
    src_hits_list = []
    gen_hits_list = []

    src_descs_list = []
    gen_descs_list = []

    n_samples = len(src_sample_filepaths)
    for i in range(n_samples):
        src_fp = [
            p for p in src_sample_filepaths if os.path.basename(p).startswith(f"{i}_")
        ]
        if not src_fp:
            continue
        src_fp = src_fp[0]
        src_pmid = load_midi_file(src_fp)
        src_subdivisions = get_subdivisions(src_pmid, resolution=4)
        src_tracks = get_representations(src_pmid, src_subdivisions)
        src_hits = src_tracks[0]["hits"].tolist()
        if len(src_hits) < 32:
            src_hits.extend([0.0] * (32 - len(src_hits)))
        src_onset_roll = src_tracks[0]["onset_roll"]
        if len(src_onset_roll) < 32:
            pad_right = np.zeros((32 - len(src_onset_roll), 128))
            src_onset_roll = np.vstack((src_onset_roll, pad_right))
        src_descs = np.array(
            list(
                pianoroll2descriptors(
                    src_onset_roll,
                    resolution=4,
                    drums=src_tracks[0]["name"] == "Drums",
                ).values()
            ),
            dtype=np.float32,
        )
        src_hits_list.append(src_hits)
        src_descs_list.append(src_descs)

        gen_fp = [
            p for p in gen_sample_filepaths if os.path.basename(p).startswith(f"{i}_")
        ][0]
        gen_pmid = load_midi_file(gen_fp)
        gen_subdivisions = get_subdivisions(gen_pmid, resolution=4)
        gen_tracks = get_representations(gen_pmid, gen_subdivisions)
        gen_hits = gen_tracks[0]["hits"].tolist()
        if len(gen_hits) < 32:
            gen_hits.extend([0.0] * (32 - len(gen_hits)))
        gen_onset_roll = gen_tracks[0]["onset_roll"]
        if len(gen_onset_roll) < 32:
            pad_right = np.zeros((32 - len(gen_onset_roll), 128))
            gen_onset_roll = np.vstack((gen_onset_roll, pad_right))
        gen_descs = np.array(
            list(
                pianoroll2descriptors(
                    gen_onset_roll,
                    resolution=4,
                    drums=gen_tracks[0]["name"] == "Drums",
                ).values()
            ),
            dtype=np.float32,
        )
        gen_descs_list.append(gen_descs)
        gen_hits_list.append(gen_hits)

    # Individual descriptor distribution comparisons
    drop_cols = ["noi", "polyDensity", "syness"]
    src_desc_df = pd.DataFrame(np.vstack(src_descs_list), columns=DESCRIPTOR_NAMES)
    src_desc_df.dropna(how="all", axis=0, inplace=True)
    src_desc_df.dropna(how="all", axis=1, inplace=True)
    src_desc_df.drop(drop_cols, axis=1, inplace=True)
    src_desc_df.to_csv(
        os.path.join(model_eval_dir, f"{sampler}_src_descs.csv"), index=False
    )

    gen_df_indiv = pd.DataFrame(np.vstack(gen_descs_list), columns=DESCRIPTOR_NAMES)
    gen_df_indiv.dropna(how="all", axis=0, inplace=True)
    gen_df_indiv.dropna(how="all", axis=1, inplace=True)
    gen_df_indiv.drop(drop_cols, axis=1, inplace=True)
    gen_df_indiv.to_csv(
        os.path.join(model_eval_dir, f"{sampler}_gen_descs.csv"), index=False
    )

    gen_df_indiv.describe().to_csv(
        os.path.join(model_eval_dir, f"{sampler}_gen_descs_summary.csv")
    )

    ref_df = ref_bass_df if config["data"]["part_1"] == "Melody" else ref_mel_df

    # cols = ['Step Density', 'Syncopation', 'Balance', 'Evenness']
    # ref_df.columns = cols
    # gen_df_indiv.columns = cols

    model_abbr = f'{config["data"]["part_1"]} to {config["data"]["part_2"]}'
    mk_descriptor_dist_plot(
        gen_df=gen_df_indiv,
        ref_df=ref_df,
        model_name=f'{model_abbr} (sampling={sampler})',
        outdir=model_eval_dir,
        title_suffix="",
        filename_suffix=f"{sampler}_bass_train_vs_gen",
    )

    # Compute OA+KLD for individual descs
    # Subset reference dfs to speed up computation
    ref_df = ref_df.sample(ref_subset)

    indiv_oa_klds = []

    print(f"Computing oa and kld using a reference subset of {ref_subset} obs")
    indv_oa_kld_dists = get_oa_kld_dists(gen_df=gen_df_indiv, ref_df=ref_df)
    for descriptor in indv_oa_kld_dists:
        print(f"{descriptor=}")
        dist_1 = indv_oa_kld_dists[descriptor]["ref_dist"]
        dist_2 = indv_oa_kld_dists[descriptor]["ref_gen_dist"]

        np.savetxt(f"ref_{sampler}_{descriptor}.csv", dist_1, delimiter=",")
        np.savetxt(f"ref_gen_{sampler}_{descriptor}.csv", dist_2, delimiter=",")

        oa = compute_oa(dist_1, dist_2)
        kld = compute_kld(dist_1, dist_2)
        print(f"{oa=} and {kld=}")

        sns.kdeplot(dist_1, color="blue", label="Train", bw_adjust=5, cut=0)
        sns.kdeplot(dist_2, color="orange", label="Train + gen", bw_adjust=5, cut=0)
        plt.ylabel("")
        plt.legend()
        plt.title(f"{sampler}\n{descriptor}\nOA={round(oa, 3)}, KLD={round(kld, 3)}")
        plt.tight_layout()
        outpath = os.path.join(
            model_eval_dir, f"{sampler}_oa-kde-dists-train_{descriptor}.png"
        )
        plt.savefig(outpath)
        plt.clf()
        print(f"Saved {outpath}")
        indiv_oa_klds.append([descriptor, round(oa, 3), round(kld, 3)])

    indiv_oa_kld_df = pd.DataFrame(indiv_oa_klds, columns=["descriptor", "oa", "kld"])
    indiv_oa_kld_df.to_csv(
        os.path.join(model_eval_dir, f"{sampler}_indiv_oa_klds.csv"), index=False
    )

    # Paired descriptor distribution comparisons
    src_hits_tensor = (torch.tensor(src_hits_list) > 0).to(int)
    gen_hits_tensor = (torch.tensor(gen_hits_list) > 0).to(int)

    sg_onset_balance = get_onset_balance(src_hits_tensor, gen_hits_tensor)
    sg_antiphony = get_antiphony(src_hits_tensor, gen_hits_tensor)

    columns = ["onset_balance", "antiphony"]
    gen_df_paired = pd.DataFrame(
        torch.stack([sg_onset_balance, sg_antiphony], axis=1),
        columns=columns,
    )
    gen_df_paired.to_csv(
        os.path.join(
            model_eval_dir, f"src_gen_paired_desc_dists_{n_samples}_{sampler}.csv"
        ),
        index=False,
    )

    gen_df_paired.describe().to_csv(
        os.path.join(model_eval_dir, f"{sampler}_gen_descs_paired_summary.csv")
    )

    # First combine the generated with the ground truth
    id_col = "Gen"
    ref_paired_df[id_col] = "Train"
    gen_df_paired[id_col] = "Gen"
    feature_cols = [c for c in ref_paired_df.columns if c != id_col]
    compare_df = pd.concat([ref_paired_df, gen_df_paired])
    compare_df = compare_df[compare_df.antiphony < 1]

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
    plt.title(f'{config["data"]["part_1"]} to {config["data"]["part_2"]}')
    plt.legend(
        loc="upper left",
        # bbox_to_anchor=(0.92, 1.2),
        fancybox=True,
        shadow=False,
        ncol=1,
    )
    plt.tight_layout()
    outpath = os.path.join(model_eval_dir, f"pair-dist-comparison-violin_{sampler}.png")
    plt.savefig(outpath)
    plt.clf()
    print(f"Saved {outpath}")

    ref_paired_df.drop(id_col, axis=1, inplace=True)
    gen_df_paired.drop(id_col, axis=1, inplace=True)

    # Compute OA+KLD
    # Use a subset of the reference data to speed up computation
    ref_paired_df_small = ref_paired_df.sample(ref_subset)

    paired_oa_klds = []

    print(f"Computing oa and kld using a reference subset of {ref_subset} obs")
    oa_kld_dists = get_oa_kld_dists(gen_df=gen_df_paired, ref_df=ref_paired_df_small)
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
        plt.title(f"{sampler}\n{descriptor}\nOA={round(oa, 3)}, KLD={round(kld, 3)}")
        plt.tight_layout()
        outpath = os.path.join(
            model_eval_dir, f"{sampler}_paired-oa-kde-dists-train_{descriptor}.png"
        )
        plt.savefig(outpath)
        plt.clf()
        print(f"Saved {outpath}")

        paired_oa_klds.append([descriptor, round(oa, 3), round(kld, 3)])

    paired_oa_kld_df = pd.DataFrame(paired_oa_klds, columns=["descriptor", "oa", "kld"])
    paired_oa_kld_df.to_csv(
        os.path.join(model_eval_dir, f"{sampler}_paired_oa_klds.csv"), index=False
    )
