"""Compare training set descriptors to generated after each epoch"""

"""
coda_2306180418

"""
import itertools
import os

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch

from rhythmic_relationships import MODELS_DIR, CHECKPOINTS_DIRNAME
from rhythmic_relationships.data import PartDataset
from rhythmic_relationships.evaluate import (
    compute_oa_and_kld,
    get_flat_nonzero_dissimilarity_matrix,
)
from rhythmic_relationships.io import write_midi_from_hits, get_roll_from_hits
from rhythmic_relationships.model_utils import load_model
from rhythmtoolbox import pianoroll2descriptors

from train import RSP_FC

DEVICE = torch.device(
    "mps"
    if torch.backends.mps.is_built()
    else torch.device("cuda:0")
    if torch.cuda.device_count() > 0
    else torch.device("cpu")
)


if __name__ == "__main__":
    model_type = "rsp_fc"
    model_name = "undeterminate_2306192352"

    checkpoint_num = None
    n_training_obs = 1000
    pitch = 72
    resolution = 4

    model_dir = os.path.join(MODELS_DIR, model_type, model_name)

    eval_dir = os.path.join(model_dir, "eval")
    if not os.path.isdir(eval_dir):
        os.makedirs(eval_dir)

    gen_dir = os.path.join(model_dir, "inference")
    if not os.path.isdir(gen_dir):
        os.makedirs(gen_dir)

    device = DEVICE

    if checkpoint_num:
        checkpoints_dir = os.path.join(model_dir, CHECKPOINTS_DIRNAME)
        model_path = os.path.join(checkpoints_dir, str(checkpoint_num))
    else:
        model_path = os.path.join(model_dir, "model.pt")

    model, config = load_model(model_path, RSP_FC)
    model = model.to(DEVICE)
    part_1 = config["data"]["part_1"]
    part_2 = config["data"]["part_2"]

    drop_features = ["noi", "polyDensity"]

    # Get distribution from training set
    full_df = PartDataset(
        dataset_name=config["data"]["dataset_name"],
        part=part_2,
        representation="descriptors",
    ).as_df(subset=n_training_obs)
    dataset_df = full_df.drop(
        ["filename", "segment_id"] + drop_features,
        axis=1,
    ).dropna(how="all", axis=1)

    # Descriptors from training dataset
    id_col = "Generated"
    dataset_df[id_col] = f"Dataset (n={len(dataset_df)})"
    feature_cols = [c for c in dataset_df.columns if c != id_col]
    train = dataset_df[feature_cols].values

    tenths = [i / 10 for i in range(11)]
    xys = list(itertools.product(tenths, repeat=2))
    n_seqs = len(xys)

    xy_gen_hits = {}
    for x, y in xys:
        src = torch.tensor([x, y], dtype=torch.float32, device=device)
        with torch.no_grad():
            gen_hits = model(src)
            xy_gen_hits[f"({x},{y})"] = gen_hits

    # Threshold and make velocity bins
    zero_threshes = [0.1, 0.2, 0.3, 0.4, 0.5]
    thresh_list = []
    for zt in zero_threshes:
        tl = [zt + i * ((1 - zt) / 4) for i in range(4)]
        thresh_list.append(tl)

    for threshes in thresh_list:
        print(threshes[0])
        thresh_dir = os.path.join(gen_dir, str(threshes[0]))
        if not os.path.isdir(thresh_dir):
            os.makedirs(thresh_dir)

        threshed_rolls = []
        threshed_descs = []

        all_zeros = 0
        all_same = 0

        for xy_str, seq in xy_gen_hits.items():
            x, y = [float(i) for i in xy_str[1:-1].split(",")]
            sampled_hits = []
            for i in seq:
                if i <= threshes[0]:
                    sampled_hits.append(0)
                elif i <= threshes[1]:
                    sampled_hits.append(0.25)
                elif i <= threshes[2]:
                    sampled_hits.append(0.5)
                elif i <= threshes[3]:
                    sampled_hits.append(0.75)
                else:
                    sampled_hits.append(1)
            sampled_hits = torch.tensor(sampled_hits, dtype=torch.float32)

            if max(sampled_hits) == 0:
                all_zeros += 1
                continue
            if len(set(sampled_hits)) == 1:
                all_same += 1
                continue

            write_midi_from_hits(
                [i * 127 for i in sampled_hits],
                outpath=os.path.join(thresh_dir, f"({x},{y}).mid"),
                part=part_2,
                pitch=pitch,
                name=f"({x},{y})",
            )

            threshed_roll = get_roll_from_hits(
                [i * 127 for i in sampled_hits], pitch=pitch, resolution=resolution
            )
            threshed_rolls.append(threshed_roll)

            t_descs = pianoroll2descriptors(
                threshed_roll,
                resolution,
                drums=part_2 == "Drums",
            )
            threshed_descs.append(t_descs)

        print(f"{n_seqs=}")
        print(f"  {all_zeros=} ({100*round(all_zeros/n_seqs, 2)}%)")
        print(f"  {all_same=} ({100*round(all_same/n_seqs, 2)}%)")

        gen_df = pd.DataFrame(threshed_descs).dropna(how="all", axis=1)
        gen_df.drop(drop_features, axis=1, inplace=True)

        # Combine the generated with the ground truth
        gen_df[id_col] = f"Generated (n={len(gen_df)})"
        compare_df = pd.concat([gen_df, dataset_df])

        # Scale the feature columns to [0, 1]
        compare_df_scaled = (
            compare_df[feature_cols] - compare_df[feature_cols].min()
        ) / (compare_df[feature_cols].max() - compare_df[feature_cols].min())
        compare_df_scaled[id_col] = compare_df[id_col]

        # Descriptors from generated dataset
        gen = gen_df[feature_cols].values

        train_dist = get_flat_nonzero_dissimilarity_matrix(train)

        # Stack training and generation
        train_gen = np.concatenate((train, gen))
        train_gen_dist = get_flat_nonzero_dissimilarity_matrix(train_gen)

        # Compute distribution comparison metrics
        oa, kld = compute_oa_and_kld(train_dist, train_gen_dist)

        print(f"  oa={round(oa, 3)}, kld={round(kld, 3)}")

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
        title += f" zero_thresh={threshes[0]}\noa={round(oa, 3)}, kld={round(kld, 3)}"
        plt.title(title)
        plt.legend(
            loc="upper center",
            bbox_to_anchor=(0.5, -0.08),
            fancybox=True,
            shadow=False,
            ncol=2,
        )
        plt.tight_layout()
        plt.savefig(os.path.join(eval_dir, f"dist-comparison-{threshes[0]}.png"))
        plt.clf()

        # TODO: save stats somewhere. Back into model obj?
