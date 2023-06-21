"""Compare training set descriptors to generated after each epoch"""
import os

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch
from tqdm import tqdm

from rhythmic_relationships import MODELS_DIR, CHECKPOINTS_DIRNAME
from rhythmic_relationships.data import PartDataset, get_hits_from_hits_seq
from rhythmic_relationships.evaluate import (
    compute_oa_and_kld,
    get_flat_nonzero_dissimilarity_matrix,
)
from rhythmic_relationships.io import write_midi_from_hits, get_roll_from_hits
from rhythmic_relationships.model_utils import load_model
from rhythmic_relationships.models.hits_decoder import TransformerDecoder
from rhythmtoolbox import pianoroll2descriptors

from train import inference

DEVICE = torch.device(
    "mps"
    if torch.backends.mps.is_built()
    else torch.device("cuda:0")
    if torch.cuda.device_count() > 0
    else torch.device("cpu")
)


if __name__ == "__main__":
    model_name = "multiovulate_2306112040"

    checkpoint_num = 13
    n_training_obs = 1000
    pitch = 72
    resolution = 4
    n_seqs = 1000
    temperature = 1
    nucleus_p = 0.92
    samplers = ["multinomial", "nucleus"]

    plots_dir = os.path.join(MODELS_DIR, model_name, "eval_plots")
    if not os.path.isdir(plots_dir):
        os.makedirs(plots_dir)

    gen_dir = os.path.join(MODELS_DIR, model_name, "inference")
    if not os.path.isdir(gen_dir):
        os.makedirs(gen_dir)

    device = DEVICE

    # Load model
    model_dir = os.path.join(MODELS_DIR, model_name)
    if checkpoint_num:
        checkpoints_dir = os.path.join(model_dir, CHECKPOINTS_DIRNAME)
        model_path = os.path.join(checkpoints_dir, str(checkpoint_num))
    else:
        model_path = os.path.join(model_dir, "model.pt")

    model, config = load_model(model_path, TransformerDecoder)
    model = model.to(DEVICE)
    part_1 = config["data"]["part_1"]
    part_2 = config["data"]["part_2"]

    drop_features = ["noi", "polyDensity", "syness"]

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

    # Use the model to generate new sequences
    generated_rolls = []
    generated_descs = []

    all_zeros = 0
    all_same = 0

    for sampler in samplers:
        print(f"{sampler=}")
        for ix in tqdm(range(n_seqs)):
            seq = inference(
                model=model,
                n_tokens=32,
                temperature=temperature,
                device=device,
                sampler=sampler,
                nucleus_p=nucleus_p,
            )

            gen_hits = get_hits_from_hits_seq(
                seq.cpu().numpy(),
                part=part_2,
                verbose=False,
            )

            if max(gen_hits) == 0:
                all_zeros += 1
                continue
            if len(set(gen_hits)) == 1:
                all_same += 1
                continue

            write_midi_from_hits(
                [i * 127 for i in gen_hits],
                outpath=os.path.join(gen_dir, f"{ix}.mid"),
                part=part_2,
                pitch=pitch,
            )

            gen_roll = get_roll_from_hits(
                [i * 127 for i in gen_hits], pitch=pitch, resolution=resolution
            )
            generated_rolls.append(gen_roll)

            gen_descs = pianoroll2descriptors(
                gen_roll,
                resolution,
                drums=part_2 == "Drums",
            )
            generated_descs.append(gen_descs)

        print(f"{n_seqs=}")
        print(f"  {all_zeros=} ({100*round(all_zeros/n_seqs, 2)}%)")
        print(f"  {all_same=} ({100*round(all_same/n_seqs, 2)}%)")

        gen_df = pd.DataFrame(generated_descs).dropna(how="all", axis=1)
        gen_df.drop(drop_features, axis=1, inplace=True)

        # Combine the generated with the ground truth
        id_col = "Generated"
        gen_df[id_col] = f"Generated (n={len(gen_df)})"
        dataset_df[id_col] = f"Dataset (n={len(dataset_df)})"
        compare_df = pd.concat([gen_df, dataset_df])

        # Scale the feature columns to [0, 1]
        feature_cols = [c for c in dataset_df.columns if c != id_col]
        compare_df_scaled = (
            compare_df[feature_cols] - compare_df[feature_cols].min()
        ) / (compare_df[feature_cols].max() - compare_df[feature_cols].min())
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
        title += f"\n{temperature=}"
        if sampler == "nucleus":
            title += f" {nucleus_p=}"
        plt.title(title)
        plt.legend(
            loc="upper center",
            bbox_to_anchor=(0.5, -0.08),
            fancybox=True,
            shadow=False,
            ncol=2,
        )
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, f"dist-comparison-{sampler}.png"))
        plt.clf()

        # Descriptors from training dataset
        train = dataset_df[feature_cols].values

        # Descriptors from generated dataset
        gen = gen_df[feature_cols].values

        train_dist = get_flat_nonzero_dissimilarity_matrix(train)

        # Stack training and generation
        train_gen = np.concatenate((train, gen))
        train_gen_dist = get_flat_nonzero_dissimilarity_matrix(train_gen)

        # Compute distribution comparison metrics
        oa, kld = compute_oa_and_kld(train_dist, train_gen_dist)

        print(f"  {oa=}, {kld=}")
        print()
