import os

import matplotlib.pyplot as plt
import pandas as pd
import torch
import seaborn as sns
from tqdm import tqdm
from rhythmtoolbox import pianoroll2descriptors

from model_utils import load_model
from rhythmic_relationships import MODELS_DIR
from rhythmic_relationships.data import PartDataset, PartPairDataset
from rhythmic_relationships.io import get_roll_from_sequence, write_midi_from_roll
from rhythmic_relationships.model import TransformerEncoderDecoderNew
from rhythmic_relationships.vocab import PAD_TOKEN, TEST_SEQ

MODEL_NAME = "restrictionist_2305230056"

DEVICE = torch.device(
    "mps"
    if torch.backends.mps.is_built()
    else torch.device("cuda:0")
    if torch.cuda.device_count() > 0
    else torch.device("cpu")
)


if __name__ == "__main__":
    model, config, stats = load_model(MODEL_NAME, TransformerEncoderDecoderNew)
    model = model.to(DEVICE)
    n_seqs = 10

    part_1 = config["data"]["part_1"]
    part_2 = config["data"]["part_2"]

    n_ticks = config["sequence_len"]

    write_generations = True
    gen_dir = os.path.join(MODELS_DIR, MODEL_NAME, "inference")
    if write_generations:
        if not os.path.isdir(gen_dir):
            os.makedirs(gen_dir)

    generated_rolls = []
    generated_descs = []
    print(f"Generating {n_seqs} sequences")
    for _ in tqdm(range(n_seqs)):
        # Generate a new sequence starting with a padding token (idy) given a full sequence (idx)
        # TODO: pull the x seq from the dataset, and also keep the original y to compare
        idx = torch.tensor([TEST_SEQ], dtype=torch.long, device=DEVICE)
        idy = torch.full((1, 1), PAD_TOKEN, dtype=torch.long, device=DEVICE)
        with torch.no_grad():
            model.eval()
            seq = model.generate(idx, idy, max_new_tokens=n_ticks - 1)[0]

        roll = get_roll_from_sequence(seq)
        generated_rolls.append(roll)

        # Compute sequence descriptors
        descs = pianoroll2descriptors(
            roll,
            config["resolution"],
            drums=part_2 == "Drums",
        )
        generated_descs.append(descs)

        if write_generations:
            for ix, roll in enumerate(generated_rolls):
                write_midi_from_roll(
                    roll,
                    outpath=os.path.join(gen_dir, f"{ix}.mid"),
                    part=part_2,
                    binary=True,
                    onset_roll=True,
                )

            # Also write the src sequence
            roll = get_roll_from_sequence(TEST_SEQ)
            write_midi_from_roll(
                roll,
                outpath=os.path.join(gen_dir, f"src.mid"),
                part=part_1,
                binary=True,
                onset_roll=True,
            )

    gen_df = pd.DataFrame(generated_descs).dropna(how="all", axis=1)
    gen_df.drop("stepDensity", axis=1, inplace=True)

    # Get distribution from training set
    full_df = PartDataset(
        dataset_name=config["data"]["dataset_name"],
        part=part_2,
        representation="descriptors",
    ).as_df()
    dataset_df = full_df.drop(["filename", "segment_id", "stepDensity"], axis=1).dropna(
        how="all", axis=1
    )

    plots_dir = os.path.join(MODELS_DIR, MODEL_NAME, "eval_plots")
    if not os.path.isdir(plots_dir):
        os.makedirs(plots_dir)

    # Combine the generated with the ground truth
    id_col = "Generated"
    gen_df[id_col] = f"Generated (n={len(gen_df)})"
    dataset_df[id_col] = f"Dataset (n={len(dataset_df)})"
    compare_df = pd.concat([gen_df, dataset_df])

    # Scale the feature columns to [0, 1]
    feature_cols = [c for c in dataset_df.columns if c != id_col]
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
    plt.title(MODEL_NAME)
    plt.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.08),
        fancybox=True,
        shadow=False,
        ncol=2,
    )
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "all-comparison.png"))
    plt.clf()
