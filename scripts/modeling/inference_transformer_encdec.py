import os

import matplotlib.pyplot as plt
import pandas as pd
import torch
import seaborn as sns
from torch.utils.data import DataLoader
from rhythmtoolbox import pianoroll2descriptors

from model_utils import load_model
from rhythmic_relationships import MODELS_DIR
from rhythmic_relationships.data import (
    PartDataset,
    PartPairDataset,
    get_roll_from_sequence,
    tokenize_roll,
)
from rhythmic_relationships.io import write_midi_from_roll
from rhythmic_relationships.model import TransformerEncoderDecoderNew
from rhythmic_relationships.vocab import get_vocab_encoder_decoder


MODEL_NAME = "noncannibalistic_2305260103"

DEVICE = torch.device("cpu")
# DEVICE = torch.device(
#     "mps"
#     if torch.backends.mps.is_built()
#     else torch.device("cuda:0")
#     if torch.cuda.device_count() > 0
#     else torch.device("cpu")
# )


if __name__ == "__main__":
    model, config, stats = load_model(MODEL_NAME, TransformerEncoderDecoderNew)
    model = model.to(DEVICE)

    n_seqs = 50
    part_1 = config["data"]["part_1"]
    part_2 = config["data"]["part_2"]
    n_ticks = config["sequence_len"]
    write_midi = True

    gen_dir = os.path.join(MODELS_DIR, MODEL_NAME, "inference")
    if write_midi:
        if not os.path.isdir(gen_dir):
            os.makedirs(gen_dir)

    # Generate new sequences using part_1s from the dataset and just a start token for part_2
    print(f"Generating {n_seqs} sequences")
    generated_rolls = []
    generated_descs = []
    n_generated = 0
    all_zeros = 0

    del config["data"]["context_len"]
    dataset = PartPairDataset(**config["data"])
    loader = DataLoader(dataset, batch_size=n_seqs, shuffle=True)

    encode, _ = get_vocab_encoder_decoder(config["data"]["part_2"])
    start_ix = encode(["start"])[0]
    idy = torch.full((n_seqs, 1), start_ix, dtype=torch.long, device=DEVICE)

    src_rolls, tgt_rolls = next(iter(loader))
    srcs = torch.LongTensor(
        [tokenize_roll(i.numpy(), config["data"]["part_1"]) for i in src_rolls]
    ).to(DEVICE)
    tgts = torch.LongTensor(
        [tokenize_roll(i.numpy(), config["data"]["part_2"]) for i in tgt_rolls]
    ).to(DEVICE)

    seqs = (
        model.generate(srcs, idy, max_new_tokens=config["sequence_len"]).cpu().numpy()
    )
    srcs = srcs.cpu().numpy()
    tgts = tgts.cpu().numpy()

    for ix, seq in enumerate(seqs):
        gen_roll = get_roll_from_sequence(seq, part=config["data"]["part_2"])

        n_generated += 1
        if gen_roll.max() == 0:
            all_zeros += 1
            continue

        generated_rolls.append(gen_roll)

        # Compute sequence descriptors
        descs = pianoroll2descriptors(
            gen_roll,
            config["resolution"],
            drums=config["data"]["part_2"] == "Drums",
        )
        generated_descs.append(descs)

        if write_midi:
            write_midi_from_roll(
                gen_roll,
                outpath=os.path.join(gen_dir, f"{ix}_gen.mid"),
                part=part_2,
                binary=False,
                onset_roll=True,
            )

            write_midi_from_roll(
                src_rolls[ix].numpy(),
                outpath=os.path.join(gen_dir, f"{ix}_src.mid"),
                part=part_1,
                binary=False,
                onset_roll=True,
            )

            write_midi_from_roll(
                tgt_rolls[ix].numpy(),
                outpath=os.path.join(gen_dir, f"{ix}_tgt.mid"),
                part=part_2,
                binary=False,
                onset_roll=True,
            )

    print(f"{n_generated=}, {all_zeros=} ({100*round(all_zeros/n_generated, 2)}%)")

    gen_df = pd.DataFrame(generated_descs).dropna(how="all", axis=1)
    if "stepDensity" in gen_df.columns:
        gen_df.drop("stepDensity", axis=1, inplace=True)

    # Get distribution from training set
    full_df = PartDataset(
        dataset_name=config["data"]["dataset_name"],
        part=part_2,
        representation="descriptors",
    ).as_df(subset=10000)
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
