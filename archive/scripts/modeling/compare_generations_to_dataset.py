"""Compare training set descriptors to generated after each epoch"""
import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch
from model_utils import load_model
from rhythmic_relationships import MODELS_DIR
from rhythmic_relationships.data import PartDataset, get_roll_from_sequence
from rhythmic_relationships.io import write_midi_from_roll
from rhythmic_relationships.model import TransformerDecoder
from rhythmtoolbox import pianoroll2descriptors
from tqdm import tqdm

DEVICE = torch.device("mps" if torch.backends.mps.is_built() else "cpu")

# model_name = "pedantocracy_2305062342"
model_name = "defease_2305091520"

# Load model
model, config, stats = load_model(model_name, TransformerDecoder)
print(f"{stats['n_params']=}")

part = config["dataset"]["part"]

# Use the model to generate new sequences
write_generations = False
gen_dir = os.path.join(MODELS_DIR, model_name, "inference")
if write_generations:
    if not os.path.isdir(gen_dir):
        os.makedirs(gen_dir)

model = model.to(DEVICE)
n_seqs = 1000

generated_rolls = []
generated_descs = []
print(f"Generating {n_seqs} sequences")
for _ in tqdm(range(n_seqs)):
    # TODO: programatically get start_token_ix
    start_token_ix = 1
    idx = torch.full((1, 1), start_token_ix, dtype=torch.long, device=DEVICE)
    seq = model.generate(idx, max_new_tokens=config["sequence_len"])[0][1:]
    roll = get_roll_from_sequence(seq, part=part)

    generated_rolls.append(roll)

    descs = pianoroll2descriptors(
        roll,
        config["resolution"],
        drums=part == "Drums",
    )
    generated_descs.append(descs)

    if write_generations:
        for ix, roll in enumerate(generated_rolls):
            write_midi_from_roll(
                roll,
                outpath=os.path.join(gen_dir, f"{ix}.mid"),
                part=part,
                binary=True,
                onset_roll=True,
            )


gen_df = pd.DataFrame(generated_descs).dropna(how="all", axis=1)
gen_df.drop("stepDensity", axis=1, inplace=True)

# Get distribution from training set
full_df = PartDataset(
    dataset_name=config["dataset"]["dataset_name"],
    part=part,
    representation="descriptors",
).as_df()
dataset_df = full_df.drop(["filename", "segment_id", "stepDensity"], axis=1).dropna(
    how="all", axis=1
)

plots_dir = os.path.join(MODELS_DIR, model_name, "eval_plots")
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
plt.title(model_name)
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


# Plot descriptor means from post-epoch evaluations, with dataset means for reference
epoch_descs = [i["generated_descriptors"] for i in stats["epoch_evals"]]
epoch_df = pd.concat(
    [
        pd.DataFrame(i["generated_descriptors"], index=[ix] * 3)
        for ix, i in enumerate(stats["epoch_evals"])
    ]
)
epoch_df = epoch_df.dropna(how="all", axis=1).drop("stepDensity", axis=1)
epochs = range(epoch_df.index.max())
# # TODO: rework this
# for col in epoch_df:
#     plt.plot(epochs, epoch_df[col].values, label=col)
#     d = dataset_df[col].describe()
#     low, mid, high = d["mean"] - d["std"], d["mean"], d["mean"] + d["std"]
#     plt.axhline(y=mid, color="black", lw=0.5, ls="--")
#     plt.axhline(y=low, color="black", lw=0.25, ls="--")
#     plt.axhline(y=high, color="black", lw=0.25, ls="--")
#     plt.yticks([low, mid, high], ["mean - std", "dataset mean", "mean + std"])
#     plt.title(col)
#     plt.xlabel("epoch")
#     plt.tight_layout()
#     plt.savefig(os.path.join(plots_dir, f"epoch_means_{col}.png"))
#     plt.clf()
