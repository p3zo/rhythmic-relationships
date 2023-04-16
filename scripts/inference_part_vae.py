import os

import numpy as np
import pandas as pd
import seaborn as sns
import torch
from model_utils import load_model
from rhythmic_relationships.io import write_image_from_roll, write_midi_from_roll
from rhythmic_relationships.model import VariationalAutoEncoder
from sklearn.manifold import TSNE
from utils import save_fig

sns.set_style("white")
sns.set_context("paper")


INFERENCE_DIR = "../output/inference"

model_name = "vergence_lmdc_500_1bar_4res_Guitar_binary onset roll_230415235749"


def get_embeddings(X, title="", outdir="."):
    """Create a 2D embedding space of the data, plot it, and save it to a csv"""
    reducer = TSNE(n_components=2, init="pca", learning_rate="auto", random_state=42)

    # Make the pair space using t-SNE
    X_transform = reducer.fit_transform(X)

    emb = pd.DataFrame(X_transform, columns=["x", "y"])
    sns.relplot(
        data=emb,
        x="x",
        y="y",
        height=8,
        aspect=1.25,
        legend=False,
    )

    save_fig(os.path.join(outdir, "latent_samples.png"), title=title)


if __name__ == "__main__":
    model, config = load_model(model_name, VariationalAutoEncoder)

    n_samples = 64

    # Sample and decode random points from the latent space
    samples = torch.randn(n_samples, config["model"]["z_dim"])
    decoded = model.decode(samples).view((n_samples, 16, 128)).detach().cpu().numpy()

    part = config["dataset"]["part"]

    outdir = os.path.join(INFERENCE_DIR, model_name)
    if not os.path.isdir(outdir):
        os.makedirs(outdir)

    get_embeddings(
        samples,
        title=f"t-SNE of 64 samples from {model_name}\n{config['dataset']['part']} {config['dataset']['representation']}\n{config['dataset']['dataset_name']}",
        outdir=outdir,
    )

    for ix, d in enumerate(decoded):
        # Apply an onset threshold
        # TODO: adjust the threshold if the input is not binary
        out = np.ma.masked_array(d, mask=(d < 0.5), fill_value=0).filled()

        write_midi_from_roll(
            out,
            os.path.join(outdir, f"predicted_{part}_{ix}.mid"),
        )
        write_image_from_roll(
            out, os.path.join(outdir, f"predicted_{part}_{ix}.png"), binary=True
        )
