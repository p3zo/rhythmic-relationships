import os

import numpy as np
import seaborn as sns
import torch
from model_utils import get_embeddings, load_model
from rhythmic_relationships import INFERENCE_DIR
from rhythmic_relationships.io import write_image_from_roll, write_midi_from_roll
from rhythmic_relationships.model import VariationalAutoEncoder

sns.set_style("white")
sns.set_context("paper")


# model_name = "vergence_lmdc_500_1bar_4res_Guitar_binary onset roll_230415235749"
model_name = "Apocrita_lmdc_3000_1bar_4res_Guitar_onset_roll_230416150734"


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

    emb = get_embeddings(
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
            part=part,
            onset_roll=True,
        )
        write_image_from_roll(
            out, os.path.join(outdir, f"predicted_{part}_{ix}.png"), binary=True
        )
