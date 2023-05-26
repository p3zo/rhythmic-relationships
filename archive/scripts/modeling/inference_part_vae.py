import os

import numpy as np
import seaborn as sns
import torch
from model_utils import get_embeddings, load_model
from rhythmic_relationships import INFERENCE_DIR
from rhythmic_relationships.io import write_image_from_roll, write_midi_from_roll
from rhythmic_relationships.model import VAE

sns.set_style("white")
sns.set_context("paper")

model_name = "resultancy_lmdc_3000_2bar_4res_Guitar_onset_roll_230425143940"

if __name__ == "__main__":
    model, config, stats = load_model(model_name, VAE)

    n_samples = 64

    # Sample and decode random points from the latent space
    samples = torch.randn(n_samples, config["model"]["z_dim"])
    decoded = model.decode(samples).view((n_samples, 32, 128))
    if config["loss_fn"] == "bce-logits":
        decoded = torch.sigmoid(decoded)
    decoded = decoded.detach().cpu().numpy()

    part = config["dataset"]["part"]

    outdir = os.path.join(INFERENCE_DIR, model_name, "random_samples")
    if not os.path.isdir(outdir):
        os.makedirs(outdir)

    emb = get_embeddings(
        samples,
        title=f"t-SNE of 64 samples from {model_name}\n{config['dataset']['part']} {config['dataset']['representation']}\n{config['dataset']['dataset_name']}",
        outdir=outdir,
    )

    for ix, d in enumerate(decoded):
        # Apply a range of onset thresholds
        for thresh in [i / 10 for i in range(0, 10)]:
            threshed = np.ma.masked_array(d, mask=(d < thresh), fill_value=0).filled()

            # Skip empty predictions
            if threshed.max() == 0:
                continue

            write_midi_from_roll(
                threshed,
                os.path.join(
                    outdir,
                    f"thresh{thresh}_{part}_{ix}.mid",
                ),
                part=part,
                onset_roll=True,
            )
            write_image_from_roll(
                threshed,
                os.path.join(outdir, f"thresh{thresh}_{part}_{ix}.png"),
                binary=True,
            )
