import os

import numpy as np
import seaborn as sns
import torch
from model_utils import get_embeddings, load_model
from rhythmic_relationships import INFERENCE_DIR
from rhythmic_relationships.data import PAD_IX
from rhythmic_relationships.io import write_midi_from_roll
from rhythmic_relationships.model import RecurrentVAE

sns.set_style("white")
sns.set_context("paper")

model_name = "semblant_2305031358"


if __name__ == "__main__":
    model, config, stats = load_model(model_name, RecurrentVAE)

    # Sample and decode random points from the latent space
    n_samples = 8
    samples = torch.randn(n_samples, model.z_dim)
    decoded = model.decode(samples).view((n_samples, model.y_dim, model.context_len))
    decoded = decoded.detach().cpu().numpy()

    x = torch.full_like(torch.from_numpy(samples), PAD_IX)
    x_recon, mu, sigma = model(x)
    x_recon = x_recon.view(x.shape[0], x.shape[1], x.shape[2])

    preds = []
    for sample in x_recon:
        out = []
        for ix in range(config["model"]["y_dim"]):
            val = torch.multinomial(sample[ix], num_samples=1).item()
            out.append(val)
        preds.append(out)

    print(",".join(i for i in preds))

    part = config["dataset"]["part"]

    outdir = os.path.join(INFERENCE_DIR, model_name, "random_samples")
    if not os.path.isdir(outdir):
        os.makedirs(outdir)

    emb = get_embeddings(
        samples,
        title=f"t-SNE of 64 random samples from {model_name}\n{config['dataset']['part']} {config['dataset']['representation']}\n{config['dataset']['dataset_name']}",
        outdir=outdir,
    )

    # roll = get_roll_from_roll_seq(decoded)

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
