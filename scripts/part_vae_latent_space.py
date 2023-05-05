"""
Sample segments from a PartDataset and visualize them in the latent space of a trained model.
Also write the MIDI of both the original and reconstructed segments.
"""

import os

import numpy as np
import torch
from model_utils import get_embeddings, load_model
from rhythmic_relationships import INFERENCE_DIR
from rhythmic_relationships.data import PartDataset
from rhythmic_relationships.io import write_midi_from_roll
from rhythmic_relationships.model import VAE
from torch.utils.data import DataLoader

# model_name = "Apocrita_lmdc_3000_1bar_4res_Guitar_onset_roll_230416150734"
# model_name = "saussuritic_lmdc_3000_1bar_4res_Bass_onset_roll_230416220437"
# model_name = "resultancy_lmdc_3000_2bar_4res_Guitar_onset_roll_230425143940"
model_name = "fadingness_lmdc_3000_2bar_4res_Guitar_onset_roll_230425160100"

WRITE_ORIGINAL_MIDI = True
WRITE_RECONSTRUCTED_MIDI = True

if __name__ == "__main__":
    n_samples = 10

    print(f"Loading model {model_name}")
    model, config = load_model(model_name, VAE)
    part = config["dataset"]["part"]

    # Load samples from the dataset
    dataset = PartDataset(**config["dataset"])
    if not n_samples:
        n_samples = len(dataset)
    print(f"{n_samples=}")
    loader = DataLoader(dataset, batch_size=n_samples, shuffle=True)
    x_in = next(iter(loader))

    # Create the output directories
    outdir = os.path.join(INFERENCE_DIR, model_name, "dataset_samples")
    midi_orig_outdir = os.path.join(outdir, "midi_original")
    midi_recon_outdir = os.path.join(outdir, "midi_reconstructed")
    for d in [outdir, midi_orig_outdir, midi_recon_outdir]:
        if not os.path.isdir(d):
            os.makedirs(d)

    # Write the sampled segments
    if WRITE_ORIGINAL_MIDI:
        for ix, xi in enumerate(x_in):
            write_midi_from_roll(
                xi.numpy(),
                os.path.join(midi_orig_outdir, f"{part}_{ix}.mid"),
                part=part,
                onset_roll=True,
            )

    # Encode the samples to the latent space
    x = x_in.view(x_in.shape[0], config["model"]["x_dim"])
    mu, sigma = model.encode(x)
    epsilon = torch.randn_like(sigma)
    z = mu + sigma * epsilon

    emb = get_embeddings(
        z.detach().cpu().numpy(),
        title=f"""t-SNE of {n_samples} {config['dataset']['part']} {config['dataset']['representation']} samples from
        {config['dataset']['dataset_name']}\nencoded with {model_name}""",
        outdir=outdir,
    )
    emb_normed = (emb - emb.min()) / (emb.max() - emb.min())
    emb_normed.to_csv(
        os.path.join(outdir, f"emb_normed_{part}_{n_samples}.csv"), index=False
    )

    n_ticks = 32
    decoded = model.decode(z).view((n_samples, n_ticks, 128))
    if config["loss_fn"] == "bce-logits":
        decoded = torch.sigmoid(decoded)
    decoded = decoded.detach().cpu().numpy()

    # Write the reconstructed segments
    if WRITE_RECONSTRUCTED_MIDI:
        for ix, recon in enumerate(decoded):
            # Apply a range of onset thresholds
            # TODO: adjust the threshold if the input is not binary
            for thresh in [i / 10 for i in range(0, 10, 2)]:
                threshed = np.ma.masked_array(
                    recon, mask=(recon < thresh), fill_value=0
                ).filled()

                # Skip empty reconstructions
                if threshed.max() == 0:
                    # To preserve ordering of output MIDI, write a single, inaudible note at the start of each bar
                    for i in range(0, n_ticks, 16):
                        threshed[i][0] = 0.1

                write_midi_from_roll(
                    threshed,
                    os.path.join(
                        midi_recon_outdir,
                        f"thresh{thresh}_{part}_{ix}.mid",
                    ),
                    part=part,
                    onset_roll=True,
                )
