import os

import numpy as np
import torch
from model_utils import load_model
from rhythmic_relationships import INFERENCE_DIR
from rhythmic_relationships.data import PartPairDataset
from rhythmic_relationships.io import (
    write_image_from_hits,
    write_image_from_roll,
    write_midi_from_hits,
    write_midi_from_roll,
)
from torch.utils.data import DataLoader


# model_name = (
#     "surgerize_lmd_clean_2_bar_24_res_5000_Bass_Drums_pattern_pattern_230222090629"
# )
model_name = (
    "antroscopy_lmdc_3000_1bar_4res_Guitar_Bass_onset_roll_onset_roll_230420235010"
)

if __name__ == "__main__":
    model, config = load_model(model_name)

    # Load a random x, y pair
    dataset = PartPairDataset(**config["dataset"])
    loader = DataLoader(dataset, batch_size=1, shuffle=True)
    x_in, y_in = next(iter(loader))

    x = x_in.view(x_in.shape[0], config["model"]["x_dim"])
    y = y_in.view(y_in.shape[0], config["model"]["y_dim"])

    # Sample a random point in the latent space
    z = torch.randn(1, config["model"]["z_dim"])

    # Generate a new x
    recon_x = model.decode(z, y)
    recon = recon_x.view(x_in.shape).detach().numpy()

    # Apply an onset threshold
    # TODO: adjust the threshold if the input is not binary
    out = np.ma.masked_array(recon, mask=(recon < 0.5), fill_value=0).filled()

    # Save the output
    outdir = os.path.join(INFERENCE_DIR, model_name, "random_samples")
    if not os.path.isdir(outdir):
        os.makedirs(outdir)

    input_x = x_in[0].numpy().astype(np.int8)
    input_y = y_in[0].numpy().astype(np.int8)

    part_1 = config["dataset"]["part_1"]
    part_2 = config["dataset"]["part_2"]

    if config["dataset"]["repr_1"] in ["roll", "onset_roll"]:
        # # Assign the onsets a reasonable MIDI velocity
        # out[out.nonzero()] = 80
        # out = out.astype(np.int8)

        # Write the prediction
        write_midi_from_roll(
            out,
            os.path.join(outdir, f"predicted_{part_1}.mid"),
            resolution=24,
            part=part_1,
            onset_roll=True,
        )
        write_image_from_roll(
            out, os.path.join(outdir, f"predicted_{part_1}.png"), binary=True
        )

        # Write the x from the dataset
        write_midi_from_roll(
            input_x,
            os.path.join(outdir, f"original_{part_1}.mid"),
            resolution=24,
        )
        write_image_from_roll(
            input_x, os.path.join(outdir, f"original_{part_1}.png"), binary=True
        )
    elif config["dataset"]["repr_1"] == "hits":
        # Write the prediction
        write_midi_from_hits(
            out, os.path.join(outdir, f"predicted_{part_1}.mid"), pitch=50
        )
        write_image_from_hits(out, os.path.join(outdir, f"predicted_{part_1}.png"))

        # Write the x from the dataset
        write_midi_from_hits(
            input_x, os.path.join(outdir, f"original_{part_1}.mid"), pitch=50
        )
        write_image_from_hits(input_x, os.path.join(outdir, f"original_{part_1}.png"))

    if config["dataset"]["repr_2"] == "hits":
        # Write the y from the dataset
        write_midi_from_hits(input_y, os.path.join(outdir, f"original_{part_2}.mid"))
        write_image_from_hits(input_y, os.path.join(outdir, f"original_{part_2}.png"))
