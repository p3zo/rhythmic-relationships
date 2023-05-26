import os

import numpy as np
import torch
from rhythmic_relationships.io import write_image_from_pattern, write_midi_from_pattern
from train_part_pair_vae import load_model

INFERENCE_DIR = "../output/inference"

model_name = (
    "surgerize_lmd_clean_2_bar_24_res_5000_Bass_Drums_pattern_pattern_230222090629"
)

if __name__ == "__main__":
    model, config = load_model(model_name)

    # User-specified y part
    y_in = np.array(
        [1, 2, 2, 2, 2, 2, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] * 8
    ).astype(np.int8)

    y = torch.from_numpy(y_in).to(torch.float32).view(1, config["model"]["y_dim"])

    n_predictions = 5
    density = 0

    # Sample a random point in the latent space
    z = torch.randn(n_predictions, config["model"]["z_dim"])

    # Generate a new x
    recon_x = model.decode(z, y.repeat(n_predictions, 1))
    recon = recon_x.detach().numpy()

    # Apply an onset threshold
    # TODO: adjust the threshold if the input is not binary
    outs = np.ma.masked_array(
        recon, mask=(recon < (1 - density)), fill_value=0
    ).filled()

    # Save the output
    outdir = os.path.join(INFERENCE_DIR, model_name)
    if not os.path.isdir(outdir):
        os.makedirs(outdir)

    part_1 = config["dataset"]["part_1"]
    part_2 = config["dataset"]["part_2"]

    # Write the user-specified y part
    write_midi_from_pattern(y_in, os.path.join(outdir, f"input_{part_2}.mid"))
    write_image_from_pattern(y_in, os.path.join(outdir, f"input_{part_2}.png"))

    # Write the predicted x parts
    for ix, out in enumerate(outs):
        write_midi_from_pattern(
            out, os.path.join(outdir, f"predicted_{part_1}_{ix}.mid"), pitch=50
        )
        write_image_from_pattern(
            out, os.path.join(outdir, f"predicted_{part_1}_{ix}.png")
        )
