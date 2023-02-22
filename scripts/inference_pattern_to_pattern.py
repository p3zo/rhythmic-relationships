import os

import numpy as np
import torch
from rhythmic_complements.io import (
    write_image_from_pattern,
    write_midi_from_pattern,
)
from rhythmic_complements.model import VariationalAutoEncoder
from train_0a import config, get_model_path, get_model_name


if __name__ == "__main__":
    model_path = get_model_path(config)
    state_dict = torch.load(model_path, map_location=torch.device(config["device"]))
    model = VariationalAutoEncoder(**config["model"])
    model.load_state_dict(state_dict=state_dict)

    # User-specified y part
    y_in = np.array([1, 0, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 0, 0, 1]).astype(np.int8)

    y = torch.from_numpy(y_in).to(torch.float32).view(1, config["model"]["y_dim"])

    n_predictions = 5
    density = 0.5

    # Sample a random point in the latent space
    z = torch.randn(n_predictions, config["model"]["z_dim"])

    # Generate a new x
    recon_x = model.decode(z, y.repeat(n_predictions, 1))
    recon = recon_x.detach().numpy()

    # Apply an onset threshold
    # TODO: adjust the threshold if the input is not binarized
    outs = np.ma.masked_array(
        recon, mask=(recon < (1 - density)), fill_value=0
    ).filled()

    # Save the output
    inference_dir = os.path.join(
        config["dataset"]["dataset_dir"], "inference", get_model_name(config)
    )
    if not os.path.isdir(inference_dir):
        os.makedirs(inference_dir)

    part_1 = config["dataset"]["part_1"]
    part_2 = config["dataset"]["part_2"]

    # Write the user-specified y part
    write_midi_from_pattern(y_in, os.path.join(inference_dir, f"input_{part_2}.mid"))
    write_image_from_pattern(y_in, os.path.join(inference_dir, f"input_{part_2}.png"))

    # Write the predicted x parts
    for ix, out in enumerate(outs):
        write_midi_from_pattern(
            out, os.path.join(inference_dir, f"predicted_{part_1}_{ix}.mid"), pitch=50
        )
        write_image_from_pattern(
            out, os.path.join(inference_dir, f"predicted_{part_1}_{ix}.png")
        )
