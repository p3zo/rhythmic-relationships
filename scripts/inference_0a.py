import os

import numpy as np
import torch
from rhythmic_complements.data import PairDataset
from rhythmic_complements.io import (
    write_image_from_roll,
    write_midi_from_pattern,
    write_midi_from_roll,
)
from rhythmic_complements.model import VariationalAutoEncoder
from torch.utils.data import DataLoader
from train_0a import config, get_model_path

if __name__ == "__main__":
    model_path = get_model_path(config)
    state_dict = torch.load(model_path, map_location=torch.device(config["device"]))
    model = VariationalAutoEncoder(**config["model"])
    model.load_state_dict(state_dict=state_dict)

    # Load a random x, y pair
    dataset = PairDataset(**config["dataset"])
    loader = DataLoader(dataset, batch_size=1, shuffle=True)
    x_in, y_in = next(iter(loader))

    x = x_in.view(x_in.shape[0], config["model"]["x_dim"])
    y = y_in.view(y_in.shape[0], config["model"]["y_dim"])

    # Encode the x with a random y
    random_y = torch.randn_like(y)
    mu, sigma = model.encode(x, random_y)

    # Sample the latent space
    epsilon = torch.randn_like(sigma)
    z = mu + sigma * epsilon

    # Reconstruct x using the sample
    recon_x = model.decode(z, random_y)
    recon_in = recon_x.view(x_in.shape).detach().numpy()

    # Apply an onset threshold
    # TODO: adjust the threshold if the input is not binarized
    out = np.ma.masked_array(recon_in, mask=(recon_in < 0.5), fill_value=0).filled()

    # Save the output
    inference_dir = os.path.join(config["dataset"]["dataset_dir"], "inference")
    if not os.path.isdir(inference_dir):
        os.makedirs(inference_dir)

    if config["dataset"]["repr_1"] == "roll":
        # Write the output roll as both MIDI and image
        write_image_from_roll(
            out, os.path.join(inference_dir, "prediction.png"), binary=True
        )
        write_midi_from_roll(
            out.T,
            os.path.join(inference_dir, "prediction.mid"),
            resolution=24,
        )

        # Also write the input for reference
        in_roll = x_in[0].numpy().astype(np.int8)
        in_roll[in_roll.nonzero()] = 80
        write_image_from_roll(in_roll, os.path.join(inference_dir, "input.png"))
        write_midi_from_roll(
            in_roll.T, os.path.join(inference_dir, "input.mid"), resolution=24
        )
    elif config["dataset"]["repr_1"] == "pattern":
        # Write the output pattern as MIDI
        write_midi_from_pattern(out.T, os.path.join(inference_dir, "prediction.mid"))

        # Write the input pattern as MIDI
        write_midi_from_pattern(x_in.T, os.path.join(inference_dir, "input.mid"))
