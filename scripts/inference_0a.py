import os

import numpy as np
import torch
from rhythmic_complements.data import PairDataset
from rhythmic_complements.io import (
    write_image_from_pattern,
    write_image_from_roll,
    write_midi_from_pattern,
    write_midi_from_roll,
)
from rhythmic_complements.model import VariationalAutoEncoder
from torch.utils.data import DataLoader
from train_0a import config, get_model_path, get_model_name


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

    # Sample a random point in the latent space
    z = torch.randn(1, config["model"]["z_dim"])

    # Generate a new x
    recon_x = model.decode(z, y)
    recon = recon_x.view(x_in.shape).detach().numpy()

    # Apply an onset threshold
    # TODO: adjust the threshold if the input is not binarized
    out = np.ma.masked_array(recon, mask=(recon < 0.5), fill_value=0).filled()[0]

    # Save the output
    inference_dir = os.path.join(
        config["dataset"]["dataset_dir"], "inference", get_model_name(config)
    )
    if not os.path.isdir(inference_dir):
        os.makedirs(inference_dir)

    input_x = x_in[0].numpy().astype(np.int8)
    input_y = y_in[0].numpy().astype(np.int8)

    part_1 = config["dataset"]["part_1"]
    part_2 = config["dataset"]["part_2"]

    if config["dataset"]["repr_1"] == "roll":
        # Assign the onsets a reasonable MIDI velocity
        out[out.nonzero()] = 80
        out = out.astype(np.int8)

        # Write the prediction
        write_midi_from_roll(
            out,
            os.path.join(inference_dir, f"predicted_{part_1}.mid"),
            resolution=24,
        )
        write_image_from_roll(
            out, os.path.join(inference_dir, f"predicted_{part_1}.png")
        )

        # Write the x from the dataset
        write_midi_from_roll(
            input_x,
            os.path.join(inference_dir, f"original_{part_1}.mid"),
            resolution=24,
        )
        write_image_from_roll(
            input_x, os.path.join(inference_dir, f"original_{part_1}.png")
        )
    elif config["dataset"]["repr_1"] == "pattern":
        # Write the prediction
        write_midi_from_pattern(
            out, os.path.join(inference_dir, f"predicted_{part_1}.mid"), pitch=50
        )
        write_image_from_pattern(
            out, os.path.join(inference_dir, f"predicted_{part_1}.png")
        )

        # Write the x from the dataset
        write_midi_from_pattern(
            input_x, os.path.join(inference_dir, f"original_{part_1}.mid"), pitch=50
        )
        write_image_from_pattern(
            input_x, os.path.join(inference_dir, f"original_{part_1}.png")
        )

    if config["dataset"]["repr_2"] == "pattern":
        # Write the y from the dataset
        write_midi_from_pattern(
            input_y, os.path.join(inference_dir, f"original_{part_2}.mid")
        )
        write_image_from_pattern(
            input_y, os.path.join(inference_dir, f"original_{part_2}.png")
        )
