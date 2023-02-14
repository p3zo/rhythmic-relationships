import argparse
import os

import numpy as np
import torch
from rhythmic_complements.io import write_midi_file, write_pil_image
from rhythmic_complements.model import VariationalAutoEncoder
from segroll_train import H_DIM, Z_DIM

DEVICE = torch.device("cpu")
INPUT_WIDTH = 96
INPUT_HEIGHT = 88
INPUT_DIM = INPUT_WIDTH * INPUT_HEIGHT


def inference(roll):
    """Generates a variations of a roll.
    Computes the mu, sigma representation of the roll and uses them to decode new samples.
    TODO: does this belong in the model class?
    """
    with torch.no_grad():
        mu, sigma = model.encode(roll.view(1, INPUT_DIM))
    epsilon = torch.randn_like(sigma)
    z = mu + sigma * epsilon
    out = model.decode(z)
    return out.view(INPUT_WIDTH, INPUT_HEIGHT).detach()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path",
        type=str,
        default="output/lmd_clean_1bar_24res/models/Drums_segroll.pt",
        help="Path to the model.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=".",
        help="Directory in which to write result.",
    )
    args = parser.parse_args()

    model_path = args.model_path
    output_dir = args.output_dir

    state_dict = torch.load(model_path, map_location=torch.device(DEVICE))
    model = VariationalAutoEncoder(INPUT_DIM, H_DIM, Z_DIM).to(DEVICE)
    model.load_state_dict(state_dict=state_dict)

    sample_dir = os.path.join(output_dir, "Drums_generated")
    if not os.path.isdir(sample_dir):
        os.makedirs(sample_dir)

    samples = model.sample(5, DEVICE).detach().cpu().numpy()
    for ix, sample in enumerate(samples):
        ss = sample.reshape((INPUT_WIDTH, INPUT_HEIGHT))
        roll = np.array(list(map(lambda x: np.interp(x, [0, 1], [0, 127]), ss))).astype(
            int
        )
        write_pil_image(roll, os.path.join(sample_dir, f"sample_{ix}.png"))
        write_midi_file(
            roll, os.path.join(sample_dir, f"sample_{ix}.mid"), resolution=24
        )
