import argparse

import torch

from rhythmic_complements.model import VariationalAutoEncoder

DEVICE = torch.device("cpu")
INPUT_WIDTH = 18
INPUT_HEIGHT = 1
INPUT_DIM = INPUT_WIDTH * INPUT_HEIGHT
H_DIM = 4
Z_DIM = 2


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path",
        type=str,
        default="output/lmd_clean_1bar_24res/models/Drums_rsp.pt",
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
    model = VariationalAutoEncoder(INPUT_DIM, H_DIM, Z_DIM)
    model.load_state_dict(state_dict=state_dict)

    samples = model.sample(n_samples=2, device=DEVICE).detach().cpu().numpy()
    print(samples)
