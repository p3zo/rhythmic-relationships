"""Compute rhythmic descriptors for many samples and compare their distribution to the dataset distribution"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from rhythmic_complements.model import VariationalAutoEncoder
from rhythmtoolbox import pianoroll2descriptors
from segroll_train import H_DIM, Z_DIM

DEVICE = torch.device("cpu")
INPUT_WIDTH = 96
INPUT_HEIGHT = 88
INPUT_DIM = INPUT_WIDTH * INPUT_HEIGHT


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

    descriptors = []
    n_samples = 1000000
    samples = model.sample(n_samples, DEVICE).detach().cpu().numpy()
    for sample in samples:
        ss = sample.reshape((INPUT_WIDTH, INPUT_HEIGHT))
        descriptors.append(pianoroll2descriptors(ss, resolution=24))
    df = pd.DataFrame(descriptors)
    pd.plotting.scatter_matrix(df, figsize=(20, 20))
    plt.title(f"{n_samples} Drums samples")
    plt.savefig("samples_rhythm_descriptors.png")

    dataset_filepath = os.path.join(
        "output/lmd_clean_1bar_24res", "part_segrolls", f"Drums.npz"
    )
    npz = np.load(dataset_filepath)
    dataset = npz["segrolls"]
    dataset_descriptors = []
    for roll in dataset:
        rr = roll.reshape((INPUT_WIDTH, INPUT_HEIGHT))
        dataset_descriptors.append(pianoroll2descriptors(rr, resolution=24))
    dataset_df = pd.DataFrame(dataset_descriptors)
    print("plotting")
    pd.plotting.scatter_matrix(dataset_df, figsize=(20, 20))
    plt.title(f"{len(dataset)} Drums segrolls")
    plt.savefig("dataset_rhythm_descriptors.png")
