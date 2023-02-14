import os

import numpy as np
import pandas as pd
import torch
from rhythmic_complements.data import DescriptorDataset, SegrollDataset
from rhythmic_complements.io import write_midi_file, write_pil_image
from rhythmic_complements.model import VariationalAutoEncoder
from rhythmtoolbox import pianoroll2descriptors
from torch.utils.data import DataLoader

DEVICE = torch.device("cpu")

SEGROLL_INPUT_WIDTH = 96
SEGROLL_INPUT_HEIGHT = 88
SEGROLL_INPUT_DIM = SEGROLL_INPUT_WIDTH * SEGROLL_INPUT_HEIGHT
SEGROLL_H_DIM = 200
SEGROLL_Z_DIM = 20

RSP_INPUT_DIM = 18
RSP_H_DIM = 6
RSP_Z_DIM = 3
RSP_N_LABELS = RSP_INPUT_DIM


def get_distance(a, b):
    """Compute the Euclidean distance between two vectors"""
    return np.linalg.norm(a - b)


if __name__ == "__main__":
    dataset_dir = "../output/lmd_clean_1bar_24res"

    # Take a Drums roll as input
    drums_path = os.path.join(dataset_dir, "part_segrolls", "Drums.npz")
    drums_dataset = SegrollDataset(drums_path)
    drums_loader = DataLoader(drums_dataset, batch_size=1, shuffle=True)
    drums_roll = next(iter(drums_loader))[0].numpy().astype(np.uint8)

    # Compute its rhythmic descriptors
    drums_rsp = pianoroll2descriptors(drums_roll)

    # Predict a Bass RSP using the RSP Pair model
    drums_bass_vae_path = os.path.join(dataset_dir, "models", "Drums_Bass_rsp_pair.pt")
    state_dict = torch.load(drums_bass_vae_path, map_location=torch.device(DEVICE))
    pair_model = VariationalAutoEncoder(
        RSP_INPUT_DIM, RSP_H_DIM, RSP_Z_DIM, conditional=True, n_labels=RSP_N_LABELS
    )
    pair_model.load_state_dict(state_dict=state_dict)
    drums_rsp_tensor = torch.tensor(list(drums_rsp.values())).to(torch.float32)
    random_rsp_tensor = torch.randn_like(drums_rsp_tensor)
    mu, sigma = pair_model.encode(drums_rsp_tensor, random_rsp_tensor)
    epsilon = torch.randn_like(sigma)
    drums_z = mu + sigma * epsilon
    random_latent = torch.randn_like(drums_z)
    bass_rsp = pair_model.decode(drums_z, random_rsp_tensor)

    # Compute its distance from all RSPs in the Bass descriptor dataset
    bass_rsp_dataset_path = os.path.join(dataset_dir, "descriptors", "Bass.csv")
    bass_rsp_dataset = DescriptorDataset(bass_rsp_dataset_path)
    bass_loader = DataLoader(bass_rsp_dataset, batch_size=1, shuffle=True)
    distances = []
    for rsp in bass_loader:
        distances.append(get_distance(rsp, bass_rsp.detach()))

    # Get the corresponding roll for the closest Bass RSP
    closest_ix = np.array(distances).argmin()
    bass_rsp_df = pd.read_csv(bass_rsp_dataset_path)
    closest = bass_rsp_df.iloc[closest_ix]
    npz_filepath = (
        os.path.splitext(closest.filepath.replace("input", dataset_dir))[0] + ".npz"
    )
    intermediate_bass_roll = np.load(npz_filepath)["Bass"][closest.segment_id]
    intermediate_bass_roll_tensor = (
        torch.from_numpy(intermediate_bass_roll)
        .to(torch.float32)
        .view(1, SEGROLL_INPUT_DIM)
    )

    # Generate a similar Bass roll
    bass_vae_path = os.path.join(dataset_dir, "models", "Bass_segroll.pt")
    state_dict = torch.load(bass_vae_path, map_location=torch.device(DEVICE))
    bass_vae = VariationalAutoEncoder(SEGROLL_INPUT_DIM, SEGROLL_H_DIM, SEGROLL_Z_DIM)
    bass_vae.load_state_dict(state_dict=state_dict)
    mu, sigma = bass_vae.encode(intermediate_bass_roll_tensor)
    epsilon = torch.randn_like(sigma)
    bass_z = mu + sigma * epsilon
    bass_roll = bass_vae.decode(bass_z)
    bass_roll = bass_roll.view(SEGROLL_INPUT_WIDTH, SEGROLL_INPUT_HEIGHT).detach()

    output_bass_roll = np.array(
        list(map(lambda x: np.interp(x, [0, 1], [0, 127]), bass_roll))
    ).astype(int)

    # Write the output as MIDI and as an image
    result_dir = os.path.join(dataset_dir, "predictions")
    if not os.path.isdir(result_dir):
        os.makedirs(result_dir)

    write_pil_image(output_bass_roll, os.path.join(result_dir, "bass_prediction.png"))
    write_midi_file(
        output_bass_roll.T,
        os.path.join(result_dir, "bass_prediction.mid"),
        resolution=24,
    )

    # Also write the input for reference
    write_pil_image(drums_roll, os.path.join(result_dir, "drum_input.png"))
    write_midi_file(
        drums_roll.T, os.path.join(result_dir, "drum_input.mid"), resolution=24
    )

    # And the intermediate bass roll
    write_pil_image(
        intermediate_bass_roll, os.path.join(result_dir, "bass_intermediate.png")
    )
    write_midi_file(
        intermediate_bass_roll.T,
        os.path.join(result_dir, "bass_intermediate.mid"),
        resolution=24,
    )
