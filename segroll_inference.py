import argparse
import os

import numpy as np
import pretty_midi as pm
import torch

from prepare_data import create_pil_image
from rhythmic_complements.model import VariationalAutoEncoder
from segroll_train import H_DIM, Z_DIM

DEVICE = torch.device("cpu")
INPUT_WIDTH = 96
INPUT_HEIGHT = 88
INPUT_DIM = INPUT_WIDTH * INPUT_HEIGHT


def inference(roll):
    """Generates a variations of a roll.
    Computes the mu, sigma representation of the roll and uses them to decode new samples.
    TODO: does this belong in the model class
    """
    with torch.no_grad():
        mu, sigma = model.encode(roll.view(1, INPUT_DIM))
    epsilon = torch.randn_like(sigma)
    z = mu + sigma * epsilon
    out = model.decode(z)
    return out.view(INPUT_WIDTH, INPUT_HEIGHT).detach()


def write_midi(roll, filepath, resolution):
    note_duration = 0.5 / resolution  # a reasonable bpm close to 120 (?)

    instrument = pm.Instrument(program=0, is_drum=False)
    for voice in range(len(roll)):
        events = roll[voice]
        for event_ix, vel in enumerate(events):
            start = event_ix * note_duration
            note = pm.Note(
                velocity=vel, pitch=voice, start=start, end=start + note_duration
            )
            instrument.notes.append(note)

    track = pm.PrettyMIDI(resolution=resolution)
    track.instruments.append(instrument)
    track.write(filepath)


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
        create_pil_image(roll, os.path.join(sample_dir, f"sample_{ix}.png"))
        write_midi(roll, os.path.join(sample_dir, f"sample_{ix}.mid"), resolution=24)

    # Compute rhythmic descriptors for many samples and compare their distribution to the dataset distribution
    import matplotlib.pyplot as plt
    import pandas as pd
    from rhythmtoolbox import pianoroll2descriptors

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
