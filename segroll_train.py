import argparse
import os

import torch
from torch.utils.data import DataLoader

from rhythmic_complements.data import SegrollDataset
from rhythmic_complements.model import VariationalAutoEncoder
from rhythmic_complements.train import train

DATASET_DIR = "output/lmd_clean_1bar_24res/"

DEVICE = torch.device("mps" if torch.backends.mps.is_built() else "cpu")
INPUT_DIM = 96 * 88
H_DIM = 200
Z_DIM = 20
NUM_EPOCHS = 5
BATCH_SIZE = 32
LR = 1e-4

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_dir",
        type=str,
        default=DATASET_DIR,
        help="Path to the dataset.",
    )
    parser.add_argument(
        "--part",
        type=str,
        default="Bass",
        help="Name of the part to train the model for. See `prepare_data.py` for a list of part names.",
    )
    args = parser.parse_args()

    dataset_dir = args.dataset_dir
    part = args.part

    data_filepath = os.path.join(dataset_dir, "part_segrolls", f"{part}.npz")
    data = SegrollDataset(data_filepath)
    loader = DataLoader(dataset=data, batch_size=BATCH_SIZE, shuffle=True)

    model = VariationalAutoEncoder(INPUT_DIM, H_DIM, Z_DIM).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    train(
        model,
        loader,
        optimizer,
        INPUT_DIM,
        device=DEVICE,
        num_epochs=NUM_EPOCHS,
        clip_gradients=True,
    )

    model_path = os.path.join(dataset_dir, "models", f"{part}_segroll.pt")
    torch.save(model.state_dict(), model_path)
    print(f"Saved {model_path}")
