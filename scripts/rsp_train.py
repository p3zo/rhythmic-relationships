import argparse
import os

import torch
from rhythmic_complements.data import DescriptorDataset
from rhythmic_complements.model import VariationalAutoEncoder
from rhythmic_complements.train import train
from torch.utils.data import DataLoader

DATASET_DIR = "output/lmd_clean_1bar_24res"

DEVICE = torch.device("mps" if torch.backends.mps.is_built() else "cpu")
INPUT_DIM = 18
H_DIM = 4
Z_DIM = 2
NUM_EPOCHS = 20
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
        default="Drums",
        help="The first of two parts to load.",
    )
    args = parser.parse_args()

    dataset_dir = args.dataset_dir
    part = args.part

    desc_path = os.path.join(dataset_dir, "descriptors", f"{part}.csv")
    data = DescriptorDataset(desc_path)
    loader = DataLoader(data, batch_size=BATCH_SIZE, shuffle=True)

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

    model_path = os.path.join(dataset_dir, "models", f"{part}_rsp.pt")
    torch.save(model.state_dict(), model_path)
    print(f"Saved {model_path}")
