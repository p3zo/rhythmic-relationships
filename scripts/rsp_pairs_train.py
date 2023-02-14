import argparse
import os

import torch
from rhythmic_complements.data import DescriptorPairsDataset
from rhythmic_complements.model import VariationalAutoEncoder
from rhythmic_complements.train import train
from torch.utils.data import DataLoader

DATASET_DIR = "output/lmd_clean_1bar_24res"

DEVICE = torch.device("mps" if torch.backends.mps.is_built() else "cpu")
INPUT_DIM = 18
H_DIM = 6
Z_DIM = 3
NUM_EPOCHS = 10
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
        "--part_1",
        type=str,
        default="Drums",
        help="The first of two parts to load.",
    )
    parser.add_argument(
        "--part_2",
        type=str,
        default="Bass",
        help="The second of two parts to load.",
    )
    args = parser.parse_args()

    dataset_dir = args.dataset_dir
    part_1 = args.part_1
    part_2 = args.part_2

    desc_pairs_dir = os.path.join(dataset_dir, "descriptors_pairs")
    train_fp = os.path.join(desc_pairs_dir, f"{part_1}_{part_2}_train.csv")
    train_data = DescriptorPairsDataset(train_fp, part_1, part_2)
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)

    # TODO: what is the actual number of labels here?
    model = VariationalAutoEncoder(
        INPUT_DIM, H_DIM, Z_DIM, conditional=True, n_labels=INPUT_DIM
    ).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    train(
        model,
        train_loader,
        optimizer,
        INPUT_DIM,
        device=DEVICE,
        num_epochs=10,
        clip_gradients=True,
        conditional=True,
    )
    model_path = os.path.join(DATASET_DIR, "models", f"{part_1}_{part_2}_rsp_pair.pt")
    torch.save(model.state_dict(), model_path)
    print(f"Saved {model_path}")
