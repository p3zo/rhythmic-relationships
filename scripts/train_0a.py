import os

import torch
from rhythmic_complements.data import PairDataset
from rhythmic_complements.model import VariationalAutoEncoder
from rhythmic_complements.train import train
from torch.utils.data import DataLoader

config = {
    "dataset": {
        "dataset_dir": "../output/lmd_clean_1bar_24res_1000",
        "part_1": "Bass",
        "part_2": "Drums",
        "repr_1": "roll",
        "repr_2": "pattern",
    },
    "model": {
        "x_dim": 8448,
        "h_dim": 200,
        "z_dim": 20,
        "y_dim": 16,
        "conditional": True,
    },
    "clip_gradients": True,
    "model_dir": "models",
    "checkpoints_dir": "models/checkpoints",
    "num_epochs": 3,
    "batch_size": 64,
    "lr": 1e-4,
    "device": torch.device("mps" if torch.backends.mps.is_built() else "cpu"),
}


def get_model_name(config):
    dc = config["dataset"]
    return f"{dc['part_1']}_{dc['part_2']}_{dc['repr_1']}_{dc['repr_2']}"


def get_model_dir(config):
    return os.path.join(config["dataset"]["dataset_dir"], config["model_dir"])


def get_model_path(config):
    return os.path.join(get_model_dir(config), f"{get_model_name(config)}.pt")


if __name__ == "__main__":
    print(f"Config: {config}")

    data = PairDataset(**config["dataset"])
    loader = DataLoader(data, batch_size=config["batch_size"], shuffle=True)

    model = VariationalAutoEncoder(**config["model"]).to(config["device"])
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])

    train(
        model,
        loader,
        optimizer,
        config,
    )

    # Save the model
    model_dir = get_model_dir(config)
    if not os.path.isdir(model_dir):
        os.makedirs(model_dir)
    model_path = get_model_path(config)
    torch.save(model.state_dict(), model_path)
    print(f"Saved {model_path}")
