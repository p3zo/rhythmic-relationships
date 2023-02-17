import os

import torch
from rhythmic_complements.data import PairDataset
from rhythmic_complements.model import VariationalAutoEncoder
from rhythmic_complements.train import train
from torch.utils.data import DataLoader

config = {
    "dataset": {
        "dataset_dir": "../output/lmd_clean_1bar_24res_3000",
        "part_1": "Bass",
        "part_2": "Drums",
        "repr_1": "pattern",
        "repr_2": "roll",
    },
    "model": {
        "x_dim": 16,
        "h_dim": 200,
        "z_dim": 20,
        "y_dim": 8448,
        "conditional": True,
    },
    "checkpoints_dir": "checkpoints",
    "models_dir": "models",
    "num_epochs": 3,
    "batch_size": 64,
    "lr": 1e-4,
    "device": torch.device("mps" if torch.backends.mps.is_built() else "cpu"),
}


def get_model_path(config):
    dc = config["dataset"]
    return os.path.join(
        dc["dataset_dir"],
        config["models_dir"],
        f"{dc['part_1']}_{dc['part_2']}_{dc['repr_1']}_{dc['repr_2']}.pt",
    )


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
        clip_gradients=True,
        conditional=True,
    )

    # Save the model
    models_dir = os.path.join(config["dataset"]["dataset_dir"], config["models_dir"])
    if not os.path.isdir(models_dir):
        os.makedirs(models_dir)
    model_path = get_model_path(config)
    torch.save(model.state_dict(), model_path)
    print(f"Saved {model_path}")
