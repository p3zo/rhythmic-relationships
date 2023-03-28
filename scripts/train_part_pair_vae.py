import os

import torch
import yaml
from model_utils import load_config, save_model
from rhythmic_relationships.data import PartPairDataset
from rhythmic_relationships.model import VariationalAutoEncoder
from rhythmic_relationships.train import train
from torch.utils.data import DataLoader

DEVICE = torch.device("mps" if torch.backends.mps.is_built() else "cpu")
MODELS_DIR = "../output/models"
CHECKPOINTS_DIR = "models/checkpoints"
CONFIG_FILEPATH = "model_config.yml"


def load_model(model_name):
    model_path = os.path.join(MODELS_DIR, f"{model_name}.pt")
    model_obj = torch.load(model_path)
    config = model_obj["config"]
    model = VariationalAutoEncoder(**config["model"])
    model.load_state_dict(state_dict=model_obj["state_dict"])
    return model, config


if __name__ == "__main__":
    config = load_config(CONFIG_FILEPATH)
    print(yaml.dump(config))

    data = PartPairDataset(**config["dataset"])
    loader = DataLoader(data, batch_size=config["batch_size"], shuffle=True)

    model = VariationalAutoEncoder(**config["model"]).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])

    reduction = config["loss_reduction"]
    if config["loss_fn"] == "bce":
        loss_fn = torch.nn.BCELoss(reduction=reduction)
    elif config["loss_fn"] == "cross-entropy":
        loss_fn = torch.nn.CrossEntropyLoss(reduction=reduction)
    elif config["loss_fn"] == "mse":
        loss_fn = torch.nn.MSELoss(reduction=reduction)
    else:
        raise ValueError(f"`{config['loss_fn']}` is not a valid loss function")

    train(
        model=model,
        loader=loader,
        optimizer=optimizer,
        loss_fn=loss_fn,
        config=config,
        device=DEVICE,
        checkpoints_dir=CHECKPOINTS_DIR,
    )

    save_model(model, config)
