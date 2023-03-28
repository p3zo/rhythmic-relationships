import datetime as dt
import os
import random
import yaml

import torch
from rhythmic_relationships.data import PartPairDataset
from rhythmic_relationships.model import VariationalAutoEncoder
from rhythmic_relationships.train import train
from torch.utils.data import DataLoader

DEVICE = torch.device("mps" if torch.backends.mps.is_built() else "cpu")
MODELS_DIR = "../output/models"
CHECKPOINTS_DIR = "models/checkpoints"
CONFIG_FILEPATH = "model_config.yml"


def load_config(filepath):
    """Loads a model config and adds derived values"""
    with open(filepath, "r") as fh:
        config = yaml.safe_load(fh)

    config["lr"] = float(config["lr"])

    return config


def get_model_name(config):
    # a copy of /usr/share/dict/web2 from a macbook air (early 2014)
    with open("words") as words_file:
        words = words_file.read().split()

    word = random.choice(words)

    today = dt.datetime.today()
    timestamp = today.strftime("%y%m%d%H%M%S")

    dc = config["dataset"]
    info_str = f"{dc['dataset_name']}_{dc['part_1']}_{dc['part_2']}_{dc['repr_1']}_{dc['repr_2']}"

    return f"{word}_{info_str}_{timestamp}"


def save_model(model, config):
    model_name = get_model_name(config)
    model_path = os.path.join(MODELS_DIR, f"{model_name}.pt")
    torch.save(
        {
            "state_dict": model.state_dict(),
            "config": config,
        },
        model_path,
    )
    print(f"Saved {model_path}")


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
