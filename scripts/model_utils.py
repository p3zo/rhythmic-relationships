import datetime as dt
import os
import random

import torch
import yaml

MODELS_DIR = "../output/models"


def load_config(filepath):
    """Loads a model config and adds derived values"""
    with open(filepath, "r") as fh:
        config = yaml.safe_load(fh)

    config["lr"] = float(config["lr"])

    return config


def get_model_name(config, paired=True):
    # a copy of /usr/share/dict/web2 from a macbook air (early 2014)
    with open("words") as words_file:
        words = words_file.read().split()

    word = random.choice(words)

    today = dt.datetime.today()
    timestamp = today.strftime("%y%m%d%H%M%S")

    dc = config["dataset"]
    info_str = ""
    if paired:
        info_str = f"{dc['dataset_name']}_{dc['part_1']}_{dc['part_2']}_{dc['repr_1']}_{dc['repr_2']}"
    else:
        info_str = f"{dc['dataset_name']}_{dc['part']}_{dc['representation']}"

    return f"{word}_{info_str}_{timestamp}"


def save_model(model, config):
    paired = "part_1" in config["dataset"]
    model_name = get_model_name(config, paired)
    model_path = os.path.join(MODELS_DIR, f"{model_name}.pt")
    torch.save(
        {
            "state_dict": model.state_dict(),
            "config": config,
        },
        model_path,
    )
    print(f"Saved {model_path}")


def load_model(model_name, model_class):
    model_path = os.path.join(MODELS_DIR, f"{model_name}.pt")
    model_obj = torch.load(model_path)
    config = model_obj["config"]
    model = model_class(**config["model"])
    model.load_state_dict(state_dict=model_obj["state_dict"])
    return model, config
