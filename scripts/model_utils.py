import datetime as dt
import glob
import os
import random
from collections import defaultdict

import pandas as pd
import torch
import yaml
from rhythmic_relationships import MODELS_DIR


def load_config(filepath):
    """Loads a model config and adds derived values"""
    with open(filepath, "r") as fh:
        config = yaml.safe_load(fh)

    config["lr"] = float(config["lr"])

    return config


def get_model_name():
    # a copy of /usr/share/dict/web2 from a macbook air (early 2014)
    with open("words") as words_file:
        words = words_file.read().split()

    word = random.choice(words).lower()

    today = dt.datetime.today()
    timestamp = today.strftime("%y%m%d%H%M")

    return f"{word}_{timestamp}"


def save_model(model, config, model_name, stats, bento=True):
    model_dir = os.path.join(MODELS_DIR, model_name)
    if not os.path.isdir(model_dir):
        os.makedirs(model_dir)
    model_path = os.path.join(model_dir, "model.pt")

    torch.save(
        {
            "name": model_name,
            "model_class": model.__class__.__name__,
            "config": config,
            "stats": stats,
            "state_dict": model.state_dict(),
        },
        model_path,
    )
    print(f"Saved {model_path}")

    if bento:
        from bentoml.pytorch import save_model as save_bento_model

        saved_model = save_bento_model(
            model_name,
            model,
            signatures={
                "encode": {
                    "batchable": True,
                },
                "decode": {
                    "batchable": True,
                },
            },
        )
        print(f"Bento model saved: {saved_model}")


def load_model(model_name, model_class):
    model_path = os.path.join(MODELS_DIR, model_name, "model.pt")
    model_obj = torch.load(model_path)
    config = model_obj["config"]
    stats = model_obj["stats"]
    model = model_class(**config["model"])
    model.load_state_dict(state_dict=model_obj["state_dict"])
    return model, config, stats


def get_model_catalog():
    model_files = glob.glob(os.path.join(MODELS_DIR, "*/*.pt"))
    model_files = [fp for fp in model_files if "exclude_from_catalog" not in fp]
    catalog = defaultdict(dict)
    for fp in model_files:
        model_obj = torch.load(fp)
        if "name" not in model_obj:
            continue

        catalog_info = {}
        catalog_info["model_class"] = model_obj.get("model_class")
        catalog_info["stats"] = model_obj.get("stats")
        catalog_info["config"] = model_obj.get("config")
        catalog[model_obj["name"]] = catalog_info

    return pd.DataFrame.from_dict(catalog, orient="index")


def get_loss_fn(config):
    reduction = config["loss_reduction"]

    if config["loss_fn"] == "bce-logits":
        return torch.nn.BCEWithLogitsLoss(reduction=reduction)
    elif config["loss_fn"] == "bce":
        return torch.nn.BCELoss(reduction=reduction)
    elif config["loss_fn"] == "cross-entropy":
        # TODO: get ignore_index programatically based on part
        return torch.nn.CrossEntropyLoss(reduction=reduction, ignore_index=1)
    elif config["loss_fn"] == "mse":
        return torch.nn.MSELoss(reduction=reduction)
    else:
        raise ValueError(f"`{config['loss_fn']}` is not a valid loss function")
