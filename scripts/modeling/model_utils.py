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


def get_model_n_params(model):
    return sum(p.nelement() for p in model.parameters())


def save_model(model_path, model, config, model_name, epoch_evals=[]):
    torch.save(
        {
            "name": model_name,
            "model_class": model.__class__.__name__,
            "config": config,
            "epoch_evals": epoch_evals,
            "n_params": get_model_n_params(model),
            "model_state_dict": model.state_dict(),
        },
        model_path,
    )
    print(f"Saved {model_path}")


def save_bento_model(model, model_name):
    from bentoml.pytorch import save_model as save_bento_model

    saved_model = save_bento_model(
        model_name,
        model,
        signatures={"encode": {"batchable": True}, "decode": {"batchable": True}},
    )
    print(f"Bento model saved: {saved_model}")


def load_model(model_path, model_class):
    model_obj = torch.load(model_path, map_location=torch.device("cpu"))
    config = model_obj["config"]
    model = model_class(**config["model"])
    model.load_state_dict(state_dict=model_obj["model_state_dict"])
    return model, config


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
        catalog_info["n_params"] = model_obj.get("n_params")
        catalog_info["epoch_evals"] = model_obj.get("epoch_evals")
        catalog_info["config"] = model_obj.get("config")
        catalog[model_obj["name"]] = catalog_info

    return pd.DataFrame.from_dict(catalog, orient="index")
