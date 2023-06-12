import datetime as dt
import glob
import os
import random
from collections import defaultdict

import pandas as pd
import torch
import yaml
from rhythmic_relationships import MODELS_DIR, CHECKPOINTS_DIRNAME


def get_causal_mask(sz, device, boolean=False):
    mask = (torch.triu(torch.ones(sz, sz, device=device)) == 1).transpose(0, 1)
    mask.requires_grad = False
    if boolean:
        return mask
    return (
        mask.float()
        .masked_fill(mask == 0, float("-inf"))
        .masked_fill(mask == 1, float(0.0))
    )


def load_config(filepath):
    """Loads a model config and adds derived values"""
    with open(filepath, "r") as fh:
        config = yaml.safe_load(fh)

    config["lr"] = float(config["lr"])

    return config


def get_model_name():
    # a copy of /usr/share/dict/web2 from a macbook air (early 2014)
    this_path = os.path.dirname(__file__)
    with open(os.path.join(this_path, "words")) as words_file:
        words = words_file.read().split()

    word = random.choice(words).lower()

    today = dt.datetime.today()
    timestamp = today.strftime("%y%m%d%H%M")

    return f"{word}_{timestamp}"


def get_model_n_params(model):
    return sum(p.nelement() for p in model.parameters())


def save_model(model_path, model, config, model_name, evals=[]):
    torch.save(
        {
            "name": model_name,
            "model_class": model.__class__.__name__,
            "config": config,
            "evals": evals,
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
        catalog_info["evals"] = model_obj.get("evals")
        catalog_info["config"] = model_obj.get("config")
        catalog[model_obj["name"]] = catalog_info

    return pd.DataFrame.from_dict(catalog, orient="index")


def save_checkpoint(
    model_dir,
    epoch,
    model,
    optimizer,
    loss,
    config,
    evals,
    delete_prev=True,
):
    checkpoints_dir = os.path.join(model_dir, CHECKPOINTS_DIRNAME)
    if not os.path.isdir(checkpoints_dir):
        os.makedirs(checkpoints_dir)

    checkpoint_path = os.path.join(checkpoints_dir, str(epoch))

    torch.save(
        {
            "model_class": model.__class__.__name__,
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": loss,
            "config": config,
            "evals": evals,
        },
        checkpoint_path,
    )
    print(f"Saved checkpoint: {checkpoint_path}")

    if delete_prev:
        prev_checkpoint_path = os.path.join(checkpoints_dir, str(epoch - 1))
        if os.path.isfile(prev_checkpoint_path):
            os.remove(prev_checkpoint_path)
            print(f"Deleted previous checkpoint: {prev_checkpoint_path}")


def get_loss_fn(config, pad_ix=-100):
    reduction = config["loss_reduction"]

    if config["loss_fn"] == "bce-logits":
        return torch.nn.BCEWithLogitsLoss(reduction=reduction)
    elif config["loss_fn"] == "bce":
        return torch.nn.BCELoss(reduction=reduction)
    elif config["loss_fn"] == "cross-entropy":
        return torch.nn.CrossEntropyLoss(reduction=reduction, ignore_index=pad_ix)
    elif config["loss_fn"] == "mse":
        return torch.nn.MSELoss(reduction=reduction)
    else:
        raise ValueError(f"`{config['loss_fn']}` is not a valid loss function")
