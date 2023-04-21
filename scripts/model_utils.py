import datetime as dt
import os
import random

from bentoml.pytorch import save_model as save_bento_model
import torch
import yaml
from rhythmic_relationships import MODELS_DIR, INFERENCE_DIR

import pandas as pd
import seaborn as sns
from sklearn.manifold import TSNE
from utils import save_fig

sns.set_style("white")
sns.set_context("paper")


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

    if paired:
        info_str = f"{dc['dataset_name']}_{dc['part_1']}_{dc['part_2']}_{dc['repr_1']}_{dc['repr_2']}"
    else:
        info_str = f"{dc['dataset_name']}_{dc['part']}_{dc['representation']}"

    return f"{word}_{info_str}_{timestamp}"


def save_model(model, config, model_name, bento=True):
    model_dir = os.path.join(MODELS_DIR, model_name)
    if not os.path.isdir(model_dir):
        os.makedirs(model_dir)
    model_path = os.path.join(model_dir, "model.pt")
    torch.save(
        {
            "state_dict": model.state_dict(),
            "config": config,
        },
        model_path,
    )
    print(f"Saved {model_path}")
    config_path = os.path.join(MODELS_DIR, model_name, f"config.yaml")
    with open(config_path, "w") as file:
        yaml.dump(config, file)

    if bento:
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
    model_path = os.path.join(MODELS_DIR, model_name, f"model.pt")
    model_obj = torch.load(model_path)
    config = model_obj["config"]
    model = model_class(**config["model"])
    model.load_state_dict(state_dict=model_obj["state_dict"])
    return model, config


def get_embeddings(X, title="", outdir="."):
    """Create a 2D embedding space of the data, plot it, and save it to a csv"""
    reducer = TSNE(n_components=2, init="pca", learning_rate="auto", random_state=42)

    # Make the pair space using t-SNE
    X_transform = reducer.fit_transform(X)

    emb = pd.DataFrame(X_transform, columns=["x", "y"])
    sns.relplot(
        data=emb,
        x="x",
        y="y",
        height=8,
        aspect=1.25,
        legend=False,
    )

    save_fig(os.path.join(outdir, "latent_samples.png"), title=title)
    return emb
