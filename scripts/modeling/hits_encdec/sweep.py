import os
import torch
from rhythmic_relationships import DATASETS_DIR, MODELS_DIR
from rhythmic_relationships.model_utils import get_model_name, load_config
import wandb
import yaml

from train import WANDB_PROJECT_NAME, train

import matplotlib.pyplot as plt
plt.switch_backend('Agg')

SWEEP_CONFIG_FILEPATH = "sweep.yaml"

DEVICE = torch.device(
    "mps"
    if torch.backends.mps.is_built()
    else torch.device("cuda:0")
    if torch.cuda.device_count() > 0
    else torch.device("cpu")
)


def main():
    this_path = os.path.dirname(__file__)
    config_path = os.path.join(this_path, "config.yml")

    model_name = get_model_name()
    print(f"{model_name=}")

    model_dir = os.path.join(MODELS_DIR, model_name)
    if not os.path.isdir(model_dir):
        os.makedirs(model_dir)

    train(
        config=load_config(config_path),
        model_name=model_name,
        datasets_dir=DATASETS_DIR,
        model_dir=model_dir,
        sweep=True,
    )


if __name__ == "__main__":
    with open(SWEEP_CONFIG_FILEPATH, "r") as fh:
        sweep_config = yaml.safe_load(fh)
    print(yaml.dump(sweep_config))

    sweep_id = wandb.sweep(sweep=sweep_config, project=WANDB_PROJECT_NAME)
    wandb.agent(sweep_id, function=main, count=25)
