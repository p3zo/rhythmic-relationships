import argparse
import os
import importlib
import torch
from rhythmic_relationships import DATASETS_DIR, MODELS_DIR
from rhythmic_relationships.model_utils import get_model_name, load_config

DEVICE = torch.device(
    "mps"
    if torch.backends.mps.is_built()
    else torch.device("cuda:0")
    if torch.cuda.device_count() > 0
    else torch.device("cpu")
)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="encdec")
    parser.add_argument("--datasets_dir", type=str, default=DATASETS_DIR)
    parser.add_argument("--config_path", type=str, default=None)
    args = parser.parse_args()

    this_path = os.path.dirname(os.path.abspath(__file__))
    model_types = sorted(
        d
        for d in os.listdir(this_path)
        if os.path.isfile(os.path.join(this_path, d, "config.yml"))
    )

    model_type = args.model
    if model_type not in model_types:
        raise ValueError(f"`{model_type}` is not a model. Choose one of {model_types}")
    print(f"{model_type=}")

    if not args.config_path:
        args.config_path = os.path.join(this_path, model_type, "config.yml")

    model_name = get_model_name()
    print(f"{model_name=}")

    config = load_config(args.config_path)

    # Seeds the default generator, which `random_split` draws the data splits from
    torch.manual_seed(config["seed"])

    model_dir = os.path.join(MODELS_DIR, model_type, model_name)
    if model_type in ["encdec", "hits_encdec"]:
        model_dir = os.path.join(
            MODELS_DIR,
            model_type,
            f"{config['data']['part_1']}_{config['data']['part_2']}",
            model_name,
        )
    if not os.path.isdir(model_dir):
        os.makedirs(model_dir)

    model_module = importlib.import_module(model_type)
    if not callable(getattr(model_module, "train", None)):
        raise AttributeError(
            f"`{model_type}` has no train() entry point. Each model package needs an "
            "__init__.py exporting `train(config, model_name, datasets_dir, model_dir)`."
        )

    model_module.train(
        config=config,
        model_name=model_name,
        datasets_dir=args.datasets_dir,
        model_dir=model_dir,
    )
