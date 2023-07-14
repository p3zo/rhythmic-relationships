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
    # parser.add_argument("--model", type=str, default="hits_encdec_polydrums")
    parser.add_argument("--model", type=str, default="hits_encdec")
    # parser.add_argument("--model", type=str, default="rsp_fc")
    parser.add_argument("--datasets_dir", type=str, default=DATASETS_DIR)
    parser.add_argument("--config_path", type=str, default=None)
    args = parser.parse_args()

    model_type = args.model
    print(f"{model_type=}")

    if not args.config_path:
        this_path = os.path.dirname(__file__)
        args.config_path = os.path.join(this_path, model_type, "config.yml")

    model_name = get_model_name()
    print(f"{model_name=}")

    model_dir = os.path.join(MODELS_DIR, model_type, model_name)
    if not os.path.isdir(model_dir):
        os.makedirs(model_dir)

    model_module = importlib.import_module(args.model)
    model_module.train(
        config=load_config(args.config_path),
        model_name=model_name,
        datasets_dir=args.datasets_dir,
        model_dir=model_dir,
    )
