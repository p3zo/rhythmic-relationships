import argparse
import importlib
import inspect
import os
import torch
from rhythmic_relationships import DATASETS_DIR, MODELS_DIR
from rhythmic_relationships.model_utils import (
    get_model_name,
    load_checkpoint,
    load_config,
)
from rhythmic_relationships.parts import PARTS

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
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Continue a run from one of its checkpoints, e.g. .../checkpoints/5. The run "
        "carries on in the same directory, with the same data splits and optimiser state.",
    )
    parser.add_argument(
        "--part_1",
        type=str,
        default=None,
        help="Override the config's input part. Twelve directed pairs off one config beats "
        "twelve near-identical config files.",
    )
    parser.add_argument(
        "--part_2",
        type=str,
        default=None,
        help="Override the config's output part.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Total epochs to train to, overriding the config. To add three epochs to a run "
        "that stopped at five, pass 8.",
    )
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

    checkpoint = None
    if args.resume:
        if args.part_1 or args.part_2:
            raise ValueError("A resumed run keeps its own parts; drop --part_1 and --part_2")
        # The checkpoint's own config is what built the model being loaded, so it wins over
        # anything on disk that may have been edited since
        checkpoint = load_checkpoint(args.resume)
        config = checkpoint["config"]
        model_dir = os.path.dirname(os.path.dirname(os.path.abspath(args.resume)))
        model_name = os.path.basename(model_dir)
        print(f"{model_name=} resuming from {args.resume}")
    else:
        config = load_config(args.config_path)
        for key, part in (("part_1", args.part_1), ("part_2", args.part_2)):
            if part:
                if part not in PARTS:
                    raise ValueError(f"`{part}` is not a part. Choose one of {PARTS}")
                config["data"][key] = part
        if config["data"]["part_1"] == config["data"]["part_2"]:
            raise ValueError("A model needs two different parts")
        model_name = get_model_name()
        print(f"{model_name=}")
        model_dir = os.path.join(MODELS_DIR, model_type, model_name)
        if model_type in ["encdec", "hits_encdec"]:
            model_dir = os.path.join(
                MODELS_DIR,
                model_type,
                f"{config['data']['part_1']}_{config['data']['part_2']}",
                model_name,
            )

    if args.epochs:
        config["n_epochs"] = args.epochs

    # Seeds the default generator, which `random_split` draws the data splits from
    torch.manual_seed(config["seed"])

    if not os.path.isdir(model_dir):
        os.makedirs(model_dir)

    model_module = importlib.import_module(model_type)
    if not callable(getattr(model_module, "train", None)):
        raise AttributeError(
            f"`{model_type}` has no train() entry point. Each model package needs an "
            "__init__.py exporting `train(config, model_name, datasets_dir, model_dir)`."
        )

    train_kwargs = dict(
        config=config,
        model_name=model_name,
        datasets_dir=args.datasets_dir,
        model_dir=model_dir,
    )
    if checkpoint is not None:
        if "resume" not in inspect.signature(model_module.train).parameters:
            raise ValueError(f"`{model_type}`'s train() cannot resume from a checkpoint yet")
        train_kwargs["resume"] = checkpoint

    model_module.train(**train_kwargs)
