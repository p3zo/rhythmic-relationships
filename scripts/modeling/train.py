import argparse
import torch
import yaml
from torch.utils.data import DataLoader, random_split

from model_utils import get_loss_fn, get_model_name, load_config, save_model
from rhythmic_relationships import DATASETS_DIR
from rhythmic_relationships.data import PartPairDatasetSequential
from rhythmic_relationships.model import TransformerEncoderDecoderNew
from rhythmic_relationships.train import train_transformer_encoder_decoder
from rhythmic_relationships.vocab import get_vocab_sizes

CONFIG_FILEPATH = "config.yml"

# DEVICE = torch.device("cpu")
DEVICE = torch.device(
    "mps"
    if torch.backends.mps.is_built()
    else torch.device("cuda:0")
    if torch.cuda.device_count() > 0
    else torch.device("cpu")
)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--datasets_dir",
        type=str,
        default=DATASETS_DIR,
        help="Path to the dir containing the dataset.",
    )
    args = parser.parse_args()

    datasets_dir = args.datasets_dir

    print(f"{DEVICE=}")

    config = load_config(CONFIG_FILEPATH)
    print(yaml.dump(config))

    torch.manual_seed(config["seed"])

    dataset = PartPairDatasetSequential(**config["data"], datasets_dir=datasets_dir)
    splits = config["splits"]
    train_data, val_data, test_data = random_split(dataset, list(splits.values()))
    print(f"{splits=}: {len(train_data)}, {len(val_data)}, {len(test_data)}")

    train_loader = DataLoader(train_data, batch_size=config["batch_size"], shuffle=True)
    val_loader = DataLoader(val_data, batch_size=config["batch_size"], shuffle=True)

    model_name = get_model_name()
    print(f"{model_name=}")

    vocab_sizes = get_vocab_sizes()
    config["model"]["src_vocab_size"] = vocab_sizes[config["data"]["part_1"]]
    config["model"]["tgt_vocab_size"] = vocab_sizes[config["data"]["part_2"]]

    # Add 1 to the context length to account for the start token
    config["model"]["context_len"] = config["sequence_len"] + 1
    model = TransformerEncoderDecoderNew(**config["model"]).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])
    loss_fn = get_loss_fn(config)

    epoch_evals = train_transformer_encoder_decoder(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        loss_fn=loss_fn,
        config=config,
        device=DEVICE,
        model_name=model_name,
    )

    # Save the stats for the last epoch
    stats = {
        "epoch_evals": epoch_evals,
        "n_params": sum(p.nelement() for p in model.parameters()),
    }

    save_model(model, config, model_name, stats, bento=config["bento"])