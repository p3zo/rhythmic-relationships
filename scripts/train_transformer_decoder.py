import torch
import yaml
from model_utils import get_loss_fn, get_model_name, load_config, save_model
from rhythmic_relationships.data import PartDatasetSequential
from rhythmic_relationships.model import TransformerDecoder
from rhythmic_relationships.train import train_transformer_decoder
from torch.utils.data import DataLoader, random_split

DEVICE = torch.device("mps" if torch.backends.mps.is_built() else "cpu")
CONFIG_FILEPATH = "transformer_decoder_config.yml"

if __name__ == "__main__":
    print(f"{DEVICE=}")

    config = load_config(CONFIG_FILEPATH)
    print(yaml.dump(config))

    torch.manual_seed(13)

    dataset = PartDatasetSequential(**config["dataset"])
    splits = [0.6, 0.3, 0.1]
    train_data, val_data, test_data = random_split(dataset, splits)
    print(f"{splits=}: {len(train_data)}, {len(val_data)}, {len(test_data)}")

    train_loader = DataLoader(train_data, batch_size=config["batch_size"], shuffle=True)
    val_loader = DataLoader(val_data, batch_size=config["batch_size"], shuffle=True)

    model_name = get_model_name()
    print(f"{model_name=}")

    config["model"]["context_len"] = config["dataset"]["context_len"]
    model = TransformerDecoder(**config["model"]).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])
    loss_fn = get_loss_fn(config)

    epoch_evals = train_transformer_decoder(
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
    print(stats)

    save_model(model, config, model_name, stats)
