import torch
import yaml
from torch.utils.data import DataLoader, random_split

from model_utils import get_model_name, load_config, save_model, get_loss_fn
from rhythmic_relationships.data import PartDatasetSequential
from rhythmic_relationships.model import TransformerDecoder
from rhythmic_relationships.train import train_transformer_decoder


DEVICE = torch.device("mps" if torch.backends.mps.is_built() else "cpu")
CONFIG_FILEPATH = "transformer_decoder_config.yml"

if __name__ == "__main__":
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
    # model = BigramDecoder(vocab_size=vocab_size).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])
    loss_fn = get_loss_fn(config)

    evaluation = train_transformer_decoder(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        loss_fn=loss_fn,
        config=config,
        device=DEVICE,
        model_name=model_name,
    )

    stats = {
        "train_loss": evaluation["train_loss"],
        "val_loss": evaluation["val_loss"],
        "n_params": sum(p.nelement() for p in model.parameters()),
    }
    print(stats)

    save_model(model, config, model_name, stats)