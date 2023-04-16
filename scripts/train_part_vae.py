import torch
import yaml
from model_utils import load_config, save_model
from rhythmic_relationships.data import PartDataset
from rhythmic_relationships.model import VariationalAutoEncoder
from rhythmic_relationships.train import train
from torch.utils.data import DataLoader


DEVICE = torch.device("mps" if torch.backends.mps.is_built() else "cpu")
CONFIG_FILEPATH = "part_vae_config.yml"


if __name__ == "__main__":
    config = load_config(CONFIG_FILEPATH)
    print(yaml.dump(config))

    dataset = PartDataset(**config["dataset"])
    loader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=True)

    model = VariationalAutoEncoder(**config["model"]).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])

    reduction = config["loss_reduction"]
    if config["loss_fn"] == "bce":
        loss_fn = torch.nn.BCELoss(reduction=reduction)
    elif config["loss_fn"] == "cross-entropy":
        loss_fn = torch.nn.CrossEntropyLoss(reduction=reduction)
    elif config["loss_fn"] == "mse":
        loss_fn = torch.nn.MSELoss(reduction=reduction)
    else:
        raise ValueError(f"`{config['loss_fn']}` is not a valid loss function")

    train(
        model=model,
        loader=loader,
        optimizer=optimizer,
        loss_fn=loss_fn,
        config=config,
        device=DEVICE,
    )

    save_model(model, config)
