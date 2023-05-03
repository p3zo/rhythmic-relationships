import torch
import yaml
from model_utils import get_model_name, load_config, save_model, get_loss_fn
from rhythmic_relationships.data import PartDatasetSequential
from rhythmic_relationships.model import RecurrentVAE
from rhythmic_relationships.train import train_recurrent
from torch.utils.data import DataLoader

DEVICE = torch.device("mps" if torch.backends.mps.is_built() else "cpu")
CONFIG_FILEPATH = "recurrent_part_vae_config.yml"


if __name__ == "__main__":
    config = load_config(CONFIG_FILEPATH)
    print(yaml.dump(config))

    dataset = PartDatasetSequential(**config["dataset"])
    loader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=True)

    config['model']['seq_dim'] = config['dataset']['context_len']
    model = RecurrentVAE(**config["model"]).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])
    loss_fn = get_loss_fn(config)

    model_name = get_model_name()
    print(f"{model_name=}")

    train_recurrent(
        model=model,
        loader=loader,
        optimizer=optimizer,
        loss_fn=loss_fn,
        config=config,
        device=DEVICE,
        model_name=model_name,
    )

    save_model(model, config, model_name)
