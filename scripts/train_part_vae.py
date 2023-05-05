import torch
import yaml
from model_utils import get_model_name, load_config, save_model, get_loss_fn
from rhythmic_relationships.data import PartDataset
from rhythmic_relationships.model import VAE
from rhythmic_relationships.train import train, compute_recon_loss
from torch.utils.data import DataLoader, random_split

DEVICE = torch.device("mps" if torch.backends.mps.is_built() else "cpu")
CONFIG_FILEPATH = "part_vae_config.yml"


if __name__ == "__main__":
    config = load_config(CONFIG_FILEPATH)
    print(yaml.dump(config))

    torch.manual_seed(13)

    dataset = PartDataset(**config["dataset"])
    splits = [0.6, 0.3, 0.1]
    train_data, val_data, test_data = random_split(dataset, splits)
    print(f"{splits=}: {len(train_data)}, {len(val_data)}, {len(test_data)}")

    train_loader = DataLoader(train_data, batch_size=config["batch_size"], shuffle=True)

    model = VAE(**config["model"]).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])
    loss_fn = get_loss_fn(config)

    model_name = get_model_name()
    print(f"{model_name=}")

    train_loss = train(
        model=model,
        loader=train_loader,
        optimizer=optimizer,
        loss_fn=loss_fn,
        config=config,
        device=DEVICE,
        model_name=model_name,
    )

    print("Evaluating validation loss...")
    val_loader = DataLoader(val_data, batch_size=len(val_data), shuffle=True)
    x = next(iter(val_loader))
    x = x.to(DEVICE).view(x.shape[0], config["model"]["x_dim"])
    with torch.no_grad():
        x_recon, mu, sigma = model(x)
        val_loss = compute_recon_loss(x_recon, x, mu, sigma, loss_fn).item()

    stats = {
        "train_loss": train_loss,
        "val_loss": val_loss,
        "n_params": sum(p.nelement() for p in model.parameters()),
    }
    print(stats)

    save_model(model, config, model_name, stats)
