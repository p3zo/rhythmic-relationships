"""
`train_part_pair_vae.py`: Train a conditional VAE over a PartPairDataset.
Update `model_config.yml` to specify a new training configuration.
Inference can be done in a general way with `inference_part_pair_vae.py` or with specific
representations with `inference_part_pair_vae_pattern_to_pattern.py` and `inference_hits_to_hits.py`.
"""
import os

import matplotlib.pyplot as plt
import torch
import yaml
from rhythmic_relationships import MODELS_DIR
from rhythmic_relationships.data import PartDataset
from rhythmic_relationships.model_utils import (
    get_loss_fn,
    get_model_name,
    load_config,
    save_checkpoint,
    save_model,
)
from rhythmic_relationships.models.vae import VAE
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

DEVICE = torch.device("mps" if torch.backends.mps.is_built() else "cpu")
CONFIG_FILEPATH = "part_vae_config.yml"


def compute_recon_loss(recons, x, mu, sigma, loss_fn):
    reconstruction_loss = loss_fn(recons, x)
    kld_loss = torch.mean(
        -0.5 * torch.sum(1 + sigma - mu**2 - sigma.exp(), dim=1), dim=0
    )
    return reconstruction_loss + kld_loss


def train(
    model,
    loader,
    optimizer,
    loss_fn,
    config,
    device,
    model_name,
):
    x_dim = config["model"]["x_dim"]
    y_dim = config["model"]["y_dim"]
    conditional = config["model"]["conditional"]
    clip_gradients = config["clip_gradients"]
    num_epochs = config["num_epochs"]

    model_dir = os.path.join(MODELS_DIR, model_name)
    if not os.path.isdir(model_dir):
        os.makedirs(model_dir)

    train_losses = []
    ud = []  # update:data ratio

    for epoch in range(num_epochs):
        batches = tqdm(loader)
        for batch in batches:
            # Forward pass
            if conditional:
                x, y = batch
                x, y = x.to(device).view(x.shape[0], x_dim), y.to(device).view(
                    y.shape[0], y_dim
                )
                x_binary = (x > 0).to(torch.float32)
                x_recon, mu, sigma = model(x_binary, y)
            else:
                x = batch
                x = x.to(device).view(x.shape[0], x_dim)
                x_binary = (x > 0).to(torch.float32)
                x_recon, mu, sigma = model(x_binary)

            # Compute reconstruction loss
            x_recon_binary = (x_recon > 0).to(torch.float32)
            onset_loss = compute_recon_loss(
                x_recon_binary, x_binary, mu, sigma, loss_fn
            )
            velocity_loss = compute_recon_loss(x_recon, x, mu, sigma, loss_fn)
            loss = onset_loss + velocity_loss
            train_losses.append(loss.item())

            # Backprop
            optimizer.zero_grad()
            loss.backward()

            if clip_gradients:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5)

            optimizer.step()

            batches.set_description(f"Epoch {epoch + 1}/{num_epochs}")
            batches.set_postfix({"loss": f"{loss.item():.4f}"})

            with torch.no_grad():
                ud.append(
                    [
                        ((config["lr"] * p.grad).std() / p.data.std()).log10().item()
                        for p in model.parameters()
                    ]
                )

        # Save plot of loss during training
        plt.plot(train_losses)
        loss_plot_path = os.path.join(model_dir, f"training_loss_{epoch}.png")
        plt.savefig(loss_plot_path)
        print(f"Saved {loss_plot_path}")
        plt.clf()

        # Save a checkpoint at the end of each epoch
        if config["save_checkpoints"]:
            save_checkpoint(
                model_dir=model_dir,
                epoch=epoch,
                model=model,
                optimizer=optimizer,
                loss=loss,
                config=config,
            )

    return loss.item()


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
