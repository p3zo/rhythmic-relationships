import os

import matplotlib.pyplot as plt
import torch
import yaml
from rhythmic_relationships.data import PartPairDataset
from rhythmic_relationships.model_utils import (
    get_loss_fn,
    get_model_name,
    load_config,
    save_checkpoint,
    save_model,
)
from rhythmic_relationships.models.vae import VAE
from torch.utils.data import DataLoader
from tqdm import tqdm

DEVICE = torch.device("mps" if torch.backends.mps.is_built() else "cpu")


def compute_kld_loss(mu, logvar):
    return torch.mean(
        -0.5 * torch.sum(1 + logvar - mu**2 - logvar.exp(), dim=1), dim=0
    )


def train_part_pair_vae(
    model,
    loader,
    optimizer,
    loss_fn,
    config,
    device,
    model_name,
    model_dir,
):
    x_dim = config["model"]["x_dim"]
    y_dim = config["model"]["y_dim"]
    conditional = config["model"]["conditional"]
    clip_gradients = config["clip_gradients"]
    num_epochs = config["num_epochs"]

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
                x_recon, mu, logvar = model(x_binary, y)
            else:
                x = batch
                x = x.to(device).view(x.shape[0], x_dim)
                x_binary = (x > 0).to(torch.float32)
                x_recon, mu, logvar = model(x_binary)

            # Compute reconstruction loss against both the onsets and the velocities. Both
            # read the raw logits: thresholding the reconstruction first is not differentiable,
            # so the onset term contributed no gradient at all.
            onset_loss = loss_fn(x_recon, x_binary)
            velocity_loss = loss_fn(x_recon, x)

            # Added once, not once per reconstruction term
            loss = onset_loss + velocity_loss + compute_kld_loss(mu, logvar)
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
                loss=loss.item(),
                config=config,
            )

    return loss.item()


def train(config, model_name, datasets_dir, model_dir):
    # These scripts want raw rolls, not the token sequences the transformers consume
    dataset = PartPairDataset(
        **config["dataset"],
        datasets_dir=datasets_dir,
        tokenize_rolls=False,
    )
    loader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=True)

    model = VAE(**config["model"]).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])
    loss_fn = get_loss_fn(config)

    print(yaml.dump(config))

    train_loss = train_part_pair_vae(
        model=model,
        loader=loader,
        optimizer=optimizer,
        loss_fn=loss_fn,
        config=config,
        device=DEVICE,
        model_name=model_name,
        model_dir=model_dir,
    )

    save_model(
        model_path=os.path.join(model_dir, "model.pt"),
        model=model,
        config=config,
        model_name=model_name,
        evals=[{"train_loss": train_loss}],
    )
