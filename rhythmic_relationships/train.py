import datetime as dt
import os

import torch
from tqdm import tqdm

from rhythmic_relationships import DATASETS_DIR, MODELS_DIRNAME, CHECKPOINTS_DIRNAME


def compute_loss(recons, x, mu, sigma, loss_fn):
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
):
    x_dim = config["model"]["x_dim"]
    y_dim = config["model"]["y_dim"]
    conditional = config["model"]["conditional"]
    clip_gradients = config["clip_gradients"]
    num_epochs = config["num_epochs"]

    training_losses = []

    for epoch in range(num_epochs):
        batches = tqdm(loader)
        for batch in batches:
            # Forward pass
            if conditional:
                x, y = batch
                x, y = x.to(device).view(x.shape[0], x_dim), y.to(device).view(
                    y.shape[0], y_dim
                )
                x_reconstructed, mu, sigma = model(x, y)
            else:
                x = batch
                x = x.to(device).view(x.shape[0], x_dim)
                x_reconstructed, mu, sigma = model(x)

            # Compute loss
            loss = compute_loss(x_reconstructed, x, mu, sigma, loss_fn)
            training_losses.append(loss.item())

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()

            if clip_gradients:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5)

            optimizer.step()

            batches.set_description(f"Epoch {epoch + 1}/{num_epochs}")
            batches.set_postfix({"loss": loss.item()})

        # Save a checkpoint at the end of each epoch
        checkpoints_dir = os.path.join(
            DATASETS_DIR, MODELS_DIRNAME, CHECKPOINTS_DIRNAME
        )
        if not os.path.isdir(checkpoints_dir):
            os.makedirs(checkpoints_dir)

        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": loss,
                "config": config,
            },
            os.path.join(
                checkpoints_dir,
                f"checkpoint_{epoch}_{dt.datetime.today().strftime('%y%m%d%H%M%S')}",
            ),
        )

        import matplotlib.pyplot as plt

        plt.plot(training_losses)
        plt.savefig("training_loss.png")
