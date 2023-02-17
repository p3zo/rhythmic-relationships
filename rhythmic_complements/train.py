import datetime as dt
import os

import torch
from tqdm import tqdm


def train(
    model,
    train_loader,
    optimizer,
    config,
):
    x_dim = config["model"]["x_dim"]
    y_dim = config["model"]["y_dim"]
    conditional = config["model"]["conditional"]
    clip_gradients = config["clip_gradients"]
    device = config["device"]
    num_epochs = config["num_epochs"]

    for epoch in range(num_epochs):
        batches = tqdm(train_loader)
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
            loss = model.loss_function(x_reconstructed, x, mu, sigma)

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
            config["dataset"]["dataset_dir"], config["checkpoints_dir"]
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
