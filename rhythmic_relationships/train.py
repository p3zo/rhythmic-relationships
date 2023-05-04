import datetime as dt
import os
import wandb

import matplotlib.pyplot as plt
import torch
from tqdm import tqdm

from rhythmic_relationships import MODELS_DIR, CHECKPOINTS_DIRNAME

WANDB_PROJECT_NAME = "rhythmic-relationships"


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
    model_name,
    save_checkpoints=False,
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

            # Compute loss
            x_recon_binary = (x_recon > 0).to(torch.float32)
            onset_loss = compute_loss(x_recon_binary, x_binary, mu, sigma, loss_fn)
            velocity_loss = compute_loss(x_recon, x, mu, sigma, loss_fn)
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
        if save_checkpoints:
            checkpoints_dir = os.path.join(MODELS_DIR, CHECKPOINTS_DIRNAME)
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
                    model_name,
                    f"{epoch}_{dt.datetime.today().strftime('%y%m%d%H%M%S')}",
                ),
            )

    return loss.item()


def train_recurrent(
    model,
    train_loader,
    val_loader,
    optimizer,
    loss_fn,
    config,
    device,
    model_name,
):
    clip_gradients = config["clip_gradients"]
    num_epochs = config["num_epochs"]

    model_dir = os.path.join(MODELS_DIR, model_name)
    if not os.path.isdir(model_dir):
        os.makedirs(model_dir)

    if config["wandb"]:
        wandb.init(project=WANDB_PROJECT_NAME, config=config, name=model_name)

        if device.type == "mps":
            # wandb.watch uses an operator that is not currently supported on MPS backends
            # This env var allows it to fall back to run on the CPU
            # See https://github.com/pytorch/pytorch/pull/96652
            os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
        wandb.watch(model, log_freq=100)

    train_losses = []
    val_losses = []

    for epoch in range(num_epochs):
        batches = tqdm(train_loader)
        for batch in batches:
            # Forward pass
            x, _ = batch
            x = x.to(device)
            x_recon, mu, sigma = model(x)
            x_recon = x_recon.view(x.shape[0], x.shape[1], x.shape[2])

            # Compute loss
            loss = compute_loss(x_recon, x, mu, sigma, loss_fn)
            train_losses.append(loss.log10().item())

            # Backprop
            optimizer.zero_grad()
            loss.backward()

            if clip_gradients:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5)

            optimizer.step()

            batches.set_description(f"Epoch {epoch + 1}/{num_epochs}")
            batches.set_postfix({"loss": f"{loss.log10().item():.4f}"})

        vx, _ = next(iter(val_loader))
        vx = vx.to(device)
        with torch.no_grad():
            vx_recon, mu, sigma = model(vx)
            vx_recon = vx_recon.view(vx.shape[0], vx.shape[1], vx.shape[2])
            val_loss = compute_loss(vx_recon, vx, mu, sigma, loss_fn)

        val_losses.append(val_loss.log10().item())
        if config["wandb"]:
            wandb.log(
                {
                    "train_log_loss": loss.log10().item(),
                    "val_log_loss": val_loss.log10().item(),
                }
            )

        # Save plot of loss during training
        plt.plot(train_losses)
        plt.plot(val_losses)
        loss_plot_path = os.path.join(model_dir, f"log_loss_{epoch}.png")
        plt.savefig(loss_plot_path)
        print(f"Saved {loss_plot_path}")
        plt.clf()

    return loss.log10().item(), val_loss.log10().item()
