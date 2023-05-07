import datetime as dt
import os
import wandb

import matplotlib.pyplot as plt
import pandas as pd
import torch
from tqdm import tqdm
from rhythmtoolbox import pianoroll2descriptors

from rhythmic_relationships import MODELS_DIR, CHECKPOINTS_DIRNAME
from rhythmic_relationships.io import write_midi_from_roll, get_roll_from_sequence

WANDB_PROJECT_NAME = "rhythmic-relationships"


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
            xb, yb = batch
            x = xb.to(device).view(xb.shape[0] * xb.shape[1], xb.shape[2])
            y = yb.to(device).view(yb.shape[0] * yb.shape[1])
            x_recon, mu, sigma = model(x)
            x_recon = x_recon.view(x.shape[0], x.shape[1])

            # Compute loss
            loss = compute_recon_loss(x_recon, x, mu, sigma, loss_fn)
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
            val_loss = compute_recon_loss(vx_recon, vx, mu, sigma, loss_fn)

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


def compute_loss(logits, targets, loss_fn):
    B, T, C = logits.shape
    return loss_fn(logits.view(B * T, C), targets)


def parse_batch(batch, device):
    xb, yb = batch
    x = xb.to(device).view(xb.shape[0] * xb.shape[1], xb.shape[2])
    targets = yb.to(device).view(yb.shape[0] * yb.shape[1] * yb.shape[2])
    return x, targets


def train_transformer_decoder(
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
    n_eval_iters = config["n_eval_iters"]
    part = config["dataset"]["part"]

    model_dir = os.path.join(MODELS_DIR, model_name)
    if not os.path.isdir(model_dir):
        os.makedirs(model_dir)

    if config["wandb"]:
        wandb.init(project=WANDB_PROJECT_NAME, config=config, name=model_name)

    train_losses = []
    epoch_evals = []

    for epoch in range(1, num_epochs + 1):
        batches = tqdm(train_loader)
        for batch in batches:
            # Forward pass
            x, targets = parse_batch(batch, device)
            logits = model(x)

            # Compute loss
            loss = compute_loss(logits, targets, loss_fn)
            train_losses.append(loss.item())

            # Backprop
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if clip_gradients:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()

            batches.set_description(f"Epoch {epoch}/{num_epochs}")
            batches.set_postfix({"loss": f"{loss.item():.4f}"})

        # Evaluate after each epoch
        print("Evaluating...")
        with torch.no_grad():
            model.eval()

            epoch_eval = {}

            # Use the model to generate new sequences
            rolls = []
            n_ticks = config["sequence_len"]
            n_seqs = config["n_eval_seqs"]
            print(f"Generating {n_seqs} {n_ticks}-tick sequences")
            for _ in range(n_seqs):
                idx = torch.zeros((1, 1), dtype=torch.long, device=device)
                seq = model.generate(idx, max_new_tokens=n_ticks - 1)[0]
                rolls.append(get_roll_from_sequence(seq))

            # Compute sequence descriptors
            descriptors = []
            for roll in rolls:
                descs = pianoroll2descriptors(
                    roll,
                    config["resolution"],
                    drums=part == "Drums",
                )
                descriptors.append(descs)

            desc_df = pd.DataFrame(descriptors)

            # TODO: Compare the distribution of descriptors against the training data
            # Save the descriptor means
            epoch_eval["generation_descriptor_means"] = desc_df.mean().to_dict()

            # Save one generated sequence as MIDI
            write_midi_from_roll(
                rolls[0],
                outpath=os.path.join(model_dir, f"generated_{epoch}.mid"),
                part=part,
                binary=True,
                onset_roll=True,
            )

            print(f"Running for {n_eval_iters} iters")
            eval_train_losses = torch.zeros(n_eval_iters)
            for k in tqdm(range(n_eval_iters)):
                x, targets = parse_batch(next(iter(train_loader)), device)
                logits = model(x)
                loss = compute_loss(logits, targets, loss_fn)
                eval_train_losses[k] = loss.item()

            eval_val_losses = torch.zeros(n_eval_iters)
            for k in tqdm(range(n_eval_iters)):
                x, targets = parse_batch(next(iter(val_loader)), device)
                logits = model(x)
                loss = compute_loss(logits, targets, loss_fn)
                eval_val_losses[k] = loss.item()

            epoch_eval.update(
                {
                    "train_loss": eval_train_losses.mean().item(),
                    "val_loss": eval_val_losses.mean().item(),
                }
            )
            epoch_evals.append(epoch_eval)

            # Log eval losses locally
            print(f"{epoch=}: {epoch_eval=}")
            e_ixs = range(epoch + 1)
            eval_train_losses = [epoch_evals[i]["train_loss"] for i in e_ixs]
            eval_val_losses = [epoch_evals[i]["val_loss"] for i in e_ixs]
            marker = "o" if epoch == 0 else None
            plt.plot(e_ixs, eval_train_losses, label="train", c="blue", marker=marker)
            plt.plot(e_ixs, eval_val_losses, label="val", c="orange", marker=marker)
            eval_loss_plot_path = os.path.join(model_dir, f"eval_loss_{epoch}.png")
            plt.legend()
            plt.title(f"{model_name}")
            plt.tight_layout()
            plt.savefig(eval_loss_plot_path)
            print(f"Saved {eval_loss_plot_path}")
            plt.clf()

            # Log eval losses remotely
            if config["wandb"]:
                wandb.log(
                    {
                        "train_loss": epoch_eval["train_loss"],
                        "val_loss": epoch_eval["val_loss"],
                    }
                )

            model.train()

        # Save plot of loss during training
        plt.plot(train_losses)
        loss_plot_path = os.path.join(model_dir, f"loss_{epoch}.png")
        plt.tight_layout()
        plt.savefig(loss_plot_path)
        print(f"Saved {loss_plot_path}")
        plt.clf()

    return epoch_evals
