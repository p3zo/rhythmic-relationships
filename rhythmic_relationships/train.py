import os

import matplotlib.pyplot as plt
import pandas as pd
import torch
import wandb
from rhythmic_relationships import CHECKPOINTS_DIRNAME, MODELS_DIR
from rhythmic_relationships.data import get_roll_from_sequence
from rhythmic_relationships.vocab import get_vocab_encoder_decoder
from rhythmtoolbox import pianoroll2descriptors
from tqdm import tqdm

WANDB_PROJECT_NAME = "rhythmic-relationships"


def save_checkpoint(model_dir, epoch, model, optimizer, loss, config):
    checkpoints_dir = os.path.join(model_dir, CHECKPOINTS_DIRNAME)
    if not os.path.isdir(checkpoints_dir):
        os.makedirs(checkpoints_dir)

    checkpoint_path = os.path.join(checkpoints_dir, str(epoch))
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": loss,
            "config": config,
        },
        checkpoint_path,
    )
    print(f"Saved checkpoint to {checkpoint_path}")


def compute_loss(logits, y, loss_fn):
    B, T, C = logits.shape
    return loss_fn(logits.view(B * T, C), y.view(y.shape[0] * y.shape[1]))


def parse_sequential_batch(batch, device):
    xb, yb = batch
    x = xb.to(device).view(xb.shape[0] * xb.shape[1], xb.shape[2])
    y = yb.to(device).view(yb.shape[0] * yb.shape[1], yb.shape[2])
    return x, y


def evaluate_transformer_encdec(
    model,
    config,
    train_loader,
    val_loader,
    loss_fn,
    device,
):
    # TODO: remove this temp workaround for a NotImplementedError related to nested tensors when running on an mps device
    prev_device = device
    device = torch.device("cpu")
    model.to(device)

    with torch.no_grad():
        model.eval()

        evaluation = {}

        n_eval_iters = config["n_eval_iters"]
        print(f"Evaluating for {n_eval_iters} iters")

        eval_train_losses = torch.zeros(n_eval_iters)
        for k in range(n_eval_iters):
            srcs, tgts = parse_sequential_batch(next(iter(train_loader)), device)
            logits = model(srcs, tgts)
            loss = compute_loss(logits, tgts, loss_fn)
            eval_train_losses[k] = loss.item()

        generated_rolls = []
        descriptors = []
        eval_val_losses = torch.zeros(n_eval_iters)
        for k in range(n_eval_iters):
            srcs, tgts = parse_sequential_batch(next(iter(val_loader)), device)
            logits = model(srcs, tgts)
            loss = compute_loss(logits, tgts, loss_fn)
            eval_val_losses[k] = loss.item()

            # Generate new sequences using part_1s from the val set and just a start token for part_2
            encode, _ = get_vocab_encoder_decoder(config["data"]["part_2"])
            start_ix = encode(["start"])[0]
            idy = torch.full(
                (srcs.shape[0], 1), start_ix, dtype=torch.long, device=device
            )
            seqs = (
                model.generate(srcs, idy, max_new_tokens=config["sequence_len"])
                .detach()
                .cpu()
                .numpy()
            )
            for seq in seqs:
                roll = get_roll_from_sequence(seq, part=config["data"]["part_2"])
                generated_rolls.append(roll)
                # TODO: also save the src rolls

                # Compute sequence descriptors
                descs = pianoroll2descriptors(
                    roll,
                    config["resolution"],
                    drums=config["data"]["part_2"] == "Drums",
                )
                descriptors.append(descs)

        evaluation.update(
            {
                "train_loss": eval_train_losses.mean().item(),
                "val_loss": eval_val_losses.mean().item(),
            }
        )

        # Log eval losses
        print(f'{evaluation["train_loss"]=}, {evaluation["val_loss"]=}')
        if config["wandb"]:
            wandb.log(
                {
                    "train_loss": evaluation["train_loss"],
                    "val_loss": evaluation["val_loss"],
                }
            )

        # Save the descriptors along with a few samples
        evaluation["generated_descriptors"] = pd.DataFrame(descriptors).to_dict()
        evaluation["generated_samples"] = generated_rolls[:10]

        model.train()

    model.to(prev_device)

    return evaluation


def train_transformer_encoder_decoder(
    model,
    train_loader,
    val_loader,
    optimizer,
    loss_fn,
    config,
    device,
    model_name,
):
    num_epochs = config["num_epochs"]

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
            srcs, tgts = parse_sequential_batch(batch, device)

            logits = model(srcs, tgts)

            # Compute loss
            loss = compute_loss(logits, tgts, loss_fn)
            train_losses.append(loss.item())

            # Backprop
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            batches.set_description(f"Epoch {epoch}/{num_epochs}")
            batches.set_postfix({"loss": f"{loss.item():.4f}"})

            # Save loss plot after each batch
            plt.plot(train_losses)
            loss_plot_path = os.path.join(model_dir, f"loss_{epoch}.png")
            plt.tight_layout()
            plt.savefig(loss_plot_path)
            plt.clf()

        # Evaluate after each epoch
        epoch_eval = evaluate_transformer_encdec(
            model=model,
            config=config,
            train_loader=train_loader,
            val_loader=val_loader,
            loss_fn=loss_fn,
            device=device,
        )
        epoch_evals.append(epoch_eval)

        # Log eval losses locally
        e_ixs = range(epoch)
        eval_train_losses = [epoch_evals[i]["train_loss"] for i in e_ixs]
        eval_val_losses = [epoch_evals[i]["val_loss"] for i in e_ixs]
        marker = "o" if epoch == 1 else None
        plt.plot(e_ixs, eval_train_losses, label="train", c="blue", marker=marker)
        plt.plot(e_ixs, eval_val_losses, label="val", c="orange", marker=marker)
        eval_loss_plot_path = os.path.join(model_dir, f"eval_loss_{epoch}.png")
        plt.legend()
        plt.title(f"{model_name}")
        plt.tight_layout()
        plt.savefig(eval_loss_plot_path)
        print(f"Saved {eval_loss_plot_path}")
        plt.clf()

        # Save loss plot after each epoch
        plt.plot(train_losses)
        loss_plot_path = os.path.join(model_dir, f"loss_{epoch}.png")
        plt.tight_layout()
        plt.savefig(loss_plot_path)
        print(f"Saved {loss_plot_path}")
        plt.clf()

        if config["save_checkpoints"]:
            save_checkpoint(
                model_dir=model_dir,
                epoch=epoch,
                model=model,
                optimizer=optimizer,
                loss=loss,
                config=config,
            )

    return epoch_evals
