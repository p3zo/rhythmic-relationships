import argparse
import os
import torch
import yaml
import pandas as pd
from rhythmic_relationships.model_utils import (
    get_loss_fn,
    get_model_name,
    load_config,
    save_model,
    save_checkpoint,
)
from rhythmtoolbox import pianoroll2descriptors
from rhythmic_relationships import DATASETS_DIR, MODELS_DIR
from rhythmic_relationships.data import PartDatasetSequential, get_roll_from_sequence
from rhythmic_relationships.io import write_midi_from_roll
from rhythmic_relationships.models.decoder import TransformerDecoder
from rhythmic_relationships.vocab import START_IX, get_vocab_sizes
from torch.utils.data import DataLoader, random_split
import wandb
import matplotlib.pyplot as plt
from tqdm import tqdm

DEFAULT_CONFIG_FILEPATH = "config.yml"
WANDB_PROJECT_NAME = "rhythmic-relationships"

DEVICE = torch.device(
    "mps"
    if torch.backends.mps.is_built()
    else torch.device("cuda:0")
    if torch.cuda.device_count() > 0
    else torch.device("cpu")
)

if DEVICE.type == "mps":
    # Allows pytorch to fall back to the CPU for operators not supported on MPS
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"


def parse_sequential_batch(batch, device):
    xb, yb = batch
    x = xb.to(device).view(xb.shape[0] * xb.shape[1], xb.shape[2])
    y = yb.to(device).view(yb.shape[0] * yb.shape[1], yb.shape[2])
    return x, y


def compute_loss(logits, y, loss_fn):
    B, T, C = logits.shape
    return loss_fn(logits.view(B * T, C), y.view(y.shape[0] * y.shape[1]))


def evaluate_transformer_decoder(
    model,
    config,
    train_loader,
    val_loader,
    loss_fn,
    device,
    model_dir,
    epoch,
):
    eval_dir = os.path.join(model_dir, "eval", f"epoch_{epoch}")
    if not os.path.isdir(eval_dir):
        os.makedirs(eval_dir)

    model.eval()

    evaluation = {}

    part = config["dataset"]["part"]
    n_eval_iters = config["n_eval_iters"]
    n_ticks = config["sequence_len"]
    n_seqs = config["n_eval_seqs"]

    # Use the model to generate new sequences
    rolls = []
    descriptors = []

    idx = torch.full((n_seqs, 1), START_IX, dtype=torch.long, device=device)
    with torch.no_grad():
        seqs = model.generate(idx, max_new_tokens=n_ticks - 1)

    seqs = seqs.cpu().numpy()

    for ix, seq in enumerate(seqs):
        roll = get_roll_from_sequence(seq, part)
        rolls.append(roll)

        # Compute sequence descriptors
        descs = pianoroll2descriptors(
            roll,
            config["resolution"],
            drums=part == "Drums",
        )
        descriptors.append(descs)

        write_midi_from_roll(
            roll,
            outpath=os.path.join(eval_dir, f"{ix}_gen.mid"),
            part=part,
            binary=False,
            onset_roll=True,
        )

    print(f"Evaluating train loss for {n_eval_iters} iters")
    eval_train_losses = torch.zeros(n_eval_iters)
    for k in range(n_eval_iters):
        x, y = parse_sequential_batch(next(iter(train_loader)), device)
        with torch.no_grad():
            logits = model(x)
            loss = compute_loss(logits, y, loss_fn)
        eval_train_losses[k] = loss.item()

    print(f"Evaluating val loss for {n_eval_iters} iters")
    eval_val_losses = torch.zeros(n_eval_iters)
    for k in range(n_eval_iters):
        x, y = parse_sequential_batch(next(iter(val_loader)), device)
        with torch.no_grad():
            logits = model(x)
            loss = compute_loss(logits, y, loss_fn)
        eval_val_losses[k] = loss.item()

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

    model.train()

    return evaluation


def train_transformer_decoder(
    model,
    train_loader,
    val_loader,
    optimizer,
    loss_fn,
    config,
    device,
    model_name,
    model_dir,
):
    num_epochs = config["num_epochs"]

    if config["wandb"]:
        wandb.init(project=WANDB_PROJECT_NAME, config=config, name=model_name)

    train_losses = []
    evals = []

    for epoch in range(1, num_epochs + 1):
        batches = tqdm(train_loader)
        for batch in batches:
            # Forward pass
            x, y = parse_sequential_batch(batch, device)
            logits = model(x)

            # Compute loss
            loss = compute_loss(logits, y, loss_fn)
            train_losses.append(loss.item())

            # Backprop
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            batches.set_description(f"Epoch {epoch}/{num_epochs}")
            batches.set_postfix({"loss": f"{loss.item():.4f}"})

        # Evaluate after each epoch
        eval = evaluate_transformer_decoder(
            model=model,
            config=config,
            train_loader=train_loader,
            val_loader=val_loader,
            loss_fn=loss_fn,
            device=device,
            model_dir=model_dir,
            epoch=epoch,
        )
        evals.append(eval)

        # Log eval losses locally
        e_ixs = range(epoch)
        eval_train_losses = [evals[i]["train_loss"] for i in e_ixs]
        eval_val_losses = [evals[i]["val_loss"] for i in e_ixs]
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
                loss=loss.item(),
                config=config,
                evals=evals,
            )

    return evals


def train(config, model_name, datasets_dir, model_dir):
    dataset = PartDatasetSequential(**config["dataset"], datasets_dir=datasets_dir)

    splits = config["splits"]
    train_data, val_data, test_data = random_split(dataset, list(splits.values()))
    print(f"{splits=}: {len(train_data)}, {len(val_data)}, {len(test_data)}")

    train_loader = DataLoader(train_data, batch_size=config["batch_size"], shuffle=True)
    val_loader = DataLoader(val_data, batch_size=config["batch_size"], shuffle=True)

    config["model"]["vocab_size"] = get_vocab_sizes()[config["dataset"]["part"]]
    config["model"]["context_len"] = config["sequence_len"]

    model = TransformerDecoder(**config["model"]).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])
    loss_fn = get_loss_fn(config)

    print(yaml.dump(config))

    evals = train_transformer_decoder(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
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
        evals=evals,
    )
