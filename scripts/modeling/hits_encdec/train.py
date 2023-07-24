import argparse
import os

import matplotlib.pyplot as plt
import pandas as pd
import torch
import wandb
import yaml
from eval import evaluate_hits_encdec
from rhythmic_relationships import DATASETS_DIR, MODELS_DIR, WANDB_PROJECT_NAME
from rhythmic_relationships.data import PartPairDataset
from rhythmic_relationships.model_utils import (
    get_loss_fn,
    get_model_name,
    load_config,
    save_checkpoint,
    save_model,
)
from rhythmic_relationships.models.hits_encdec import TransformerEncoderDecoder
from rhythmic_relationships.vocab import get_hits_vocab_size
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

DEFAULT_CONFIG_FILEPATH = "config.yml"

DEVICE = torch.device(
    "mps"
    if torch.backends.mps.is_built()
    else torch.device("cuda:0")
    if torch.cuda.device_count() > 0
    else torch.device("cpu")
)


def parse_batch(batch, device):
    xb, yb = batch
    yb_shifted = torch.roll(yb, 1)
    yb_shifted[:, 0] = torch.zeros((yb.shape[0],))
    return xb.to(device), yb_shifted.to(device), yb.to(device)


def compute_loss(logits, y, loss_fn):
    B, T, C = logits.shape
    return loss_fn(logits.view(B * T, C), y.view(y.shape[0] * y.shape[1]))


def train_hits_encdec(
    model,
    train_loader,
    val_loader,
    optimizer,
    loss_fn,
    config,
    model_name,
    model_dir,
    device,
):
    n_epochs = config["n_epochs"]
    eval_interval = config["eval_interval"]

    evals = []
    train_losses = []

    model.train()

    ix = 0
    for epoch in range(1, n_epochs + 1):
        batches = tqdm(train_loader)
        for batch in batches:
            src, ctx, tgt = parse_batch(batch, device)
            logits = model(src, ctx)
            loss = compute_loss(logits=logits, y=tgt, loss_fn=loss_fn)

            train_losses.append(loss.item())
            batches.set_postfix({"loss": f"{loss.item():.4f}"})
            if config["wandb"]:
                wandb.log({"batch_total_loss": loss.item()})

            # Backprop
            optimizer.zero_grad(set_to_none=True)
            if config["clip_gradients"]:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            loss.backward()
            optimizer.step()

            # Save loss after each batch
            plt.plot(train_losses)
            loss_plot_path = os.path.join(model_dir, "loss.png")
            plt.tight_layout()
            plt.savefig(loss_plot_path)
            plt.clf()

            ix += 1

            if ix % eval_interval == 0:
                val = evaluate_hits_encdec(
                    train_loader=train_loader,
                    val_loader=val_loader,
                    model=model,
                    config=config,
                    epoch=epoch,
                    loss_fn=loss_fn,
                    model_name=model_name,
                    model_dir=model_dir,
                    device=device,
                )
                evals.append(val)

                e_ixs = range(len(evals))
                eval_val_losses = [evals[i]["val_loss"] for i in e_ixs]
                eval_train_losses = [evals[i]["train_loss"] for i in e_ixs]
                marker = "o" if epoch == 1 else None
                plt.plot(
                    e_ixs, eval_train_losses, label="train", c="blue", marker=marker
                )
                plt.plot(e_ixs, eval_val_losses, label="val", c="orange", marker=marker)
                eval_loss_plot_path = os.path.join(model_dir, "eval_loss.png")
                plt.legend()
                plt.title(f"{model_name}")
                plt.tight_layout()
                plt.savefig(eval_loss_plot_path)
                plt.clf()

        if config["checkpoints"]:
            save_checkpoint(
                model_dir=model_dir,
                epoch=epoch,
                model=model,
                optimizer=optimizer,
                loss=loss,
                config=config,
                evals=evals,
                delete_prev=True,
            )

    # Final eval
    evals.append(
        evaluate_hits_encdec(
            train_loader=train_loader,
            val_loader=val_loader,
            model=model,
            config=config,
            epoch=epoch,
            loss_fn=loss_fn,
            model_name=model_name,
            model_dir=model_dir,
            device=device,
        )
    )

    return evals


def train(config, model_name, datasets_dir, model_dir, sweep=False):
    if config["wandb"]:
        wandb.init(project=WANDB_PROJECT_NAME, config=config, name=model_name)
        wandb.config.update(config)
        if sweep:
            config = wandb.config

    assert config["n_eval_seqs"] >= 3

    dataset = PartPairDataset(**config["data"], datasets_dir=datasets_dir)

    splits = config["splits"]
    train_data, val_data, test_data = random_split(dataset, list(splits.values()))
    for k, v in {"train": train_data, "val": val_data, "test": test_data}.items():
        ix_path = os.path.join(model_dir, f"{k}_ixs.csv")
        pd.Series(v.indices).to_csv(ix_path, index=False, header=False)
    print(f"{splits=}: {len(train_data)}, {len(val_data)}, {len(test_data)}")

    train_loader = DataLoader(train_data, batch_size=config["batch_size"], shuffle=True)
    val_loader = DataLoader(val_data, batch_size=config["batch_size"], shuffle=True)

    # hits_vocab = get_hits_vocab()
    # ttoi = {v: k for k, v in hits_vocab.items()}
    # start_ix = ttoi["start"]
    hits_vocab_size = get_hits_vocab_size(block_size=config["data"]["block_size"])

    config["model"]["src_vocab_size"] = hits_vocab_size
    config["model"]["tgt_vocab_size"] = hits_vocab_size
    config["model"]["context_len"] = int(
        config["sequence_len"] / config["data"]["block_size"]
    )

    model = TransformerEncoderDecoder(**config["model"]).to(DEVICE)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config["lr"],
        weight_decay=config["weight_decay"],
    )
    loss_fn = get_loss_fn(config)

    print(yaml.dump(config))

    evals = train_hits_encdec(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        loss_fn=loss_fn,
        config=config,
        model_name=model_name,
        model_dir=model_dir,
        device=DEVICE,
    )

    model_path = os.path.join(model_dir, "model.pt")
    save_model(
        model_path=model_path,
        model=model,
        config=config,
        model_name=model_name,
        evals=evals,
    )


if __name__ == "__main__":
    print(f"{DEVICE=}")

    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets_dir", type=str, default=DATASETS_DIR)
    parser.add_argument("--config_path", type=str, default=DEFAULT_CONFIG_FILEPATH)
    args = parser.parse_args()

    datasets_dir = args.datasets_dir
    config = load_config(args.config_path)

    torch.manual_seed(config["seed"])

    model_name = get_model_name()
    print(f"{model_name=}")

    model_dir = os.path.join(MODELS_DIR, model_name)
    if not os.path.isdir(model_dir):
        os.makedirs(model_dir)

    train(
        config=config,
        model_name=model_name,
        datasets_dir=datasets_dir,
        model_dir=model_dir,
    )
