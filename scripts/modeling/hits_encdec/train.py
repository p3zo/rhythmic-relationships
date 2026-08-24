import argparse
import os

import matplotlib.pyplot as plt
import pandas as pd
import torch
import wandb
import yaml
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
from torch.utils.data import DataLoader, Subset, random_split
from tqdm import tqdm

from .eval import evaluate_hits_encdec
from .utils import compute_loss, parse_batch

DEFAULT_CONFIG_FILEPATH = "config.yml"

DEVICE = torch.device(
    "mps"
    if torch.backends.mps.is_built()
    else torch.device("cuda:0")
    if torch.cuda.device_count() > 0
    else torch.device("cpu")
)


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
    start_epoch=1,
    prior_evals=None,
):
    n_epochs = config["n_epochs"]
    eval_interval = config["eval_interval"]

    # A resumed run carries its eval history forward so eval_loss.png stays one curve. The
    # per-batch losses are not in the checkpoint, so loss.png restarts at the resume point.
    evals = list(prior_evals or [])
    train_losses = []

    model.train()

    ix = 0
    for epoch in range(start_epoch, n_epochs + 1):
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
            loss.backward()
            if config["clip_gradients"]:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()


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

        # Save the batch loss curve once per epoch rather than once per batch
        plt.plot(train_losses)
        plt.tight_layout()
        plt.savefig(os.path.join(model_dir, "loss.png"))
        plt.clf()

        if config["checkpoints"]:
            save_checkpoint(
                model_dir=model_dir,
                epoch=epoch,
                model=model,
                optimizer=optimizer,
                loss=loss.item(),
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


def train(config, model_name, datasets_dir, model_dir, sweep=False, resume=None):
    if config["wandb"]:
        wandb.init(project=WANDB_PROJECT_NAME, config=config, name=model_name)
        wandb.config.update(config)
        if sweep:
            config = wandb.config

    assert config["n_eval_seqs"] >= 3

    dataset = PartPairDataset(**config["data"], datasets_dir=datasets_dir)

    splits = config["splits"]
    if resume:
        # Re-splitting would put some of the original validation set into training, so the
        # resumed run reads back the exact indices the first one wrote
        train_data, val_data, test_data = (
            Subset(dataset, pd.read_csv(os.path.join(model_dir, f"{k}_ixs.csv"), header=None)[0].tolist())
            for k in ("train", "val", "test")
        )
    else:
        train_data, val_data, test_data = random_split(dataset, list(splits.values()))
        for k, v in {"train": train_data, "val": val_data, "test": test_data}.items():
            ix_path = os.path.join(model_dir, f"{k}_ixs.csv")
            pd.Series(v.indices).to_csv(ix_path, index=False, header=False)
    print(f"{splits=}: {len(train_data)}, {len(val_data)}, {len(test_data)}")

    # Every item opens an .npz, so the step is mostly waiting on disk. Workers overlap that with
    # the forward and backward pass, and the dataset does not track which segments it read, which
    # is the one thing that would not survive being loaded in another process.
    #
    # Not persistent_workers, however tempting: the eval below iterates train_loader again from
    # inside the epoch that is already iterating it, and a persistent pool is shared between the
    # two iterators. The inner one resets it, the outer one is fed forever, and the epoch never
    # ends - no checkpoint, no next epoch, just evals every eval_interval steps until killed.
    loader_kwargs = dict(batch_size=config["batch_size"], shuffle=True,
                         num_workers=config["num_workers"])
    if config["num_workers"]:
        loader_kwargs |= dict(prefetch_factor=4)
    train_loader = DataLoader(train_data, **loader_kwargs)
    val_loader = DataLoader(val_data, **loader_kwargs)

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

    start_epoch, prior_evals = 1, None
    if resume:
        model.load_state_dict(resume["model_state_dict"])
        # Adam's moment estimates go with it, so this continues the run rather than restarting
        # the optimiser at the weights it had reached
        optimizer.load_state_dict(resume["optimizer_state_dict"])
        start_epoch = resume["epoch"] + 1
        prior_evals = resume["evals"]
        if start_epoch > config["n_epochs"]:
            raise ValueError(
                f"Checkpoint is at epoch {resume['epoch']} and n_epochs is "
                f"{config['n_epochs']}; pass --epochs with a larger total to continue"
            )
        print(f"Resuming at epoch {start_epoch} of {config['n_epochs']}")

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
        start_epoch=start_epoch,
        prior_evals=prior_evals,
    )

    model_path = os.path.join(model_dir, "model.pt")
    save_model(
        model_path=model_path,
        model=model,
        config=config,
        model_name=model_name,
        evals=evals,
    )
