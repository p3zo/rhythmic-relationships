import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import wandb
import yaml
from scipy.stats import entropy
from rhythmtoolbox import pianoroll2descriptors
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from rhythmic_relationships.evaluate import temperatured_softmax
from rhythmic_relationships.model_utils import (
    get_model_name,
    load_config,
    save_model,
    save_checkpoint,
    get_loss_fn,
)
from rhythmic_relationships import DATASETS_DIR, MODELS_DIR
from rhythmic_relationships.data import (
    PartDataset,
    get_roll_from_sequence,
    get_hits_from_hits_seq,
)
from rhythmic_relationships.models.hits_decoder_xformer import HitsDecoder
from rhythmic_relationships.io import write_midi_from_hits
from rhythmic_relationships.vocab import PAD_IX, START_IX, get_hits_vocab_size

DEFAULT_CONFIG_FILEPATH = "config.yml"
WANDB_PROJECT_NAME = "rhythmic-relationships"

DEVICE = torch.device(
    "mps"
    if torch.backends.mps.is_built()
    else torch.device("cuda:0")
    if torch.cuda.device_count() > 0
    else torch.device("cpu")
)


def parse_batch(batch, device):
    yb = batch

    # Teacher forcing: the decoder reads the target shifted right by one, seeded with `start`
    yb_shifted = torch.roll(yb, 1, dims=1)
    yb_shifted[:, 0] = START_IX

    return yb_shifted.to(device), yb.to(device)


def inference(model, n_tokens, temperature, device):
    y = torch.tensor(
        [[START_IX]],
        dtype=torch.long,
        requires_grad=False,
        device=device,
    )

    entropies = []

    for _ in range(n_tokens):
        # Get the predictions
        with torch.no_grad():
            logits = model(y)

        # Take the logits for the last tokens
        logits = logits[:, -1, :]

        # Apply softmax to get probabilities
        probs = temperatured_softmax(logits.cpu().numpy(), temperature)
        entropies.append(entropy(probs))

        y_next = torch.multinomial(
            torch.tensor(probs, dtype=torch.float32, device=device),
            num_samples=1,
        )

        y = torch.cat([y, y_next], dim=1)

    # Drop the start token the sequence was seeded with
    return y[:, 1:], np.array(entropies)


def compute_loss(logits, y, loss_fn):
    B, T, C = logits.shape
    return loss_fn(logits.view(B * T, C), y.view(y.shape[0] * y.shape[1]))


def evaluate_hits_decoder(
    val_loader,
    model,
    config,
    epoch,
    loss_fn,
    model_name,
    model_dir,
    device,
):
    model.eval()

    evaluation = {}

    eval_dir = os.path.join(model_dir, "eval", f"epoch_{epoch}")
    eix = 0
    while os.path.isdir(eval_dir):
        eval_dir = os.path.join(model_dir, "eval", f"epoch_{epoch}_{eix}")
        eix += 1
    if not os.path.isdir(eval_dir):
        os.makedirs(eval_dir)

    n_eval_iters = config["n_eval_iters"]
    part = config["data"]["part"]

    print(f"Evaluating for {n_eval_iters} iters")

    evals_loss = []

    for k, batch in enumerate(val_loader):
        if k == n_eval_iters:
            break
        ctx, tgt = parse_batch(batch, device)
        with torch.no_grad():
            logits = model(ctx)
            loss = compute_loss(logits=logits, y=tgt, loss_fn=loss_fn)
            evals_loss.append(loss.item())

    n_generated = 0
    all_zeros = 0
    all_same = 0

    # seqs, entropies = inference(
    #     model=model, n_samples=10, n_tokens=32, temperature=1.2, device=device
    # )
    # print("[entropy] {:.4f} (+/- {:.4f})".format(np.mean(entropies), np.std(entropies)))
    # for ix, seq in enumerate(seqs):

    n_seqs = 1
    for ix in range(n_seqs):
        seq, _ = inference(
            model=model,
            n_tokens=config["sequence_len"],
            temperature=1.2,
            device=device,
        )

        gen_hits = get_hits_from_hits_seq(seq.squeeze(0).cpu().numpy(), part=part)

        n_generated += 1
        if max(gen_hits) == 0:
            all_zeros += 1
            continue
        if len(set(gen_hits)) == 1:
            all_same += 1
            continue

        write_midi_from_hits(
            [i * 127 for i in gen_hits],
            outpath=os.path.join(eval_dir, f"{k}_{ix}_gen.mid"),
            part=part,
            pitch=72,
        )

    print(f"{n_generated=}")
    print(f"  {all_zeros=} ({100*round(all_zeros/n_generated, 2)}%)")
    print(f"  {all_same=} ({100*round(all_same/n_generated, 2)}%)")

    curr_eval = {"val_loss": np.mean(evals_loss)}
    print(f"{curr_eval=}")

    evaluation.update(curr_eval)

    if config["wandb"]:
        wandb.log(curr_eval)

    model.train()

    return evaluation


def train_hits_decoder(
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
    val_interval = config["val_interval"]

    evals = []
    train_losses = []

    model.train()

    ix = 0
    for epoch in range(1, n_epochs + 1):
        batches = tqdm(train_loader)
        for batch in batches:
            ctx, tgt = parse_batch(batch, device)
            logits = model(ctx)
            loss = compute_loss(logits=logits, y=tgt, loss_fn=loss_fn)

            train_losses.append(loss.item())
            batches.set_postfix({"loss": f"{loss.item():.4f}"})
            if config["wandb"]:
                wandb.log({"batch_total_loss": loss.item()})

            # Backprop
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if config["clip_gradients"]:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.25)
            optimizer.step()


            ix += 1

            if ix % val_interval == 0:
                val = evaluate_hits_decoder(
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
        evaluate_hits_decoder(
            val_loader=val_loader,
            model=model,
            config=config,
            epoch=epoch,
            loss_fn=loss_fn,
            model_name=model_name,
            model_dir=model_dir,
            device=DEVICE,
        )
    )

    return evals


def train(config, model_name, datasets_dir, model_dir):
    del config["data"]["context_len"]
    dataset = PartDataset(**config["data"], datasets_dir=datasets_dir)
    splits = config["splits"]
    train_data, val_data, test_data = random_split(dataset, list(splits.values()))
    for k, v in {"train": train_data, "val": val_data, "test": test_data}.items():
        ix_path = os.path.join(model_dir, f"{k}_ixs.csv")
        pd.Series(v.indices).to_csv(ix_path, index=False, header=False)
    print(f"{splits=}: {len(train_data)}, {len(val_data)}, {len(test_data)}")

    train_loader = DataLoader(train_data, batch_size=config["batch_size"], shuffle=True)
    val_loader = DataLoader(val_data, batch_size=config["batch_size"], shuffle=True)

    pad_ix = PAD_IX

    config["model"]["vocab_size"] = get_hits_vocab_size(config["data"].get("block_size", 1))
    config["model"]["context_len"] = config["sequence_len"]
    config["model"]["pad_ix"] = pad_ix

    model = HitsDecoder(**config["model"]).to(DEVICE)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config["lr"],
        weight_decay=config["weight_decay"],
    )
    loss_fn = get_loss_fn(config, pad_ix=pad_ix)

    if config["wandb"]:
        wandb.init(project=WANDB_PROJECT_NAME, config=config, name=model_name)
        wandb.config.update(config)

    print(yaml.dump(config))
    evals = train_hits_decoder(
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
