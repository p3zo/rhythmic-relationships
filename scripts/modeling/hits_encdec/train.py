import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import wandb
import yaml
from rhythmic_relationships import DATASETS_DIR, MODELS_DIR
from rhythmic_relationships.data import (
    PartPairDatasetSequential,
    get_hits_from_hits_seq,
)
from rhythmic_relationships.io import write_midi_from_hits
from rhythmic_relationships.model_utils import (
    get_loss_fn,
    get_model_name,
    load_config,
    save_checkpoint,
    save_model,
)
from rhythmic_relationships.models.hits_encdec import TransformerEncoderDecoder
from rhythmic_relationships.vocab import get_hits_vocab
from torch.utils.data import DataLoader, random_split
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


def parse_sequential_batch(batch, device):
    xb, yb = batch
    x = xb.to(device).view(xb.shape[0] * xb.shape[1], xb.shape[2])
    y = yb.to(device).view(yb.shape[0] * yb.shape[1], yb.shape[2])
    return x, y


def temperatured_softmax(logits, temperature):
    try:
        probs = np.exp(logits / temperature) / np.sum(np.exp(logits / temperature))
        assert np.count_nonzero(np.isnan(probs)) == 0
    except:
        print("overflow detected, use 128-bit")
        logits = logits.astype(np.float128)
        probs = np.exp(logits / temperature) / np.sum(np.exp(logits / temperature))
        probs = probs.astype(float)
    return probs


def nucleus(probs, p):
    probs /= sum(probs)
    sorted_probs = np.sort(probs)[::-1]
    sorted_ixs = np.argsort(probs)[::-1]
    cumsum_sorted_probs = np.cumsum(sorted_probs)
    after_thresh = cumsum_sorted_probs > p
    if after_thresh.sum() > 0:
        last_index = np.where(after_thresh)[0][-1]
        candidate_ixs = sorted_ixs[:last_index]
    else:
        # just assign a value
        candidate_ixs = sorted_ixs[:3]
    candidate_probs = np.array([probs[i] for i in candidate_ixs], dtype=np.float64)
    candidate_probs /= sum(candidate_probs)
    return np.random.choice(candidate_ixs, size=1, p=candidate_probs)[0]


def compute_loss(logits, y, loss_fn):
    B, T, C = logits.shape
    return loss_fn(logits.view(B * T, C), y.view(y.shape[0] * y.shape[1]))


def inference(
    model,
    src,
    n_tokens,
    temperature,
    device,
    sampler="multinomial",
    nucleus_p=0.92,
):
    if sampler not in ["multinomial", "nucleus"]:
        raise ValueError(f"Unsupported {sampler}: sampler")

    hits_vocab = get_hits_vocab()
    ttoi = {v: k for k, v in hits_vocab.items()}
    pad_ix = ttoi["pad"]

    y = torch.tensor(
        [[pad_ix] * n_tokens],
        dtype=torch.long,
        requires_grad=False,
        device=device,
    )

    for ix in range(n_tokens):
        # Get the predictions
        with torch.no_grad():
            logits = model(src=src, tgt=y)

        # Take the logits for the last tokens
        logits = logits[:, -1, :]

        # Apply softmax to get probabilities
        probs = temperatured_softmax(logits.cpu().numpy(), temperature)

        if sampler == "nucleus":
            y_next = []
            for j in range(probs.shape[0]):
                yn = nucleus(probs[j], p=nucleus_p)
                y_next.append(yn)
            y_next = torch.tensor(y_next, dtype=torch.long, device=DEVICE).unsqueeze(1)
        else:
            y_next = torch.multinomial(
                torch.tensor(probs, dtype=torch.float32, device=DEVICE),
                num_samples=1,
            )

        y[:, ix] = y_next.item()

    return y.squeeze(0)


def pct_diff(x, y):
    return 100 * np.abs(x - y) / ((x + y) / 2)


def evaluate_hits_encdec(
    train_loader,
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
    part_1 = config["data"]["part_1"]
    part_2 = config["data"]["part_2"]

    print(f"Evaluating for {n_eval_iters} iters")

    evals_train_loss = []
    for k in range(n_eval_iters):
        src, tgt = parse_sequential_batch(next(iter(train_loader)), device)
        with torch.no_grad():
            logits = model(src, tgt)
            loss = compute_loss(logits=logits, y=tgt, loss_fn=loss_fn)
            evals_train_loss.append(loss.item())

    evals_val_loss = []
    for k in range(n_eval_iters):
        src, tgt = parse_sequential_batch(next(iter(val_loader)), device)
        with torch.no_grad():
            logits = model(src, tgt)
            loss = compute_loss(logits=logits, y=tgt, loss_fn=loss_fn)
            evals_val_loss.append(loss.item())

    sampler_stats = {}

    n_seqs = 10
    gen_srcs, gen_tgts = parse_sequential_batch(next(iter(val_loader)), device)
    for sampler in ["multinomial", "nucleus"]:

        sample_stats = {
            "n_generated": 0,
            "all_zeros": 0,
            "all_same": 0,
        }

        for ix in range(n_seqs):
            src = gen_srcs[ix].unsqueeze(0)

            seq = inference(
                model=model,
                src=src,
                n_tokens=32,
                temperature=1.2,
                sampler=sampler,
                device=device,
            )

            gen_hits = get_hits_from_hits_seq(
                seq.cpu().numpy(), part=part_2, verbose=True
            )

            sample_stats["n_generated"] += 1
            if max(gen_hits) == 0:
                sample_stats["all_zeros"] += 1
                continue
            if len(set(gen_hits)) == 1:
                sample_stats["all_same"] += 1
                continue

            write_midi_from_hits(
                [i * 127 for i in gen_hits],
                outpath=os.path.join(eval_dir, f"{ix}_{sampler}_gen.mid"),
                part=part_2,
                pitch=72,
            )

            tgt = gen_tgts[ix]
            tgt_hits = get_hits_from_hits_seq(tgt.cpu().numpy(), part=part_2)
            write_midi_from_hits(
                [i * 127 for i in tgt_hits],
                outpath=os.path.join(eval_dir, f"{ix}_{sampler}_tgt.mid"),
                part=part_2,
                pitch=72,
            )

            src_hits = get_hits_from_hits_seq(src.squeeze(0).cpu().numpy(), part=part_1)
            write_midi_from_hits(
                [i * 127 for i in src_hits],
                outpath=os.path.join(eval_dir, f"{ix}_{sampler}_src.mid"),
                part=part_1,
                pitch=72,
            )

        sample_stats["pct_all_zeros"] = 100 * round(
            sample_stats["all_zeros"] / sample_stats["n_generated"],
            2,
        )
        sample_stats["pct_all_same"] = 100 * round(
            sample_stats["all_same"] / sample_stats["n_generated"],
            2,
        )
        sampler_stats[sampler] = sample_stats

    print(sampler_stats)

    val_loss_mean = np.mean(evals_val_loss)
    train_loss_mean = np.mean(evals_train_loss)
    curr_eval = {
        "val_loss": val_loss_mean,
        "train_loss": train_loss_mean,
        "val_train_loss_pct_diff": pct_diff(val_loss_mean, train_loss_mean),
        "sampler_stats": sampler_stats,
    }
    print(f"{curr_eval=}")

    evaluation.update(curr_eval)

    if config["wandb"]:
        wandb.log(curr_eval)

    model.train()

    return evaluation


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
    val_interval = config["val_interval"]

    evals = []
    train_losses = []

    model.train()

    ix = 0
    for epoch in range(1, n_epochs + 1):
        batches = tqdm(train_loader)
        for batch in batches:
            src, tgt = parse_sequential_batch(batch, device)
            logits = model(src, tgt)
            loss = compute_loss(logits=logits, y=tgt, loss_fn=loss_fn)

            train_losses.append(loss.item())
            batches.set_postfix({"loss": f"{loss.item():.4f}"})
            if config["wandb"]:
                wandb.log({"batch_total_loss": loss.item()})

            # Backprop
            optimizer.zero_grad(set_to_none=True)
            if config["clip_gradients"]:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.25)
            loss.backward()
            optimizer.step()

            # Save loss after each batch
            plt.plot(train_losses)
            loss_plot_path = os.path.join(model_dir, "loss.png")
            plt.tight_layout()
            plt.savefig(loss_plot_path)
            plt.clf()

            ix += 1

            if ix % val_interval == 0:
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


def train(config, model_name, datasets_dir, model_dir):
    dataset = PartPairDatasetSequential(**config["data"], datasets_dir=datasets_dir)

    splits = config["splits"]
    train_data, val_data, test_data = random_split(dataset, list(splits.values()))
    for k, v in {"train": train_data, "val": val_data, "test": test_data}.items():
        ix_path = os.path.join(model_dir, f"{k}_ixs.csv")
        pd.Series(v.indices).to_csv(ix_path, index=False, header=False)
    print(f"{splits=}: {len(train_data)}, {len(val_data)}, {len(test_data)}")

    train_loader = DataLoader(train_data, batch_size=config["batch_size"], shuffle=True)
    val_loader = DataLoader(val_data, batch_size=config["batch_size"], shuffle=True)

    hits_vocab = get_hits_vocab()
    pad_ix = {v: k for k, v in hits_vocab.items()}["pad"]

    config["model"]["src_vocab_size"] = len(hits_vocab)
    config["model"]["tgt_vocab_size"] = len(hits_vocab)
    config["model"]["context_len"] = config["sequence_len"]
    config["model"]["pad_ix"] = pad_ix

    model = TransformerEncoderDecoder(**config["model"]).to(DEVICE)

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
