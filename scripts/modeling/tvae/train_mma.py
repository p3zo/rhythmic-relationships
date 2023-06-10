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

from rhythmic_relationships.model_utils import get_model_name, load_config, save_model
from rhythmic_relationships import DATASETS_DIR, MODELS_DIR
from rhythmic_relationships.data import (
    PartPairDatasetSequential,
    get_roll_from_sequence,
)
from rhythmic_relationships.ext_models import MuseMorphoseAdapted
from rhythmic_relationships.io import write_midi_from_roll
from rhythmic_relationships.vocab import get_vocab_encoder_decoder, get_vocab_sizes
from rhythmic_relationships.train import save_checkpoint

DEFAULT_CONFIG_FILEPATH = "config_mma.yml"
WANDB_PROJECT_NAME = "rhythmic-relationships"

DEVICE = torch.device(
    "mps"
    if torch.backends.mps.is_built()
    else torch.device("cuda:0")
    if torch.cuda.device_count() > 0
    else torch.device("cpu")
)


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
    sorted_index = np.argsort(probs)[::-1]
    cumsum_sorted_probs = np.cumsum(sorted_probs)
    after_threshold = cumsum_sorted_probs > p
    if sum(after_threshold) > 0:
        last_index = np.where(after_threshold)[0][1]
        candi_index = sorted_index[:last_index]
    else:
        candi_index = sorted_index[:3]  # just assign a value
    candi_probs = np.array([probs[i] for i in candi_index], dtype=np.float64)
    candi_probs /= sum(candi_probs)
    word = np.random.choice(candi_index, size=1, p=candi_probs)[0]
    return word


def inference_latent_vanilla_truncate(
    model,
    latents,
    y,
    n_new_tokens,
    nucleus_p,
    temperature,
):
    latent_placeholder = torch.zeros(
        n_new_tokens + len(y),
        1,
        latents.size(-1),
        device=DEVICE,
    )

    entropies = []

    for _ in range(n_new_tokens):
        latent_placeholder[len(y) - 1, 0, :] = latents[len(y)]
        dec_seg_emb = latent_placeholder[: len(y), :]

        # Get the predictions
        with torch.no_grad():
            logits = model.generate(y=y, latent=dec_seg_emb)

        # Take the logits for the last tokens
        logits = logits[:, -1, :]

        # Apply softmax to get probabilities
        probs = temperatured_softmax(logits.cpu().numpy(), temperature)

        y_next = []
        for ix in range(probs.shape[0]):
            yn = nucleus(probs[ix], nucleus_p)
            y_next.append(yn)
        y_next = torch.tensor(y_next, dtype=torch.long, device=DEVICE).unsqueeze(1)

        y = torch.cat((y, y_next), dim=1)
        entropies.append(entropy(probs))

    return y, np.array(entropies)


def parse_mma_batch(batch):
    src, ctx, tgt = batch
    src = src.to(DEVICE).view(src.shape[0] * src.shape[1], src.shape[2])
    ctx = ctx.to(DEVICE).view(ctx.shape[0] * ctx.shape[1], ctx.shape[2])
    tgt = tgt.to(DEVICE).view(tgt.shape[0] * tgt.shape[1], tgt.shape[2])
    return src, ctx, tgt


def evaluate_mma(
    val_loader,
    model,
    config,
    epoch,
    model_name,
    model_dir,
):
    model.eval()

    evaluation = {}

    eval_dir = os.path.join(model_dir, "eval", f"epoch_{epoch}")
    if not os.path.isdir(eval_dir):
        os.makedirs(eval_dir)

    n_eval_iters = config["n_eval_iters"]
    part_2 = config["data"]["part_2"]

    vocab_encoder, _ = get_vocab_encoder_decoder(part_2)
    start_ix = vocab_encoder(["start"])[0]

    print(f"Evaluating for {n_eval_iters} iters")

    evals_total_loss = []
    evals_kldiv_loss = []
    evals_kldiv_raw = []
    evals_recons_loss = []

    for k in range(n_eval_iters):
        src, ctx, tgt = parse_mma_batch(next(iter(val_loader)))

        with torch.no_grad():
            mu, logvar, dec_logits = model(src, ctx)

            losses = model.compute_loss(
                mu=mu,
                logvar=logvar,
                beta=config["kl_max_beta"],
                fb_lambda=config["free_bit_lambda"],
                dec_logits=dec_logits,
                dec_tgt=tgt,
            )

            evals_total_loss.append(losses["total_loss"].item())
            evals_kldiv_loss.append(losses["kldiv_loss"].item())
            evals_kldiv_raw.append(losses["kldiv_raw"].item())
            evals_recons_loss.append(losses["recons_loss"].item())

        desc_dfs = []
        n_generated = 0
        all_zeros = 0

        with torch.no_grad():
            latents = model.get_sampled_latent(
                src,
                use_sampling=False,
                sampling_var=0.0,
            )

        start_ctx = torch.full(
            (src.shape[0], 1),
            start_ix,
            dtype=torch.long,
            device=DEVICE,
        )

        seqs, entropies = inference_latent_vanilla_truncate(
            model=model,
            latents=latents,
            y=start_ctx,
            n_new_tokens=32,
            nucleus_p=0.9,
            temperature=1.2,
        )

        print(
            "[entropy] {:.4f} (+/- {:.4f})".format(
                np.mean(entropies), np.std(entropies)
            )
        )

        for ix, seq in enumerate(seqs):
            gen_roll = get_roll_from_sequence(seq.cpu().numpy(), part=part_2)
            tgt_roll = get_roll_from_sequence(tgt[ix].cpu().numpy(), part=part_2)

            # Compare descriptors of the generated and target rolls
            gen_roll_descs = pianoroll2descriptors(
                gen_roll,
                config["resolution"],
                drums=part_2 == "Drums",
            )
            tgt_roll_descs = pianoroll2descriptors(
                tgt_roll,
                config["resolution"],
                drums=part_2 == "Drums",
            )
            df = pd.DataFrame.from_dict(
                {"generated": gen_roll_descs, "target": tgt_roll_descs},
                orient="index",
            )
            desc_dfs.append(df)

            n_generated += 1
            if gen_roll.max() == 0:
                all_zeros += 1
                continue

            write_midi_from_roll(
                gen_roll,
                outpath=os.path.join(eval_dir, f"{k}_{ix}_gen.mid"),
                part=part_2,
                binary=False,
                onset_roll=True,
            )

        print(f"{n_generated=}, {all_zeros=} ({100*round(all_zeros/n_generated, 2)}%)")

        desc_df = pd.concat(desc_dfs).dropna(how="all", axis=1)
        if "stepDensity" in desc_df.columns:
            desc_df.drop("stepDensity", axis=1, inplace=True)

        # Scale the feature columns to [0, 1]
        desc_df_scaled = (desc_df - desc_df.min()) / (desc_df.max() - desc_df.min())

        # Plot a comparison of distributions for all descriptors
        sns.boxplot(
            x="variable",
            y="value",
            hue="index",
            data=pd.melt(desc_df_scaled.reset_index(), id_vars="index"),
        )
        plt.ylabel("")
        plt.xlabel("")
        plt.yticks([])
        plt.xticks(rotation=90)
        plt.title(f"{model_name}\nn={n_eval_iters}")
        plt.tight_layout()
        plt.savefig(os.path.join(eval_dir, "gen_tgt_comparison.png"))
        plt.clf()

        curr_eval = {
            "val_total_loss": np.mean(evals_total_loss),
            "val_kldiv_loss": np.mean(evals_kldiv_loss),
            "val_kldiv_raw": np.mean(evals_kldiv_raw),
            "val_recons_loss": np.mean(evals_recons_loss),
        }
        print(f"{curr_eval=}")

        evaluation.update(curr_eval)

        if config["wandb"]:
            wandb.log(curr_eval)

    model.train()

    return evaluation


def train_mma(
    model,
    train_loader,
    val_loader,
    optimizer,
    config,
    model_name,
    model_dir,
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
            src, ctx, tgt = parse_mma_batch(batch)

            mu, logvar, dec_logits = model(src, ctx)

            losses = model.compute_loss(
                mu=mu,
                logvar=logvar,
                beta=config["kl_max_beta"],
                fb_lambda=config["free_bit_lambda"],
                dec_logits=dec_logits,
                dec_tgt=tgt,
            )

            train_losses.append(losses["total_loss"].item())
            batches.set_postfix({"loss": f"{losses['total_loss'].item():.4f}"})
            if config["wandb"]:
                wandb.log({"batch_total_loss": losses["total_loss"].item()})

            # Backprop
            optimizer.zero_grad(set_to_none=True)
            # TODO: try removing gradient clips
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            losses["total_loss"].backward()
            optimizer.step()

            # Save loss after each batch
            plt.plot(train_losses)
            loss_plot_path = os.path.join(model_dir, "loss.png")
            plt.tight_layout()
            plt.savefig(loss_plot_path)
            plt.clf()

            ix += 1

            if ix % val_interval == 0:
                val = evaluate_mma(
                    val_loader=val_loader,
                    model=model,
                    config=config,
                    epoch=epoch,
                    model_name=model_name,
                    model_dir=model_dir,
                )
                evals.append(val)

        if config["checkpoints"]:
            save_checkpoint(
                model_dir=model_dir,
                epoch=epoch,
                model=model,
                optimizer=optimizer,
                loss=losses["total_loss"],
                config=config,
                evals=evals,
                delete_prev=True,
            )

    return evals


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

    dataset = PartPairDatasetSequential(
        **config["data"],
        datasets_dir=datasets_dir,
        with_ctx=True,
    )
    splits = config["splits"]
    train_data, val_data, test_data = random_split(dataset, list(splits.values()))
    for k, v in {"train": train_data, "val": val_data, "test": test_data}.items():
        ix_path = os.path.join(model_dir, f"{k}_ixs.csv")
        pd.Series(v.indices).to_csv(ix_path, index=False, header=False)
    print(f"{splits=}: {len(train_data)}, {len(val_data)}, {len(test_data)}")

    train_loader = DataLoader(train_data, batch_size=config["batch_size"], shuffle=True)
    val_loader = DataLoader(val_data, batch_size=config["batch_size"], shuffle=True)

    vocab_sizes = get_vocab_sizes()
    encode, _ = get_vocab_encoder_decoder(config["data"]["part_2"])
    config["model"]["src_vocab_size"] = vocab_sizes[config["data"]["part_1"]]
    config["model"]["tgt_vocab_size"] = vocab_sizes[config["data"]["part_2"]]
    config["model"]["pad_ix"] = encode(["pad"])[0]
    # Add 1 to the context length to account for the start token
    config["model"]["context_len"] = config["sequence_len"] + 1
    print(yaml.dump(config))

    model = MuseMorphoseAdapted(**config["model"]).to(DEVICE)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config["lr"],
        weight_decay=config["weight_decay"],
    )

    if config["wandb"]:
        wandb.init(project=WANDB_PROJECT_NAME, config=config, name=model_name)
        wandb.config.update(config)

    evals = train_mma(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        config=config,
        model_name=model_name,
        model_dir=model_dir,
    )

    model_path = os.path.join(model_dir, "model.pt")
    save_model(
        model_path=model_path,
        model=model,
        config=config,
        model_name=model_name,
        evals=evals,
    )
