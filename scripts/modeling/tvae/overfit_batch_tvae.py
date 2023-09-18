"""Same as overfit_single_batch, but for MuseMorphoseAdapted"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
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
from rhythmic_relationships.evaluate import temperatured_softmax, nucleus
from rhythmic_relationships.model_tvae import MuseMorphoseAdapted
from rhythmic_relationships.io import write_midi_from_roll
from rhythmic_relationships.vocab import get_vocab_encoder_decoder, get_vocab_sizes


DEFAULT_CONFIG_FILEPATH = "config.yml"

DEVICE = torch.device(
    "mps"
    if torch.backends.mps.is_built()
    else torch.device("cuda:0")
    if torch.cuda.device_count() > 0
    else torch.device("cpu")
)


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
            logits = model.generate(dec_seg_emb=dec_seg_emb, ctx=y)

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


def evaluate_single_batch(
    src,
    tgt,
    model,
    config,
    epoch,
    model_name,
    model_dir,
):
    model.eval()

    eval_dir = os.path.join(model_dir, "eval", f"epoch_{epoch}")
    if not os.path.isdir(eval_dir):
        os.makedirs(eval_dir)

    n_eval_iters = config["n_eval_iters"]
    part_2 = config["data"]["part_2"]

    # # TODO: select eval ixs automatically
    # src = src[[32, -1]]
    # tgt = tgt[[32, -1]]

    print(f"Evaluating for {n_eval_iters} iters")

    vocab_encoder, _ = get_vocab_encoder_decoder(part_2)
    start_ix = vocab_encoder(["start"])[0]

    desc_dfs = []
    n_generated = 0
    all_zeros = 0

    for k in range(n_eval_iters):
        with torch.no_grad():
            latents = model.get_sampled_latent(
                src,
                use_sampling=False,
                sampling_var=0.0,
            )

        ctx = torch.full(
            (src.shape[0], 1),
            start_ix,
            dtype=torch.long,
            device=DEVICE,
        )

        seqs, entropies = inference_latent_vanilla_truncate(
            model=model,
            latents=latents,
            y=ctx,
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

        # Log eval losses
        model.train()


def overfit_single_batch(
    model,
    src,
    ctx,
    tgt,
    optimizer,
    config,
    model_name,
    model_dir,
):
    n_epochs = config["n_epochs"]
    evaluate_every = n_epochs

    train_losses = []

    model.train()

    epochs = tqdm(range(1, n_epochs + 1))

    for epoch in epochs:
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
        epochs.set_postfix({"loss": f"{losses['total_loss'].item():.4f}"})

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

        if epoch > 1 and epoch % evaluate_every == 0:
            evaluate_single_batch(
                src=src,
                tgt=tgt,
                model=model,
                config=config,
                epoch=epoch,
                model_name=model_name,
                model_dir=model_dir,
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

    dataset = PartPairDatasetSequential(
        **config["data"],
        datasets_dir=datasets_dir,
        with_ctx=True,
    )
    splits = config["splits"]
    train_data, val_data, test_data = random_split(dataset, list(splits.values()))

    train_loader = DataLoader(train_data, batch_size=config["batch_size"], shuffle=True)

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
    model_name = get_model_name()
    print(f"{model_name=}")

    model_dir = os.path.join(MODELS_DIR, model_name)
    if not os.path.isdir(model_dir):
        os.makedirs(model_dir)

    src, ctx, tgt = next(iter(train_loader))

    for ix in range(config["batch_size"]):
        src_roll = get_roll_from_sequence(
            src[ix][-1].numpy(),
            part=config["data"]["part_2"],
        )
        tgt_roll = get_roll_from_sequence(
            tgt[ix][-1].numpy(),
            part=config["data"]["part_2"],
        )

        write_midi_from_roll(
            src_roll,
            outpath=os.path.join(model_dir, f"{ix}_src.mid"),
            part=config["data"]["part_1"],
            binary=False,
            onset_roll=True,
        )

        write_midi_from_roll(
            tgt_roll,
            outpath=os.path.join(model_dir, f"{ix}_tgt.mid"),
            part=config["data"]["part_2"],
            binary=False,
            onset_roll=True,
        )

    src = src.to(DEVICE).view(src.shape[0] * src.shape[1], src.shape[2])
    ctx = ctx.to(DEVICE).view(ctx.shape[0] * ctx.shape[1], ctx.shape[2])
    tgt = tgt.to(DEVICE).view(tgt.shape[0] * tgt.shape[1], tgt.shape[2])

    overfit_single_batch(
        model=model,
        src=src,
        ctx=ctx,
        tgt=tgt,
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
    )
