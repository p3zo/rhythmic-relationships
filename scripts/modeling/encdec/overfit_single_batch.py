import argparse
import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch
import yaml
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from rhythmic_relationships.model_utils import get_model_name, load_config, save_model
from rhythmic_relationships import DATASETS_DIR, MODELS_DIR
from rhythmic_relationships.data import (
    PartPairDatasetSequential,
    get_roll_from_sequence,
)
from rhythmic_relationships.models.encdec import TransformerEncoderDecoder
from rhythmic_relationships.io import write_midi_from_roll
from rhythmic_relationships.vocab import get_vocab_sizes, get_vocab_encoder_decoder
from rhythmtoolbox import pianoroll2descriptors

DEFAULT_CONFIG_FILEPATH = "config_encdec.yml"

DEVICE = torch.device(
    "mps"
    if torch.backends.mps.is_built()
    else torch.device("cuda:0")
    if torch.cuda.device_count() > 0
    else torch.device("cpu")
)


def evaluate_single_batch(
    src,
    tgt,
    model,
    config,
    device,
    epoch,
    model_name,
    model_dir,
):
    eval_dir = os.path.join(model_dir, "eval", f"epoch_{epoch}")
    if not os.path.isdir(eval_dir):
        os.makedirs(eval_dir)

    n_eval_iters = config["n_eval_iters"]
    part_2 = config["data"]["part_2"]

    # TODO: select ixs automatically
    # eval_ixs = [32, 65, 98, 131, 164, 197, 230, 263]
    # n_seqs = config["data"]["context_len"] * config["batch_size"]
    # eval_ixs = list(range(32,n_seqs + 1, 33))

    src = src[-1].unsqueeze(0)
    tgt = tgt[-1].unsqueeze(0)

    # TODO: no grad here necessary since its in the model generate method?
    with torch.no_grad():
        model.eval()

        print(f"Evaluating for {n_eval_iters} iters")

        desc_dfs = []
        n_generated = 0
        all_zeros = 0

        for k in range(n_eval_iters):
            encode, _ = get_vocab_encoder_decoder(part_2)
            start_ix = encode(["start"])[0]
            idy = torch.full(
                (src.shape[0], 1), start_ix, dtype=torch.long, device=device
            )
            seqs = (
                model.generate(src, idy, max_new_tokens=config["sequence_len"])
                .detach()
                .cpu()
                .numpy()
            )

            for ix, seq in enumerate(seqs):
                gen_roll = get_roll_from_sequence(seq, part=part_2)
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
    tgt,
    optimizer,
    loss_fn,
    config,
    device,
    model_name,
    model_dir,
):
    n_epochs = config["n_epochs"]
    evaluate_every = n_epochs

    train_losses = []

    model.train()

    epochs = tqdm(range(1, n_epochs + 1))

    for epoch in epochs:
        logits = model(src, tgt)

        B, T, C = logits.shape
        loss = loss_fn(logits.view(B * T, C), tgt.view(B * T))

        train_losses.append(loss.item())
        epochs.set_postfix({"loss": f"{loss.item():.4f}"})

        # Backprop
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
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
                device=device,
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
    config["batch_size"] = 1
    config["n_eval_iters"] = 3
    config["n_epochs"] = 500
    config["wandb"] = False
    config["bento"] = False
    config["checkpoints"] = False

    torch.manual_seed(config["seed"])

    dataset = PartPairDatasetSequential(**config["data"], datasets_dir=datasets_dir)
    splits = config["splits"]
    train_data, val_data, test_data = random_split(dataset, list(splits.values()))

    train_loader = DataLoader(train_data, batch_size=config["batch_size"], shuffle=True)

    vocab_sizes = get_vocab_sizes()
    encode, _ = get_vocab_encoder_decoder(config["data"]["part_2"])
    pad_ix = encode(["pad"])[0]
    config["model"]["src_vocab_size"] = vocab_sizes[config["data"]["part_1"]]
    config["model"]["tgt_vocab_size"] = vocab_sizes[config["data"]["part_2"]]
    # Add 1 to the context length to account for the start token
    config["model"]["context_len"] = config["sequence_len"] + 1
    config["model"]["pad_ix"] = pad_ix
    print(yaml.dump(config))

    model = TransformerEncoderDecoder(**config["model"]).to(DEVICE)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config["lr"],
        weight_decay=config["weight_decay"],
    )
    loss_fn = torch.nn.CrossEntropyLoss(
        reduction=config["loss_reduction"],
        ignore_index=pad_ix,
    )

    model_name = get_model_name()
    print(f"{model_name=}")

    model_dir = os.path.join(MODELS_DIR, model_name)
    if not os.path.isdir(model_dir):
        os.makedirs(model_dir)

    xb, yb = next(iter(train_loader))

    for ix in range(config["batch_size"]):
        src_roll = get_roll_from_sequence(
            xb[ix][-1].numpy(),
            part=config["data"]["part_2"],
        )
        tgt_roll = get_roll_from_sequence(
            yb[ix][-1].numpy(),
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

    x = xb.to(DEVICE).view(xb.shape[0] * xb.shape[1], xb.shape[2])
    y = yb.to(DEVICE).view(yb.shape[0] * yb.shape[1], yb.shape[2])

    overfit_single_batch(
        model=model,
        src=x,
        tgt=y,
        optimizer=optimizer,
        loss_fn=loss_fn,
        config=config,
        device=DEVICE,
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
