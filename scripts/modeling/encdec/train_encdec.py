import argparse
import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch
import wandb
import yaml
from rhythmic_relationships import DATASETS_DIR, MODELS_DIR
from rhythmic_relationships.data import (
    PartPairDatasetSequential,
    get_roll_from_sequence,
)
from rhythmic_relationships.io import write_midi_from_roll
from rhythmic_relationships.model import TransformerEncoderDecoder
from rhythmic_relationships.model_utils import (
    get_model_name,
    load_config,
    save_bento_model,
    save_checkpoint,
    save_model,
)
from rhythmic_relationships.vocab import get_vocab_encoder_decoder, get_vocab_sizes
from rhythmtoolbox import pianoroll2descriptors
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

DEFAULT_CONFIG_FILEPATH = "config_encdec.yml"
WANDB_PROJECT_NAME = "rhythmic-relationships"

DEVICE = torch.device(
    "mps"
    if torch.backends.mps.is_built()
    else torch.device("cuda:0")
    if torch.cuda.device_count() > 0
    else torch.device("cpu")
)


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
    epoch,
    model_name,
    model_dir,
):
    eval_dir = os.path.join(model_dir, "eval", f"epoch_{epoch}")
    if not os.path.isdir(eval_dir):
        os.makedirs(eval_dir)

    n_eval_iters = config["n_eval_iters"]
    part_1 = config["data"]["part_1"]
    part_2 = config["data"]["part_2"]

    evaluation = {}

    with torch.no_grad():
        model.eval()

        print(f"Evaluating for {n_eval_iters} iters")

        eval_train_losses = torch.zeros(n_eval_iters)
        for k in range(n_eval_iters):
            srcs, tgts = parse_sequential_batch(next(iter(train_loader)), device)
            logits = model(srcs, tgts)
            loss = compute_loss(logits, tgts, loss_fn)
            eval_train_losses[k] = loss.item()

        desc_dfs = []
        eval_val_losses = torch.zeros(n_eval_iters)
        for k in range(n_eval_iters):
            srcs, tgts = parse_sequential_batch(next(iter(val_loader)), device)
            logits = model(srcs, tgts)
            loss = compute_loss(logits, tgts, loss_fn)
            eval_val_losses[k] = loss.item()

            # Generate new sequences using part_1s from the val set and just a start token for part_2
            encode, _ = get_vocab_encoder_decoder(part_2)
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

            for ix, seq in enumerate(seqs):
                gen_roll = get_roll_from_sequence(seq, part=part_2)
                write_midi_from_roll(
                    gen_roll,
                    outpath=os.path.join(eval_dir, f"{k}_{ix}_gen.mid"),
                    part=part_2,
                    binary=False,
                    onset_roll=True,
                )

                src_roll = get_roll_from_sequence(srcs[ix].cpu().numpy(), part=part_1)
                write_midi_from_roll(
                    src_roll,
                    outpath=os.path.join(eval_dir, f"{k}_{ix}_src.mid"),
                    part=part_1,
                    binary=False,
                    onset_roll=True,
                )

                tgt_roll = get_roll_from_sequence(tgts[ix].cpu().numpy(), part=part_2)
                write_midi_from_roll(
                    tgt_roll,
                    outpath=os.path.join(eval_dir, f"{k}_{ix}_tgt.mid"),
                    part=part_2,
                    binary=False,
                    onset_roll=True,
                )

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
        plt.title(f"{model_name}\nEpoch {epoch}, n={n_eval_iters}")
        plt.legend(
            loc="upper center",
            bbox_to_anchor=(0.9, 1.2),
            fancybox=True,
            shadow=False,
            ncol=1,
        )
        plt.tight_layout()
        plt.savefig(os.path.join(eval_dir, "gen_tgt_comparison.png"))
        plt.clf()

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
                    "epoch": epoch,
                }
            )

    model.train()

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
    model_dir,
):
    num_epochs = config["num_epochs"]

    train_losses = []
    epoch_evals = []

    model.train()

    for epoch in range(1, num_epochs + 1):
        batches = tqdm(train_loader)
        for batch in batches:
            # Forward pass
            srcs, tgts = parse_sequential_batch(batch, device)

            logits = model(srcs, tgts)

            # Compute loss
            loss = compute_loss(logits, tgts, loss_fn)
            train_losses.append(loss.item())
            if config["wandb"]:
                wandb.log({"batch_loss": loss.item()})

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
            epoch=epoch,
            model_name=model_name,
            model_dir=model_dir,
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

        if config["checkpoints"]:
            save_checkpoint(
                model_dir=model_dir,
                epoch=epoch,
                model=model,
                optimizer=optimizer,
                loss=loss,
                config=config,
                evals=epoch_evals,
                delete_prev=True,
            )

    return epoch_evals


if __name__ == "__main__":
    print(f"{DEVICE=}")

    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets_dir", type=str, default=DATASETS_DIR)
    parser.add_argument("--config_path", type=str, default=DEFAULT_CONFIG_FILEPATH)
    args = parser.parse_args()

    datasets_dir = args.datasets_dir
    config = load_config(args.config_path)
    print(yaml.dump(config))

    torch.manual_seed(config["seed"])

    model_name = get_model_name()
    print(f"{model_name=}")

    model_dir = os.path.join(MODELS_DIR, model_name)
    if not os.path.isdir(model_dir):
        os.makedirs(model_dir)

    dataset = PartPairDatasetSequential(**config["data"], datasets_dir=datasets_dir)
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
    pad_ix = encode(["pad"])[0]
    config["model"]["src_vocab_size"] = vocab_sizes[config["data"]["part_1"]]
    config["model"]["tgt_vocab_size"] = vocab_sizes[config["data"]["part_2"]]
    # Add 1 to the context length to account for the start token
    config["model"]["context_len"] = config["sequence_len"] + 1
    config["model"]["pad_ix"] = pad_ix
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

    if config["wandb"]:
        wandb.init(project=WANDB_PROJECT_NAME, config=config, name=model_name)
        wandb.config.update(config)

    epoch_evals = train_transformer_encoder_decoder(
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

    model_path = os.path.join(model_dir, "model.pt")
    save_model(
        model_path=model_path,
        model=model,
        config=config,
        model_name=model_name,
        evals=epoch_evals,
    )
    if config["bento"]:
        save_bento_model(model=model, model_name=model_name)
