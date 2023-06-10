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

from rhythmic_relationships.model_utils import get_model_name, load_config, save_model, save_checkpoint, get_loss_fn
from rhythmic_relationships import DATASETS_DIR, MODELS_DIR
from rhythmic_relationships.data import (
    PartPairDataset,
    get_roll_from_sequence,
    get_hits_from_hits_seq,
)
from rhythmic_relationships.model_hits import TVAE
from rhythmic_relationships.io import write_midi_from_hits

DEFAULT_CONFIG_FILEPATH = "config_hits.yml"
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


def inference(
    src,
    model,
    n_new_tokens,
    nucleus_p,
    temperature,
):
    with torch.no_grad():
        latents = model.get_sampled_latent(
            src,
            use_sampling=False,
            sampling_var=0.0,
        )

    # TODO: get start_ix from hits vocab
    start_token = 4
    y = torch.full(
        (src.shape[0], 1),
        start_token,
        dtype=torch.long,
        device=DEVICE,
    )

    entropies = []

    for _ in range(n_new_tokens):
        # Get the predictions
        with torch.no_grad():
            logits = model.generate(y=y, latent=latents)

        # Take the logits for the last tokens
        logits = logits[:, -1, :]

        # Apply softmax to get probabilities
        probs = temperatured_softmax(logits.cpu().numpy(), temperature)

        # y_next = []
        # for ix in range(probs.shape[0]):
        #     yn = nucleus(probs[ix], nucleus_p)
        #     y_next.append(yn)
        # y_next = torch.tensor(y_next, dtype=torch.long, device=DEVICE).unsqueeze(1)

        y_next = torch.multinomial(
            torch.tensor(probs, dtype=torch.float32, device=DEVICE),
            num_samples=1,
        )

        y = torch.cat((y, y_next), dim=1)
        entropies.append(entropy(probs))

    return y, np.array(entropies)


def parse_hits_batch(batch):
    src, tgt = batch
    return src.to(DEVICE), tgt.to(DEVICE)


def compute_loss(loss_fn, dec_logits, dec_tgt, mu, logvar, beta, fb_lambda):
    # TODO: use loss_fn
    # recons_loss = loss_fn(
    #     dec_logits.view(-1, dec_logits.size(-1)),
    #     dec_tgt.contiguous().view(-1),
    # )
    recons_loss = torch.nn.functional.cross_entropy(
        dec_logits.view(-1, dec_logits.size(-1)),
        dec_tgt.contiguous().view(-1),
        reduction="mean",
    )

    kl_raw = -0.5 * (1 + logvar - mu**2 - logvar.exp()).mean(dim=0)
    kl_before_free_bits = kl_raw.mean()
    kl_after_free_bits = kl_raw.clamp(min=fb_lambda)
    kldiv_loss = kl_after_free_bits.mean()

    return {
        "beta": beta,
        "total_loss": recons_loss + beta * kldiv_loss,
        "kldiv_loss": kldiv_loss,
        "kldiv_raw": kl_before_free_bits,
        "recons_loss": recons_loss,
    }


def evaluate_hits(
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

    evals_total_loss = []
    evals_kldiv_loss = []
    evals_kldiv_raw = []
    evals_recons_loss = []

    for k in range(n_eval_iters):
        src, tgt = parse_hits_batch(next(iter(val_loader)))

        with torch.no_grad():
            mu, logvar, dec_logits = model(src, tgt)

            losses = compute_loss(
                loss_fn=loss_fn,
                dec_logits=dec_logits,
                dec_tgt=tgt,
                mu=mu,
                logvar=logvar,
                beta=config["kl_max_beta"],
                fb_lambda=config["free_bit_lambda"],
            )

            evals_total_loss.append(losses["total_loss"].item())
            evals_kldiv_loss.append(losses["kldiv_loss"].item())
            evals_kldiv_raw.append(losses["kldiv_raw"].item())
            evals_recons_loss.append(losses["recons_loss"].item())

        desc_dfs = []
        n_generated = 0
        all_zeros = 0

        seqs, entropies = inference(
            src=src,
            model=model,
            n_new_tokens=31,
            nucleus_p=0.9,
            temperature=1.2,
        )

        print(
            "[entropy] {:.4f} (+/- {:.4f})".format(
                np.mean(entropies), np.std(entropies)
            )
        )

        for ix, seq in enumerate(seqs):
            gen_hits = get_hits_from_hits_seq(seq.cpu().numpy(), part=part_2)
            tgt_hits = get_hits_from_hits_seq(tgt[ix].cpu().numpy(), part=part_2)

            # TODO: Compare descriptors of the generated and target rolls
            # gen_roll_descs = pianoroll2descriptors(
            #     gen_roll,
            #     config["resolution"],
            #     drums=part_2 == "Drums",
            # )
            # tgt_roll_descs = pianoroll2descriptors(
            #     tgt_roll,
            #     config["resolution"],
            #     drums=part_2 == "Drums",
            # )
            # df = pd.DataFrame.from_dict(
            #     {"generated": gen_roll_descs, "target": tgt_roll_descs},
            #     orient="index",
            # )
            # desc_dfs.append(df)

            n_generated += 1
            if max(gen_hits) == 0:
                all_zeros += 1
                continue

            write_midi_from_hits(
                gen_hits,
                outpath=os.path.join(eval_dir, f"{k}_{ix}_gen.mid"),
                part=part_2,
                pitch=48,
            )

            write_midi_from_hits(
                tgt_hits,
                outpath=os.path.join(eval_dir, f"{k}_{ix}_tgt.mid"),
                part=part_2,
                pitch=48,
            )

            # write_midi_from_hits(
            #     tgt_roll,
            #     outpath=os.path.join(eval_dir, f"{k}_{ix}_tgt.mid"),
            #     part=part_2,
            #     pitch=48,
            # )

            src_hits = get_hits_from_hits_seq(src[ix].cpu().numpy(), part=part_1)
            write_midi_from_hits(
                src_hits,
                outpath=os.path.join(eval_dir, f"{k}_{ix}_src.mid"),
                part=part_1,
                pitch=65,
            )

        print(f"{n_generated=}, {all_zeros=} ({100*round(all_zeros/n_generated, 2)}%)")

        # TOOD: add this back
        if len(desc_dfs) > 0:
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


def train_hits(
    model,
    train_loader,
    val_loader,
    optimizer,
    loss_fn,
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
            src, tgt = parse_hits_batch(batch)

            mu, logvar, dec_logits = model(src, tgt)

            losses = compute_loss(
                loss_fn=loss_fn,
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
                val = evaluate_hits(
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

    # Final eval
    evals.append(
        evaluate_hits(
            val_loader=val_loader,
            model=model,
            config=config,
            epoch=epoch,
            model_name=model_name,
            model_dir=model_dir,
        )
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

    dataset = PartPairDataset(**config["data"], datasets_dir=datasets_dir)
    splits = config["splits"]
    train_data, val_data, test_data = random_split(dataset, list(splits.values()))
    for k, v in {"train": train_data, "val": val_data, "test": test_data}.items():
        ix_path = os.path.join(model_dir, f"{k}_ixs.csv")
        pd.Series(v.indices).to_csv(ix_path, index=False, header=False)
    print(f"{splits=}: {len(train_data)}, {len(val_data)}, {len(test_data)}")

    train_loader = DataLoader(train_data, batch_size=config["batch_size"], shuffle=True)
    val_loader = DataLoader(val_data, batch_size=config["batch_size"], shuffle=True)

    # TODO: add vocab utils for hits repr
    config["model"]["src_vocab_size"] = 5
    config["model"]["tgt_vocab_size"] = 5
    # Add 1 to the context length to account for the start token
    config["model"]["context_len"] = config["sequence_len"] + 1
    print(yaml.dump(config))

    model = TVAE(**config["model"]).to(DEVICE)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config["lr"],
        weight_decay=config["weight_decay"],
    )
    loss_fn = get_loss_fn(config)

    if config["wandb"]:
        wandb.init(project=WANDB_PROJECT_NAME, config=config, name=model_name)
        wandb.config.update(config)

    evals = train_hits(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        loss_fn=loss_fn,
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
