"""Generate samples and compare them to the training set"""
import os

import numpy as np
import pandas as pd
import torch
import wandb
from rhythmic_relationships import CHECKPOINTS_DIRNAME, MODELS_DIR
from rhythmic_relationships.data import (
    PartDataset,
    get_roll_from_sequence,
)
from rhythmic_relationships.evaluate import (
    compute_kld,
    compute_oa,
    get_oa_kld_dists,
    make_oa_kld_plot,
    mk_descriptor_dist_plot,
    nucleus,
    temperatured_softmax,
)
from rhythmic_relationships.io import get_roll_from_hits, write_midi_from_roll
from rhythmic_relationships.model_utils import load_model
from rhythmic_relationships.models.decoder import TransformerDecoder
from rhythmic_relationships.vocab import get_hits_vocab
from rhythmtoolbox import pianoroll2descriptors
from torch.utils.data import DataLoader
from utils import compute_loss, parse_batch

DEVICE = torch.device(
    "mps"
    if torch.backends.mps.is_built()
    else torch.device("cuda:0")
    if torch.cuda.device_count() > 0
    else torch.device("cpu")
)


def inference(
    model,
    n_tokens,
    temperature,
    device,
    sampler="multinomial",
    nucleus_p=0.92,
):
    if sampler not in ["multinomial", "nucleus", "greedy"]:
        raise ValueError(f"Unsupported {sampler}: sampler")

    hits_vocab = get_hits_vocab()
    ttoi = {v: k for k, v in hits_vocab.items()}
    start_ix = ttoi["start"]

    y = torch.tensor(
        [[start_ix]],
        dtype=torch.long,
        requires_grad=False,
        device=device,
    )

    for ix in range(n_tokens):
        # Get the predictions
        with torch.no_grad():
            logits = model(y)

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
        elif sampler == "multinomial":
            y_next = torch.multinomial(
                torch.tensor(probs, dtype=torch.float32, device=DEVICE),
                num_samples=1,
            )
        else:
            y_next = torch.tensor(
                [probs.argmax()], dtype=torch.long, device=DEVICE
            ).unsqueeze(1)

        y = torch.cat([y, y_next], dim=1)

    return y.squeeze(0)[1:]


def pct_diff(x, y):
    return 100 * np.abs(x - y) / ((x + y) / 2)


def mk_oa_kld_plots(
    descriptor_dists,
    label,
    model_name,
    eval_dir,
    sampler,
):
    print(f"{label} oa_kld_dists")
    for descriptor in descriptor_dists:
        if descriptor != "all_descriptors":
            print("TODO remove this. skipping", descriptor)
            continue
        print(f"{descriptor=}")
        dist_1 = descriptor_dists[descriptor]["ref_dist"]
        dist_2 = descriptor_dists[descriptor]["ref_gen_dist"]

        oa = compute_oa(dist_1, dist_2)
        kld = compute_kld(dist_1, dist_2)

        make_oa_kld_plot(
            dist_1=dist_1,
            dist_2=dist_2,
            oa=oa,
            kld=kld,
            label=f"{label}_{descriptor}",
            model_name=model_name,
            outdir=eval_dir,
            suffix=sampler,
            descriptor=descriptor,
        )


def get_sampler_eval(
    sampler,
    n_seqs,
    model,
    device,
    eval_dir,
    config,
    model_name,
    temperature=1.0,
    nucleus_p=0.92,
    resolution=4,
    train_df=None,
):
    gen_dir = os.path.join(eval_dir, "inference")
    if not os.path.isdir(gen_dir):
        os.makedirs(gen_dir)

    print(f"{sampler=}")

    # drop_cols = ["noi", "polyDensity", "syness"]

    part = config["data"]["part"]

    sampler_eval = {}
    generated_rolls = []
    generated_descs = []
    target_descs = []
    all_zeros = 0
    all_same = 0

    for ix in range(n_seqs):
        seq = inference(
            model=model,
            n_tokens=config["model"]["context_len"],
            temperature=temperature,
            device=device,
            sampler=sampler,
            nucleus_p=nucleus_p,
        )

        gen_roll = get_roll_from_sequence(seq.cpu().numpy(), part=part) * 127

        # TODO: adapt to roll
        # if max(gen_roll) == 0:
        #     all_zeros += 1
        #     continue

        # if len(set(gen_roll)) == 1:
        #     all_same += 1
        #     continue

        generated_rolls.append(gen_roll)
        write_midi_from_roll(
            gen_roll,
            outpath=os.path.join(gen_dir, f"{ix}_{sampler}_gen.mid"),
            part=part,
            onset_roll=True,
        )

        gen_descs = pianoroll2descriptors(
            gen_roll,
            resolution,
            drums=part == "Drums",
        )
        generated_descs.append(gen_descs)

    target_df = pd.DataFrame(target_descs).dropna(how="all", axis=1)
    # target_df.drop(drop_cols, axis=1, inplace=True)

    oa_klds = {}
    if n_seqs == all_zeros:
        return sampler_eval

    gen_df = pd.DataFrame(generated_descs).dropna(how="all", axis=1)
    # gen_df.drop(drop_cols, axis=1, inplace=True)

    title_suffix = f"\n{temperature=}"
    if sampler == "nucleus":
        title_suffix += f" {nucleus_p=}"

    train_gen_df = pd.concat((train_df, gen_df))
    target_gen_df = pd.concat((target_df, gen_df))

    for col in list(target_df.columns):
        print(
            "target", col, target_df[col].mean().round(3), target_df[col].std().round(3)
        )
        print(
            "target gen",
            col,
            target_gen_df[col].mean().round(3),
            target_gen_df[col].std().round(3),
        )
        print("train", col, train_df[col].mean().round(3), train_df[col].std().round(3))
        print(
            "train_gen",
            col,
            train_gen_df[col].mean().round(3),
            train_gen_df[col].std().round(3),
        )

    # mk_descriptor_dist_plot(
    #     gen_df=train_df,
    #     ref_df=gen_df,
    #     model_name=model_name,
    #     outdir=eval_dir,
    #     label="Train",
    #     title_suffix=title_suffix,
    #     filename_suffix=f"rel_{sampler}_train_vs_gen",
    # )
    #
    # mk_descriptor_dist_plot(
    #     gen_df=target_df,
    #     ref_df=gen_df,
    #     model_name=model_name,
    #     outdir=eval_dir,
    #     label="Target",
    #     title_suffix=title_suffix,
    #     filename_suffix=f"rel_{sampler}_target_vs_gen",
    # )
    #
    # if train_df is not None:
    #     mk_descriptor_dist_plot(
    #         gen_df=gen_df,
    #         ref_df=train_df,
    #         model_name=model_name,
    #         outdir=eval_dir,
    #         label="Train",
    #         title_suffix=title_suffix,
    #         filename_suffix=f"{sampler}_train_gen",
    #     )
    #
    #     train_oa_kld_dists = get_oa_kld_dists(gen_df=gen_df, ref_df=train_df)
    #     mk_oa_kld_plots(
    #         train_oa_kld_dists,
    #         label="train",
    #         model_name=model_name,
    #         eval_dir=eval_dir,
    #         sampler=sampler,
    #     )

    sample_stats = {
        "pct_all_zero": 100 * round(all_zeros / n_seqs, 2),
        "pct_all_same": 100 * round(all_same / n_seqs, 2),
        "oa_klds": oa_klds,
    }
    print(sample_stats)

    sampler_eval["sampler_stats"] = sample_stats
    # sampler_eval["generated_rolls"] = generated_rolls
    # sampler_eval["generated_descs"] = generated_descs
    # sampler_eval["target_descs"] = target_descs

    return sampler_eval


def eval_chroma(
    model,
    config,
    loader,
    eval_dir,
    n_seqs,
    model_name,
    device,
    temperature=1.0,
    nucleus_p=0.92,
    resolution=4,
    samplers=("multinomial", "nucleus", "greedy"),
    train_df=None,
):
    print(f"Generating {n_seqs} eval sequences")

    gen_eval = {}

    for sampler in samplers:
        gen_eval["sampler_stats"] = {}
        gen_eval["sampler_stats"][sampler] = get_sampler_eval(
            sampler=sampler,
            n_seqs=n_seqs,
            model=model,
            device=device,
            eval_dir=eval_dir,
            config=config,
            model_name=model_name,
            temperature=temperature,
            nucleus_p=nucleus_p,
            resolution=resolution,
            train_df=train_df,
        )

    return gen_eval


def evaluate_chroma(
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

    print(f"Evaluating train loss for {n_eval_iters} iters")

    evals_train_loss = []
    for _ in range(n_eval_iters):
        src, tgt = parse_batch(next(iter(train_loader)), device)
        with torch.no_grad():
            logits = model(src)
            loss = compute_loss(logits=logits, y=tgt, loss_fn=loss_fn)
            evals_train_loss.append(loss.item())

    print(f"Evaluating val loss for {n_eval_iters} iters")

    evals_val_loss = []
    for _ in range(n_eval_iters):
        src, tgt = parse_batch(next(iter(val_loader)), device)
        with torch.no_grad():
            logits = model(src)
            loss = compute_loss(logits=logits, y=tgt, loss_fn=loss_fn)
            evals_val_loss.append(loss.item())

    gen_eval = eval_chroma(
        model=model,
        config=config,
        loader=val_loader,
        eval_dir=eval_dir,
        n_seqs=config["n_eval_seqs"],
        model_name=model_name,
        device=device,
        samplers=["nucleus", "multinomial"],
    )

    val_loss_mean = np.mean(evals_val_loss)
    train_loss_mean = np.mean(evals_train_loss)
    curr_eval = {
        "val_loss": val_loss_mean,
        "train_loss": train_loss_mean,
        "val_train_loss_pct_diff": pct_diff(val_loss_mean, train_loss_mean),
        # "sampler_stats": gen_eval["sampler_stats"],
    }
    print(f"{curr_eval=}")

    evaluation.update(curr_eval)

    if config["wandb"]:
        wandb.log(curr_eval)

    model.train()

    return evaluation
