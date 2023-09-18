"""Generate samples and compare them to the training set"""
import os

import numpy as np
import pandas as pd
import torch
import wandb
from rhythmic_relationships import CHECKPOINTS_DIRNAME, MODELS_DIR
from rhythmic_relationships.data import (
    PartDataset,
    PartPairDataset,
    get_roll_from_sequence,
)
from rhythmic_relationships.evaluate import (
    compute_kld,
    compute_oa,
    get_oa_kld_dists,
    make_oa_kld_plot,
    mk_descriptor_dist_plot,
    temperatured_softmax,
    nucleus
)
from rhythmic_relationships.io import write_midi_from_roll
from rhythmic_relationships.model_utils import load_model
from rhythmic_relationships.models.hits_encdec import TransformerEncoderDecoder
from rhythmic_relationships.vocab import get_hits_vocab
from rhythmtoolbox import pianoroll2descriptors
from torch.utils.data import DataLoader

from .utils import compute_loss, parse_batch

DEVICE = torch.device(
    "mps"
    if torch.backends.mps.is_built()
    else torch.device("cuda:0")
    if torch.cuda.device_count() > 0
    else torch.device("cpu")
)


def inference(
    model,
    src,
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
    srcs,
    tgts,
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

    drop_cols = ["noi", "polyDensity", "syness"]

    part_1 = config["data"]["part_1"]
    part_2 = config["data"]["part_2"]
    # block_size = config["data"]["block_size"]

    sampler_eval = {}
    generated_rolls = []
    generated_descs = []
    target_descs = []
    # TODO: re-establish these checks
    # all_zeros = 0
    # all_same = 0

    for ix in range(n_seqs):
        src = srcs[ix].unsqueeze(0)

        seq = inference(
            model=model,
            src=src,
            n_tokens=config["model"]["context_len"],
            temperature=temperature,
            device=device,
            sampler=sampler,
            nucleus_p=nucleus_p,
        )

        gen_roll = get_roll_from_sequence(seq.cpu().numpy(), part=part_2)

        # if gen_roll.max() == 0:
        #     all_zeros += 1
        #     continue
        generated_rolls.append(gen_roll)

        gen_descs = pianoroll2descriptors(
            gen_roll,
            resolution,
            drums=part_2 == "Drums",
        )
        generated_descs.append(gen_descs)

        write_midi_from_roll(
            gen_roll,
            outpath=os.path.join(gen_dir, f"{ix}_{sampler}_gen.mid"),
            part=part_2,
        )

        tgt_roll = get_roll_from_sequence(tgts[ix].cpu().numpy(), part=part_2)
        write_midi_from_roll(
            tgt_roll,
            outpath=os.path.join(gen_dir, f"{ix}_{sampler}_tgt.mid"),
            part=part_2,
        )

        tgt_descs = pianoroll2descriptors(
            tgt_roll,
            resolution,
            drums=part_2 == "Drums",
        )
        target_descs.append(tgt_descs)

        src_roll = get_roll_from_sequence(src.squeeze(0).cpu().numpy(), part=part_2)
        write_midi_from_roll(
            src_roll,
            outpath=os.path.join(gen_dir, f"{ix}_{sampler}_src.mid"),
            part=part_1,
        )

    target_df = pd.DataFrame(target_descs).dropna(how="all", axis=1)
    target_df.drop(drop_cols, axis=1, inplace=True)

    # TODO: remove after adapting oa_kld eval to rolls
    if True:
        return sampler_eval

    oa_klds = {}
    # if n_seqs == all_zeros:
    #     return sampler_eval

    gen_df = pd.DataFrame(generated_descs).dropna(how="all", axis=1)
    gen_df.drop(drop_cols, axis=1, inplace=True)

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

    mk_descriptor_dist_plot(
        gen_df=train_df,
        ref_df=gen_df,
        model_name=model_name,
        outdir=eval_dir,
        label="Train",
        title_suffix=title_suffix,
        filename_suffix=f"rel_{sampler}_train_vs_gen",
    )

    mk_descriptor_dist_plot(
        gen_df=target_df,
        ref_df=gen_df,
        model_name=model_name,
        outdir=eval_dir,
        label="Target",
        title_suffix=title_suffix,
        filename_suffix=f"rel_{sampler}_target_vs_gen",
    )

    # TODO: make pair descriptor plots

    if train_df is not None:
        mk_descriptor_dist_plot(
            gen_df=gen_df,
            ref_df=train_df,
            model_name=model_name,
            outdir=eval_dir,
            label="Train",
            title_suffix=title_suffix,
            filename_suffix=f"{sampler}_train_gen",
        )

        train_oa_kld_dists = get_oa_kld_dists(gen_df=gen_df, ref_df=train_df)
        mk_oa_kld_plots(
            train_oa_kld_dists,
            label="train",
            model_name=model_name,
            eval_dir=eval_dir,
            sampler=sampler,
        )

    sample_stats = {
        # "pct_all_zero": 100 * round(all_zeros / n_seqs, 2),
        # "pct_all_same": 100 * round(all_same / n_seqs, 2),
        "oa_klds": oa_klds,
    }
    print(sample_stats)

    sampler_eval["sampler_stats"] = sample_stats
    # sampler_eval["generated_rolls"] = generated_rolls
    # sampler_eval["generated_descs"] = generated_descs
    # sampler_eval["target_descs"] = target_descs

    return sampler_eval


def eval_gen_encdec(
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

    print(f"Loading {n_seqs} sequences for inference...")
    gen_srcs, _, gen_tgts = [], [], []
    while len(gen_srcs) < n_seqs:
        gs, gx, gt = parse_batch(next(iter(loader)), device=device)
        gen_srcs.extend(gs)
        gen_tgts.extend(gt)
    gen_srcs = torch.stack(gen_srcs)
    gen_tgts = torch.stack(gen_tgts)

    for sampler in samplers:
        gen_eval["sampler_stats"] = {}
        gen_eval["sampler_stats"][sampler] = get_sampler_eval(
            srcs=gen_srcs,
            tgts=gen_tgts,
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


def evaluate_encdec(
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
        src, ctx, tgt = parse_batch(next(iter(train_loader)), device)
        with torch.no_grad():
            logits = model(src, ctx)
            loss = compute_loss(logits=logits, y=tgt, loss_fn=loss_fn)
            evals_train_loss.append(loss.item())

    print(f"Evaluating val loss for {n_eval_iters} iters")

    evals_val_loss = []
    for _ in range(n_eval_iters):
        src, ctx, tgt = parse_batch(next(iter(val_loader)), device)
        with torch.no_grad():
            logits = model(src, ctx)
            loss = compute_loss(logits=logits, y=tgt, loss_fn=loss_fn)
            evals_val_loss.append(loss.item())

    gen_eval = eval_gen_encdec(
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
        "sampler_stats": gen_eval["sampler_stats"],
    }
    print(f"{curr_eval=}")

    evaluation.update(curr_eval)

    if config["wandb"]:
        wandb.log(curr_eval)

    model.train()

    return evaluation


if __name__ == "__main__":
    model_type = "encdec"

    # Melody -> Bass
    # model_name = "fragmental_2306210056"

    # Bass -> Melody
    model_name = "literation_2307011858"
    # model_name = "dematerialize_2307012124"

    checkpoint_num = None

    n_training_obs = 10000
    n_eval_seqs = 100
    resolution = 4
    temperature = 1
    nucleus_p = 0.92
    samplers = ["nucleus"]

    model_dir = os.path.join(MODELS_DIR, model_type, model_name)

    eval_dir = os.path.join(model_dir, "eval")
    if not os.path.isdir(eval_dir):
        os.makedirs(eval_dir)

    gen_dir = os.path.join(model_dir, "inference")
    if not os.path.isdir(gen_dir):
        os.makedirs(gen_dir)

    device = DEVICE

    if checkpoint_num:
        checkpoints_dir = os.path.join(model_dir, CHECKPOINTS_DIRNAME)
        model_path = os.path.join(checkpoints_dir, str(checkpoint_num))
    else:
        model_path = os.path.join(model_dir, "model.pt")

    model, config = load_model(model_path, TransformerEncoderDecoder)
    model = model.to(DEVICE)
    part_1 = config["data"]["part_1"]
    part_2 = config["data"]["part_2"]
    drop_cols = ["noi", "polyDensity", "syness"]

    # Get distribution from training set
    full_df = PartDataset(
        dataset_name=config["data"]["dataset_name"],
        part=part_2,
        representation="descriptors",
    ).as_df(subset=n_training_obs)
    dataset_df = full_df.drop(
        ["filename", "segment_id"] + drop_cols,
        axis=1,
    ).dropna(how="all", axis=1)

    # Load data for inference
    # TODO: load only data from val split
    dataset = PartPairDataset(**config["data"])
    loader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=True)

    # TODO: option to load previous samples
    sampled = eval_gen_encdec(
        model=model,
        config=config,
        loader=loader,
        n_seqs=n_eval_seqs,
        eval_dir=eval_dir,
        model_name=model_name,
        device=device,
        train_df=dataset_df,
        samplers=samplers,
    )
