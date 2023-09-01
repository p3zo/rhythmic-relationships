import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import wandb
import yaml
from rhythmic_relationships import DATASETS_DIR, MODELS_DIR, WANDB_PROJECT_NAME
from rhythmic_relationships.data import (
    PartPairDataset,
    get_hits_from_hits_seq,
    get_roll_from_sequence,
)
from rhythmic_relationships.evaluate import (
    compute_oa_kld_dists,
    compute_oa_and_kld,
    make_oa_kld_plot,
    mk_descriptor_dist_plot,
)
from rhythmic_relationships.io import (
    get_roll_from_hits,
    write_midi_from_hits,
    write_midi_from_roll,
)
from rhythmic_relationships.model_utils import (
    get_loss_fn,
    get_model_name,
    load_config,
    save_checkpoint,
    save_model,
)
from rhythmic_relationships.models.hits_encdec import TransformerEncoderDecoder
from rhythmic_relationships.vocab import (
    get_hits_vocab,
    get_hits_vocab_size,
    get_vocab_sizes,
)
from rhythmtoolbox import pianoroll2descriptors
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

DEFAULT_CONFIG_FILEPATH = "config.yml"

DEVICE = torch.device(
    "mps"
    if torch.backends.mps.is_built()
    else torch.device("cuda:0")
    if torch.cuda.device_count() > 0
    else torch.device("cpu")
)


def parse_batch(batch, device):
    xb, yb = batch
    yb_shifted = torch.roll(yb, 1)
    yb_shifted[:, 0] = torch.zeros((yb.shape[0],))
    return xb.to(device), yb_shifted.to(device), yb.to(device)


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
    part_1_pitch=55,
    part_2_pitch=72,
    resolution=4,
    train_df=None,
    train_dist=None,
):
    gen_dir = os.path.join(eval_dir, "inference")
    if not os.path.isdir(gen_dir):
        os.makedirs(gen_dir)

    print(f"{sampler=}")

    drop_cols = ["noi", "polyDensity", "syness"]

    part_1 = config["data"]["part_1"]
    part_2 = config["data"]["part_2"]
    block_size = config["data"]["block_size"]

    sampler_eval = {}
    generated_rolls = []
    generated_descs = []
    target_descs = []
    all_zeros = 0
    all_same = 0

    for ix in range(n_seqs):
        print(ix)
        src = srcs[ix].unsqueeze(0)
        tgt = tgts[ix]

        seq = inference(
            model=model,
            src=src,
            n_tokens=config["model"]["context_len"],
            temperature=temperature,
            device=device,
            sampler=sampler,
            nucleus_p=nucleus_p,
        )

        if part_2 == "Drums":
            gen_roll = get_roll_from_sequence(seq.cpu().numpy(), part=part_2)

            if gen_roll.max() == 0:
                all_zeros += 1
                continue
            # TODO: check all_same

            tgt_roll = get_roll_from_sequence(tgt.cpu().numpy(), part=part_2)

            write_midi_from_roll(
                gen_roll,
                outpath=os.path.join(gen_dir, f"{ix}_{sampler}_gen.mid"),
                part=part_2,
                binary=False,
                scaled=True,
                onset_roll=True,
            )
            write_midi_from_roll(
                tgt_roll,
                outpath=os.path.join(gen_dir, f"{ix}_{sampler}_tgt.mid"),
                part=part_2,
                binary=False,
                scaled=True,
                onset_roll=True,
            )
        else:
            gen_hits = get_hits_from_hits_seq(
                seq.cpu().numpy(),
                block_size=block_size,
                part=part_2,
                verbose=True,
            )
            gen_hits = [i * 127.0 for i in gen_hits]

            if max(gen_hits) == 0:
                all_zeros += 1
                continue
            if len(set(gen_hits)) == 1:
                all_same += 1
                continue

            write_midi_from_hits(
                gen_hits,
                outpath=os.path.join(gen_dir, f"{ix}_{sampler}_gen.mid"),
                part=part_2,
                pitch=part_2_pitch,
            )

            gen_roll = get_roll_from_hits(
                gen_hits,
                pitch=part_2_pitch,
                resolution=resolution,
            )
            tgt_hits = get_hits_from_hits_seq(
                tgt.cpu().numpy(),
                block_size=block_size,
                part=part_2,
            )
            tgt_hits = [i * 127.0 for i in tgt_hits]
            write_midi_from_hits(
                tgt_hits,
                outpath=os.path.join(gen_dir, f"{ix}_{sampler}_tgt.mid"),
                part=part_2,
                pitch=part_2_pitch,
            )

            tgt_roll = get_roll_from_hits(
                tgt_hits,
                pitch=part_2_pitch,
                resolution=resolution,
            )

        generated_rolls.append(gen_roll)

        gen_descs = pianoroll2descriptors(
            gen_roll,
            resolution,
            drums=part_2 == "Drums",
        )
        generated_descs.append(gen_descs)

        tgt_descs = pianoroll2descriptors(
            tgt_roll,
            resolution,
            drums=part_2 == "Drums",
        )
        target_descs.append(tgt_descs)

        src_hits = get_hits_from_hits_seq(
            src.squeeze(0).cpu().numpy(),
            block_size=block_size,
            part=part_1,
        )
        src_hits = [i * 127.0 for i in src_hits]
        write_midi_from_hits(
            src_hits,
            outpath=os.path.join(gen_dir, f"{ix}_{sampler}_src.mid"),
            part=part_1,
            pitch=part_1_pitch,
        )

    target_df = pd.DataFrame(target_descs).dropna(how="all", axis=1)
    target_df.drop(drop_cols, axis=1, inplace=True)

    oa_klds = {}
    if n_seqs > all_zeros:
        gen_df = pd.DataFrame(generated_descs).dropna(how="all", axis=1)
        gen_df.drop(drop_cols, axis=1, inplace=True)

        title_suffix = f"\n{temperature=}"
        if sampler == "nucleus":
            title_suffix += f" {nucleus_p=}"

        labels = ["ref"]
        if train_df is not None:
            labels.append("train")
            mk_descriptor_dist_plot(
                gen_df=gen_df,
                ref_df=train_df,
                model_name=model_name,
                outdir=eval_dir,
                label="Train",
                title_suffix=title_suffix,
                filename_suffix=f"{sampler}_train_gen",
            )

        mk_descriptor_dist_plot(
            gen_df=gen_df,
            ref_df=target_df,
            model_name=model_name,
            outdir=eval_dir,
            label="Target",
            title_suffix=title_suffix,
            filename_suffix=f"{sampler}_ref_gen",
        )

        # Compute distribution comparison metrics
        oa_kld_dists = compute_oa_kld_dists(
            gen_df=gen_df,
            ref_df=target_df,
            train_df=train_df,
            train_dist=train_dist,
        )

        oa_klds = compute_oa_and_kld(oa_kld_dists)
        print(oa_klds)

        for label in labels:
            make_oa_kld_plot(
                dist_1=oa_kld_dists[f"{label}_dist"],
                dist_2=oa_kld_dists[f"{label}_gen_dist"],
                oa=oa_klds[f"{label}_gen_oa"],
                kld=oa_klds[f"{label}_gen_kld"],
                label=f"{label}",
                model_name=model_name,
                outdir=eval_dir,
                suffix=sampler,
            )

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


def eval_gen_hits_encdec(
    model,
    config,
    loader,
    eval_dir,
    n_seqs,
    model_name,
    device,
    temperature=1.0,
    nucleus_p=0.92,
    part_1_pitch=55,
    part_2_pitch=72,
    resolution=4,
    samplers=("multinomial", "nucleus", "greedy"),
    train_df=None,
    train_dist=None,
):
    print(f"Generating {n_seqs} eval sequences")

    gen_eval = {}

    print(f"Loading {n_seqs} inference sequences...")
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
            part_1_pitch=part_1_pitch,
            part_2_pitch=part_2_pitch,
            resolution=resolution,
            train_df=train_df,
            train_dist=train_dist,
        )

    return gen_eval


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

    gen_eval = eval_gen_hits_encdec(
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
    eval_interval = config["eval_interval"]

    evals = []
    train_losses = []

    model.train()

    ix = 0
    for epoch in range(1, n_epochs + 1):
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
            if config["clip_gradients"]:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            loss.backward()
            optimizer.step()

            # Save loss after each batch
            plt.plot(train_losses)
            loss_plot_path = os.path.join(model_dir, "loss.png")
            plt.tight_layout()
            plt.savefig(loss_plot_path)
            plt.clf()

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


def train(config, model_name, datasets_dir, model_dir, sweep=False):
    if config["wandb"]:
        wandb.init(project=WANDB_PROJECT_NAME, config=config, name=model_name)
        wandb.config.update(config)
        if sweep:
            config = wandb.config

    assert config["n_eval_seqs"] >= 3

    dataset = PartPairDataset(
        **config["data"],
        datasets_dir=datasets_dir,
        tokenize_rolls=True,
    )

    splits = config["splits"]
    train_data, val_data, test_data = random_split(dataset, list(splits.values()))
    for k, v in {"train": train_data, "val": val_data, "test": test_data}.items():
        ix_path = os.path.join(model_dir, f"{k}_ixs.csv")
        pd.Series(v.indices).to_csv(ix_path, index=False, header=False)
    print(f"{splits=}: {len(train_data)}, {len(val_data)}, {len(test_data)}")

    train_loader = DataLoader(train_data, batch_size=config["batch_size"], shuffle=True)
    val_loader = DataLoader(val_data, batch_size=config["batch_size"], shuffle=True)

    # hits_vocab = get_hits_vocab()
    # ttoi = {v: k for k, v in hits_vocab.items()}
    # start_ix = ttoi["start"]
    hits_vocab_size = get_hits_vocab_size(block_size=config["data"]["block_size"])

    drums_vocab_size = get_vocab_sizes()["Drums"]

    src_vocab_size = hits_vocab_size
    tgt_vocab_size = hits_vocab_size
    if config["data"]["part_1"] == "Drums":
        src_vocab_size = drums_vocab_size
    if config["data"]["part_2"] == "Drums":
        tgt_vocab_size = drums_vocab_size

    config["model"]["src_vocab_size"] = src_vocab_size
    config["model"]["tgt_vocab_size"] = tgt_vocab_size
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
