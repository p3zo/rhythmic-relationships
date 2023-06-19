import os
import yaml

import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
import wandb
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from rhythmic_relationships import WANDB_PROJECT_NAME
from rhythmic_relationships.data import PartPairDatasetRSP
from rhythmic_relationships.model_utils import save_checkpoint, get_loss_fn, save_model

DEFAULT_CONFIG_FILEPATH = "config.yml"

DEVICE = torch.device(
    "mps"
    if torch.backends.mps.is_built()
    else torch.device("cuda:0")
    if torch.cuda.device_count() > 0
    else torch.device("cpu")
)


def compute_loss(logits, y, loss_fn):
    return loss_fn(
        logits.view(logits.shape[0] * logits.shape[1]), y.view(y.shape[0] * y.shape[1])
    )


class RSP_FC(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Linear(input_dim, 4),
            nn.ReLU(),
            nn.Linear(4, 16),
            nn.ReLU(),
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, output_dim),
            nn.Sigmoid(),
        )

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, x):
        out = self.layers(x)
        return out


def evaluate_rsp_fc(
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
    part = config["data"]["part"]

    print(f"Evaluating for {n_eval_iters} iters")

    evals_train_loss = []
    for k in range(n_eval_iters):
        src, tgt = next(iter(train_loader))
        src = src.to(device)
        tgt = tgt.to(device)

        with torch.no_grad():
            logits = model(src)

        loss = compute_loss(logits=logits, y=tgt, loss_fn=loss_fn)
        evals_train_loss.append(loss.item())

    evals_val_loss = []
    for k in range(n_eval_iters):
        src, tgt = next(iter(val_loader))
        src = src.to(device)
        tgt = tgt.to(device)

        with torch.no_grad():
            logits = model(src)

        loss = compute_loss(logits=logits, y=tgt, loss_fn=loss_fn)
        evals_val_loss.append(loss.item())

    n_generated = 0
    all_zeros = 0
    all_same = 0

    n_seqs = 2

    src, tgt = next(iter(val_loader))

    for ix in range(n_seqs):
        seq = inference(model=model, n_tokens=32, temperature=1.2, device=device)

        gen_hits = get_hits_from_hits_seq(seq.cpu().numpy(), part=part)

        n_generated += 1
        if max(gen_hits) == 0:
            all_zeros += 1
            continue
        if len(set(gen_hits)) == 1:
            all_same += 1
            continue

        write_midi_from_hits(
            [i * 127 for i in gen_hits],
            outpath=os.path.join(eval_dir, f"{ix}_gen.mid"),
            part=part,
            pitch=72,
        )

    print(f"{n_generated=}")
    print(f"  {all_zeros=} ({100*round(all_zeros/n_generated, 2)}%)")
    print(f"  {all_same=} ({100*round(all_same/n_generated, 2)}%)")

    val_loss_mean = np.mean(evals_val_loss)
    train_loss_mean = np.mean(evals_train_loss)
    curr_eval = {
        "val_loss": val_loss_mean,
        "train_loss": train_loss_mean,
        "val_train_loss_pct_diff": pct_diff(val_loss_mean, train_loss_mean),
    }
    print(f"{curr_eval=}")

    evaluation.update(curr_eval)

    if config["wandb"]:
        wandb.log(curr_eval)

    model.train()

    return evaluation


def train_rsp_fc(
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
            src, tgt = batch
            src = src.to(device)
            tgt = tgt.to(device)

            logits = model(src)

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

            if ix % eval_interval == 0:
                val = []
                # val = evaluate_rsp_fc(
                #     train_loader=train_loader,
                #     val_loader=val_loader,
                #     model=model,
                #     config=config,
                #     epoch=epoch,
                #     loss_fn=loss_fn,
                #     model_name=model_name,
                #     model_dir=model_dir,
                #     device=device,
                # )
                evals.append(val)

            #     e_ixs = range(len(evals))
            #     eval_val_losses = [evals[i]["val_loss"] for i in e_ixs]
            #     eval_train_losses = [evals[i]["train_loss"] for i in e_ixs]
            #     marker = "o" if epoch == 1 else None
            #     plt.plot(
            #         e_ixs, eval_train_losses, label="train", c="blue", marker=marker
            #     )
            #     plt.plot(e_ixs, eval_val_losses, label="val", c="orange", marker=marker)
            #     eval_loss_plot_path = os.path.join(model_dir, "eval_loss.png")
            #     plt.legend()
            #     plt.title(f"{model_name}")
            #     plt.tight_layout()
            #     plt.savefig(eval_loss_plot_path)
            #     plt.clf()

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

    # # Final eval
    # evals.append(
    #     evaluate_rsp_fc(
    #         train_loader=train_loader,
    #         val_loader=val_loader,
    #         model=model,
    #         config=config,
    #         epoch=epoch,
    #         loss_fn=loss_fn,
    #         model_name=model_name,
    #         model_dir=model_dir,
    #         device=device,
    #     )
    # )

    return evals


def train(config, model_name, datasets_dir, model_dir):
    if config["wandb"]:
        wandb.init(project=WANDB_PROJECT_NAME, config=config, name=model_name)
        wandb.config.update(config)

    dataset = PartPairDatasetRSP(**config["data"], datasets_dir=datasets_dir)

    splits = config["splits"]
    train_data, val_data, test_data = random_split(dataset, list(splits.values()))
    for k, v in {"train": train_data, "val": val_data, "test": test_data}.items():
        ix_path = os.path.join(model_dir, f"{k}_ixs.csv")
        pd.Series(v.indices).to_csv(ix_path, index=False, header=False)
    print(f"{splits=}: {len(train_data)}, {len(val_data)}, {len(test_data)}")

    train_loader = DataLoader(train_data, batch_size=config["batch_size"], shuffle=True)
    val_loader = DataLoader(val_data, batch_size=config["batch_size"], shuffle=True)

    model = RSP_FC(**config["model"]).to(DEVICE)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config["lr"],
        weight_decay=config["weight_decay"],
    )
    loss_fn = get_loss_fn(config)

    print(yaml.dump(config))

    evals = train_rsp_fc(
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
