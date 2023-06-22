"""Compare training set descriptors to generated after each epoch"""

import itertools
import os

import numpy as np
import pandas as pd
import torch

from rhythmic_relationships import MODELS_DIR, CHECKPOINTS_DIRNAME
from rhythmic_relationships.data import PartDataset
from rhythmic_relationships.evaluate import (
    compute_oa_and_kld,
    get_flat_nonzero_dissimilarity_matrix,
    make_oa_kld_plot,
    mk_descriptor_dist_plot,
)
from rhythmic_relationships.io import write_midi_from_hits, get_roll_from_hits
from rhythmic_relationships.model_utils import load_model
from rhythmtoolbox import pianoroll2descriptors

from train import RSP_FC

DEVICE = torch.device(
    "mps"
    if torch.backends.mps.is_built()
    else torch.device("cuda:0")
    if torch.cuda.device_count() > 0
    else torch.device("cpu")
)


if __name__ == "__main__":
    model_type = "rsp_fc"
    model_name = "pharmacomaniac_2306200028"
    checkpoint_num = None

    n_training_obs = 1000
    pitch = 72
    resolution = 4

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

    model, config = load_model(model_path, RSP_FC)
    model = model.to(DEVICE)
    part_1 = config["data"]["part_1"]
    part_2 = config["data"]["part_2"]
    drop_features = ["noi", "polyDensity"]

    # Get distribution from training set
    full_df = PartDataset(
        dataset_name=config["data"]["dataset_name"],
        part=part_2,
        representation="descriptors",
    ).as_df(subset=n_training_obs)
    dataset_df = full_df.drop(
        ["filename", "segment_id"] + drop_features,
        axis=1,
    ).dropna(how="all", axis=1)

    train_dist = get_flat_nonzero_dissimilarity_matrix(dataset_df.values)

    tenths = [i / 10 for i in range(11)]
    xys = list(itertools.product(tenths, repeat=2))
    n_seqs = len(xys)

    xy_gen_hits = {}
    for x, y in xys:
        src = torch.tensor([x, y], dtype=torch.float32, device=device)
        with torch.no_grad():
            gen_hits = model(src)
            xy_gen_hits[f"({x},{y})"] = gen_hits

    # Threshold and make velocity bins
    zero_threshes = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1]
    thresh_list = []
    for zt in zero_threshes:
        tl = [zt + i * ((1 - zt) / 4) for i in range(4)]
        thresh_list.append(tl)

    for threshes in thresh_list:
        print(threshes[0])
        thresh_dir = os.path.join(gen_dir, str(threshes[0]))
        if not os.path.isdir(thresh_dir):
            os.makedirs(thresh_dir)

        threshed_rolls = []
        threshed_descs = []

        all_zeros = 0
        all_same = 0

        for xy_str, seq in xy_gen_hits.items():
            x, y = [float(i) for i in xy_str[1:-1].split(",")]
            sampled_hits = []
            for i in seq:
                if i <= threshes[0]:
                    sampled_hits.append(0)
                elif i <= threshes[1]:
                    sampled_hits.append(0.25)
                elif i <= threshes[2]:
                    sampled_hits.append(0.5)
                elif i <= threshes[3]:
                    sampled_hits.append(0.75)
                else:
                    sampled_hits.append(1)
            sampled_hits = torch.tensor(sampled_hits, dtype=torch.float32)

            if max(sampled_hits) == 0:
                all_zeros += 1
                continue
            if len(set(sampled_hits)) == 1:
                all_same += 1
                continue

            write_midi_from_hits(
                [i * 127 for i in sampled_hits],
                outpath=os.path.join(thresh_dir, f"({x},{y}).mid"),
                part=part_2,
                pitch=pitch,
                name=f"({x},{y})",
            )

            threshed_roll = get_roll_from_hits(
                [i * 127 for i in sampled_hits], pitch=pitch, resolution=resolution
            )
            threshed_rolls.append(threshed_roll)

            t_descs = pianoroll2descriptors(
                threshed_roll,
                resolution,
                drums=part_2 == "Drums",
            )
            threshed_descs.append(t_descs)

        print(f"{n_seqs=}")
        print(f"  {all_zeros=} ({100*round(all_zeros/n_seqs, 2)}%)")
        print(f"  {all_same=} ({100*round(all_same/n_seqs, 2)}%)")

        gen_df = pd.DataFrame(threshed_descs).dropna(how="all", axis=1)
        gen_df.drop(drop_features, axis=1, inplace=True)

        mk_descriptor_dist_plot(
            gen_df=gen_df,
            ref_df=dataset_df,
            outdir=eval_dir,
            model_name=model_name,
            checkpoint_num=checkpoint_num,
            title_suffix=f" zero_thresh={threshes[0]}",
            filename_suffix=str(threshes[0]),
        )

        # Stack training and generation descriptors
        train_gen = np.concatenate((dataset_df.values, gen_df.values))
        train_gen_dist = get_flat_nonzero_dissimilarity_matrix(train_gen)

        # Compute distribution comparison metrics
        oa, kld = compute_oa_and_kld(train_dist, train_gen_dist)
        print(f"  oa={round(oa, 3)}, kld={round(kld, 3)}")

        make_oa_kld_plot(
            train_dist=train_dist,
            train_gen_dist=train_gen_dist,
            oa=oa,
            kld=kld,
            model_name=model_name,
            outdir=eval_dir,
            suffix=str(threshes[0]),
        )

        # TODO: save stats somewhere. Back into model obj?
