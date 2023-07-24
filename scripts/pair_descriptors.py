import argparse
import os

import pandas as pd
import seaborn as sns
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader

from rhythmic_relationships import DATASETS_DIR, PLOTS_DIRNAME
from rhythmic_relationships.data import PartPairDataset

from utils import save_fig

sns.set_style("white")
sns.set_context("paper")


def get_center(x):
    print("Computing centers... (TODO: vectorize)")
    nz = x.nonzero()
    centers = []
    min_centers = []
    max_centers = []
    for i in tqdm(range(x.shape[0])):
        density = x[i].sum()

        ixs = nz[:, 1][torch.where(nz[:, 0] == i)[0]]
        centers.append(sum(ixs) / density)

        triangle = sum(list(range(density + 1)))
        min_center = triangle / density
        min_centers.append(min_center)
        max_centers.append(len(x[i]) - min_center)

    centers = torch.tensor(centers)
    min_centers = torch.tensor(min_centers)
    max_centers = torch.tensor(max_centers)

    return centers, min_centers, max_centers


def get_onset_balance(a, b):
    stacked_densities = torch.stack([a.sum(axis=1), b.sum(axis=1)])
    pair_densities = stacked_densities.sum(axis=0)
    abs_onset_diffs = stacked_densities.diff(axis=0)[0].abs()
    return 1 - (abs_onset_diffs / pair_densities)


def get_antiphony(a, b):
    a_center, a_min_center, a_max_center = get_center(a)
    b_center, b_min_center, b_max_center = get_center(b)
    return (a_center - b_center).abs() / (a_max_center - b_min_center).abs()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        default="lmdc_1000_2bar_4res",
        help="Name of the dataset to make plots for. Create a new dataset using `prepare_dataset.py`.",
    )
    parser.add_argument(
        "--part_1",
        default="Bass",
        help="Names of the first part to analyze",
    )
    parser.add_argument(
        "--part_2",
        default="Melody",
        help="Names of the second part to analyze",
    )
    args = parser.parse_args()

    dconfig = {
        "dataset_name": args.dataset,
        "part_1": args.part_1,
        "part_2": args.part_2,
        "repr_1": "hits",
        "repr_2": "hits",
        "block_size": 1,
    }
    print(f'Loading dataset with config: {dconfig}')
    dataset = PartPairDataset(**dconfig)
    loader = DataLoader(dataset, batch_size=len(dataset), shuffle=True)

    # Create the output directory
    plots_dir = os.path.join(
        DATASETS_DIR,
        dconfig["dataset_name"],
        PLOTS_DIRNAME,
        f"{dconfig['part_1']}_{dconfig['part_2']}",
    )
    if not os.path.isdir(plots_dir):
        os.makedirs(plots_dir)

    xx = next(iter(loader))
    s0 = (xx[0] > 1).to(int)
    s1 = (xx[1] > 1).to(int)

    onset_balance = get_onset_balance(s0, s1)
    antiphony = get_antiphony(s0, s1)

    # TODO: 2 antiphonies are > 1. Is that desired?
    # This happens when one is all 1s.
    # Antiphony was expected to be [0, 1] but it's actually [0, 32] (?)
    # If abs(max_center(a) - min_center(b)) < 1 and abs(center(a) - center(b)) > 1
    # min possible max_center is 0 (pattern is empty)
    # center is 0 for empty pattern, 0 for [1, 0, 0, 0], 32 for [0, ..., 1]

    columns = ["onset_balance", "antiphony"]
    df = pd.DataFrame(
        torch.stack([onset_balance, antiphony], axis=1),
        columns=columns,
    )

    for col in columns:
        sns.displot(df, x=col)
        save_fig(os.path.join(plots_dir, f"{col}_cmp.png"))

        # As CDFs
        sns.displot(df, x=col, kind="ecdf")
        save_fig(os.path.join(plots_dir, f"{col}_cdf.png"), title="CDF")

        # As KDEs
        sns.displot(df, x=col, kind="kde", fill=True, cut=0)
        save_fig(os.path.join(plots_dir, f"{col}_kde.png"), title="KDE")
