import argparse
import itertools
import os

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from rhythmic_relationships import DATASETS_DIR, PLOTS_DIRNAME
from rhythmic_relationships.data import PartDataset
from rhythmic_relationships.parts import get_part_pairs

sns.set_style("white")
sns.set_context("paper")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        default="lmdc_3000_2bar_4res",
        help="Name of the dataset to analyze. Create a new dataset using `prepare_dataset.py`.",
    )
    args = parser.parse_args()

    dataset_name = args.dataset

    subset = None

    # Create the output directory
    plots_dir = os.path.join(DATASETS_DIR, dataset_name, PLOTS_DIRNAME)
    if not os.path.isdir(plots_dir):
        os.makedirs(plots_dir)

    drop_cols = ["noi", "polyDensity", "syness"]

    bass_df = PartDataset(dataset_name, "Bass", "descriptors").as_df(subset=subset)
    bass_df.dropna(how="all", axis=0, inplace=True)
    bass_df.dropna(how="all", axis=1, inplace=True)
    bass_df.drop(drop_cols, axis=1, inplace=True)
    bass_df.drop(["filename", "segment_id"], axis=1, inplace=True)
    bass_df.to_csv(
        os.path.join(plots_dir, f"bass_descriptors_{len(bass_df)}.csv"),
        index=False,
    )

    mel_df = PartDataset(dataset_name, "Melody", "descriptors").as_df(subset=subset)
    drop_cols = ["noi", "polyDensity", "syness"]
    mel_df.dropna(how="all", axis=0, inplace=True)
    mel_df.dropna(how="all", axis=1, inplace=True)
    mel_df.drop(drop_cols, axis=1, inplace=True)
    mel_df.drop(["filename", "segment_id"], axis=1, inplace=True)
    mel_df.to_csv(
        os.path.join(plots_dir, f"melody_descriptors_{len(mel_df)}.csv"),
        index=False,
    )

    counts = pd.DataFrame([['Bass', len(bass_df)], ['Melody', len(mel_df)]])
    counts.to_csv(os.path.join(plots_dir, 'segment_counts.csv'), index=False, header=False)
