"""Perform dimensionality reduction over PartDatasets and visualize the resulting low-dimensional spaces."""

import argparse
import os

from rhythmic_relationships import DATASETS_DIR, PLOTS_DIRNAME
from rhythmic_relationships.data import PartDataset
from utils import get_embeddings


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        default="lmdc_3000_1bar_4res",
        help="Name of the dataset to make plots for. Create a new dataset using `prepare_dataset.py`.",
    )
    parser.add_argument(
        "--parts_to_analyze",
        nargs="+",
        default=["Melody", "Bass"],
        help="Names of the parts to analyze",
    )
    parser.add_argument(
        "--subset",
        type=int,
        default=None,
        help="The number of segments to load.",
    )
    parser.add_argument(
        "--method",
        type=str,
        default="mds",
        help="Dimensionality reduction method to use. MDS or t-SNE.",
    )
    args = parser.parse_args()

    dataset_name = args.dataset
    parts_to_analyze = args.parts_to_analyze
    subset = args.subset
    method = args.method

    for part in parts_to_analyze:
        output_dir = os.path.join(DATASETS_DIR, dataset_name, PLOTS_DIRNAME, part)
        if not os.path.isdir(output_dir):
            os.makedirs(output_dir)

        df = PartDataset(dataset_name, part, "descriptors").as_df(subset=subset)
        df = df.dropna(how="all", axis=1)

        filenames = df["filename"].values
        segment_ids = df["segment_id"].values
        feature_names = [i for i in df.columns if i not in ["filename", "segment_id"]]
        fdf = df[[c for c in df.columns if any([i in c for i in feature_names])]]

        # Normalize features to [0, 1] for consistent dimensionality reduction
        fdf_scaled = (fdf - fdf.min()) / (fdf.max() - fdf.min())

        space = get_embeddings(
            fdf_scaled,
            filenames,
            segment_ids,
            title=part,
            method=method,
            outdir=output_dir,
            dataset_name=dataset_name,
            normalize=True,
        )
