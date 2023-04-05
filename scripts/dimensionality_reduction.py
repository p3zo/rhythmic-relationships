"""Perform dimensionality reduction over PartDatasets and visualize the resulting low-dimensional spaces."""
import argparse
import os

import pandas as pd
import seaborn as sns
from utils import save_fig
from rhythmic_relationships import DATASETS_DIR, PLOTS_DIRNAME
from rhythmic_relationships.data import PartDataset
from sklearn.manifold import MDS, TSNE

sns.set_style("white")
sns.set_context("paper")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        default="babyslakh_20_1bar_4res",
        help="Name of the dataset to make plots for. Create a new dataset using `prepare_dataset.py`.",
    )
    parser.add_argument(
        "--parts_to_analyze",
        nargs="+",
        default=["Drums", "Piano", "Guitar", "Bass"],
        help="Names of the parts to analyze",
    )

    args = parser.parse_args()

    dataset_name = args.dataset
    parts_to_analyze = args.parts_to_analyze

    for part in parts_to_analyze:
        df = PartDataset(dataset_name, part, "descriptors").as_df()
        # TODO: remove this once ensured no NA can be in dataset
        df = df.dropna()

        # Create the output directory
        part_plots_dir = os.path.join(DATASETS_DIR, dataset_name, PLOTS_DIRNAME, part)
        if not os.path.isdir(part_plots_dir):
            os.makedirs(part_plots_dir)

        print("Performing MDS...")
        mds = MDS(n_components=2, n_init=1, random_state=42)
        X = df.drop("filename", axis=1)
        X_transform = mds.fit_transform(X)

        mds_emb_df = pd.DataFrame(X_transform, columns=["component_1", "component_2"])
        mds_emb_df["filename"] = df["filename"]
        sns.relplot(
            data=mds_emb_df,
            x="component_1",
            y="component_2",
            hue="filename",
            height=8,
            aspect=1.25,
            legend=False,
        )
        save_fig(
            os.path.join(part_plots_dir, "MDS.png"),
            title=f"{part} MDS embeddings\n{dataset_name}\nColored by file",
        )

        print("Performing t-SNE...")
        tsne = TSNE(n_components=2, init="pca", learning_rate="auto", random_state=42)
        X = df.drop("filename", axis=1)
        X_transform = tsne.fit_transform(X)

        tsne_emb_df = pd.DataFrame(X_transform, columns=["component_1", "component_2"])
        tsne_emb_df["filename"] = df["filename"]
        sns.relplot(
            data=tsne_emb_df,
            x="component_1",
            y="component_2",
            hue="filename",
            height=8,
            aspect=1.25,
            legend=False,
        )
        save_fig(
            os.path.join(part_plots_dir, "t-SNE.png"),
            title=f"{part} t-SNE embeddings\n{dataset_name}\nColored by file",
        )
