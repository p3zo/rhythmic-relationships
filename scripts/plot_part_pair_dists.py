import argparse
import itertools
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rhythmtoolbox as rtb
import seaborn as sns
from rhythmic_relationships import DATASETS_DIR, PLOTS_DIRNAME
from rhythmic_relationships.data import PartPairDataset
from rhythmic_relationships.parts import get_part_pairs
from torch.utils.data import DataLoader

sns.set_style("white")
sns.set_context("paper")


def get_distance(point1, point2):
    """Compute the Euclidean distance between two points"""
    return np.sqrt((point2[0] - point1[0]) ** 2 + (point2[1] - point1[1]) ** 2)


def save_fig(filepath, title=None):
    if title:
        plt.title(title)
    plt.tight_layout()
    plt.savefig(filepath)
    print(f"Saved {filepath}")
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="babyslakh_20_1bar_4res",
        help="Name of the dataset to make plots for. Create a new dataset using `prepare_dataset.py`.",
    )
    parser.add_argument(
        "--skip_pairplot_constituents",
        action="store_true",
        help="Several joint distributions are shown by the pairplot. Switching this flag to true skips individual plots in favor of viewing them in the pairplot.",
    )
    parser.add_argument(
        "--parts_to_analyze",
        nargs="+",
        default=["Drums", "Piano", "Guitar", "Bass"],
        help="Names of the parts to analyze",
    )
    parser.add_argument(
        "--descriptors_to_analyze",
        nargs="+",
        default=[
            "polyD",
            "polysync",
            "polybalance",
            "polyevenness",
            "noi",
        ],
        help="Names of the descriptors to analyze",
    )
    args = parser.parse_args()

    dataset_name = args.dataset_name
    skip_pairplot_constituents = args.skip_pairplot_constituents
    parts_to_analyze = args.parts_to_analyze
    descriptors_to_analyze = args.descriptors_to_analyze

    for p1, p2 in get_part_pairs(parts_to_analyze):
        print(f"Loading {p1}_{p2} pairs")

        # Load the data
        config = {
            "dataset_name": dataset_name,
            "part_1": p1,
            "part_2": p2,
            "repr_1": "descriptors",
            "repr_2": "descriptors",
        }
        dataset = PartPairDataset(**config)
        n_pairs = len(dataset)

        loader = DataLoader(dataset, batch_size=n_pairs)
        x, y = next(iter(loader))

        xdf = pd.DataFrame(x, columns=rtb.DESCRIPTOR_NAMES)
        ydf = pd.DataFrame(y, columns=rtb.DESCRIPTOR_NAMES)

        xdf = xdf[descriptors_to_analyze]
        ydf = ydf[descriptors_to_analyze]

        if "noi" in descriptors_to_analyze:
            xdf["noi"] = xdf["noi"].fillna(0).astype(int)
            ydf["noi"] = ydf["noi"].fillna(0).astype(int)

        if "polysync" in descriptors_to_analyze:
            xdf["polysync"] = xdf["polysync"].fillna(0).astype(int)
            ydf["polysync"] = ydf["polysync"].fillna(0).astype(int)

        # Each row is a p1_p2 pair with all descriptors for both parts
        pair_df = xdf.join(ydf, lsuffix=p1, rsuffix=p2)

        xdf["part"] = p1
        ydf["part"] = p2
        stacked_df = pd.concat([xdf, ydf]).reset_index(drop=True)

        # Create the output directory
        plots_dir = os.path.join(
            DATASETS_DIR, dataset_name, PLOTS_DIRNAME, f"{p1}_{p2}"
        )
        if not os.path.isdir(plots_dir):
            os.makedirs(plots_dir)

        # Grid of joint and marginal distributions
        print("Creating pairplot...")
        sns.pairplot(
            stacked_df,
            hue="part",
            markers=["_", "|"],
            corner=True,
        )
        save_fig(os.path.join(plots_dir, "pairplot.png"))

        fig, ax = plt.subplots()

        # Univariate distributions
        for col in descriptors_to_analyze:
            sns.displot(stacked_df, x=col, y="part")
            save_fig(os.path.join(plots_dir, f"{col}_cmp.png"))

            # As CDFs
            sns.displot(stacked_df, x=col, hue="part", kind="ecdf")
            save_fig(os.path.join(plots_dir, f"{col}_cdf.png"), title="CDF")

            if not skip_pairplot_constituents:
                # As KDEs
                sns.displot(stacked_df, x=col, hue="part", kind="kde", fill=True, cut=0)
                save_fig(os.path.join(plots_dir, f"{col}_kde.png"), title="KDE")

        # Bivariate distributions
        for xcol, ycol in itertools.combinations(descriptors_to_analyze, 2):
            # Scatter plots with pairs of points connected
            print("Plotting connected scatter...")
            fig, ax = plt.subplots(figsize=(6, 6), dpi=300)
            sns.scatterplot(
                data=stacked_df,
                x=xcol,
                y=ycol,
                hue="part",
                style="part",
                markers={p1: "_", p2: "|"},
                ax=ax,
            )
            pdf = pair_df[[xcol + p1, ycol + p1, xcol + p2, ycol + p2]]
            distances = []
            for ix, row in pdf.iterrows():
                point1 = row[xcol + p1], row[ycol + p1]
                point2 = row[xcol + p2], row[ycol + p2]
                distances.append(get_distance(point1, point2))
                xs = [point1[0], point2[0]]
                ys = [point1[1], point2[1]]
                ax.plot(xs, ys, "k-", lw=0.1, alpha=0.1)
            plt.legend(loc="best")
            sns.despine()
            save_fig(
                os.path.join(plots_dir, f"{xcol}_{ycol}_paired_scatter.png"),
                title=f"{dataset_name}\n{n_pairs} pairs",
            )

            # Plot distribution of distances between pairs
            sns.displot(distances, kde=True)
            save_fig(
                os.path.join(plots_dir, f"{xcol}_{ycol}_pair_distances.png"),
                title=f"Distribution of distances between pairs\n{p1} {p2}, {xcol} {ycol}\n{dataset_name}, {n_pairs} pairs",
            )

            if not skip_pairplot_constituents:
                sns.displot(stacked_df, x=xcol, y=ycol, hue="part", kind="kde")
                save_fig(
                    os.path.join(plots_dir, f"{xcol}_{ycol}_contour.png"),
                    title="Contour",
                )
                sns.displot(stacked_df, x=xcol, y=ycol, hue="part")
                save_fig(
                    os.path.join(plots_dir, f"{xcol}_{ycol}_heatmap.png"),
                    title="Heatmap",
                )

                # Scatter
                fig, ax = plt.subplots(figsize=(6, 6), dpi=300)
                sns.scatterplot(
                    data=stacked_df,
                    x=xcol,
                    y=ycol,
                    hue="part",
                    style="part",
                    markers={p1: "_", p2: "|"},
                    ax=ax,
                )
                plt.xlabel(xcol)
                plt.ylabel(ycol)
                plt.legend(loc="best")
                sns.despine()
                save_fig(
                    os.path.join(plots_dir, f"{xcol}_{ycol}_scatter.png"),
                    title=f"{dataset_name}\n{n_pairs} pairs",
                )

        """
        Question: When p1 is syncopated by a given amount, what is the distribution of syncopation in the p2?
        """
        # Skip these for now
        if False:
            for col1, col2 in itertools.combinations(["noi", "polysync", "polyD"], 2):
                col1 = f"{col1}{p1}"
                col2 = f"{col1}{p2}"

                vcs = pair_df[col1].value_counts()
                for val in vcs.index:
                    sns.kdeplot(data=pair_df[pair_df[col1] == val], x=col2)
                    save_fig(os.path.join(plots_dir, f"{col1}={val}_{col2}_kde.png"))

                # When the p1 has n instruments, what n instruments in the p2?
                sns.kdeplot(data=pair_df, x=col1, hue=col2)
                save_fig(os.path.join(plots_dir, f"{col1}_{col2}_kde.png"))

                # When the p2 has n instruments, what n instruments in the p1?
                sns.kdeplot(data=pair_df, x=col2, hue=col1)
                save_fig(os.path.join(plots_dir, f"{col2}_{col1}_kde.png"))
