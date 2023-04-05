"""
The transparent glue

Perform dimensionality reduction over PartPairDatasets and visualize the resulting low-dimensional spaces.
Use both MDS and t-SNE.

Example usage:
    python scripts/pairspace.py --subset=20
"""
import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from utils import save_fig, play_midi_file
from rhythmic_relationships import DATASETS_DIR, PLOTS_DIRNAME
from rhythmic_relationships.data import PartPairDataset
from rhythmic_relationships.io import get_pmid_segment, load_midi_file
from sklearn.manifold import MDS, TSNE

sns.set_style("white")
sns.set_context("paper")

INPUT_DATA_DIR = "../input/babyslakh"


def get_distance(point1, point2):
    """Compute the Euclidean distance between two points"""
    return np.sqrt((point2[0] - point1[0]) ** 2 + (point2[1] - point1[1]) ** 2)


def get_embeddings(X, filenames, method="t-SNE", title=""):
    """Create a 2D embedding space of the data, plot it, and save it to a csv"""
    reducer = TSNE(n_components=2, init="pca", learning_rate="auto", random_state=42)
    if method == "MDS":
        reducer = MDS(n_components=2, n_init=1, random_state=42)

    # Make the pair space using t-SNE
    X_transform = reducer.fit_transform(X)

    emb = pd.DataFrame(X_transform, columns=["component_1", "component_2"])
    emb["filename"] = filenames
    outname = f"{method}_{title}"
    emb.to_csv(os.path.join(plots_dir, f"{outname}.csv"))
    sns.relplot(
        data=emb,
        x="component_1",
        y="component_2",
        hue="filename",
        height=8,
        aspect=1.25,
        legend=False,
    )
    for ix, row in emb.iterrows():
        plt.text(
            row["component_1"],
            row["component_2"] + 1,
            str(ix),
            ha="center",
            va="center",
            color="gray",
        )
    plt.grid("minor")

    save_fig(
        os.path.join(plots_dir, f"{outname}.png"),
        title=f"Paired {method} embeddings ({title})\n{dataset_name}\nColored by file",
    )

    return emb


def get_n_closest(coord, emb, n):
    """Find the n points in emb closest to coord

    :parameters:
        coord : tuple
            The (x, y) coordinates of the point to find the closest to
        emb : pd.DataFrame
            The embedding of the data
        n : int
            The number of points to find

    :returns:
        closest_emb: pd.DataFrame
            The n closest points in emb
    """
    closest_ixs = []
    for ix, row in emb.iterrows():
        dist = get_distance(coord, row[["component_1", "component_2"]])
        if len(closest_ixs) < n:
            closest_ixs.append((ix, dist))
            closest_ixs = sorted(closest_ixs, key=lambda x: x[1])
        else:
            if dist < closest_ixs[-1][1]:
                closest_ixs[-1] = (ix, dist)
                closest_ixs = sorted(closest_ixs, key=lambda x: x[1])

    return emb.loc[[i[0] for i in closest_ixs]]


def find_corresponding_point(coord, emb1, emb2):
    """Given a coordinate in the embeddings pace of one part, find a corresponding point in the embedding space of the other part

    :parameters:
        coord : tuple
            The (x, y) coordinates of the point to find the closest to
        emb1 : pd.DataFrame
            The embedding of the data for the first part
        emb2 : pd.DataFrame
            The embedding of the data for the second part

    :returns:
        closest_ix : int
    """
    # Find the three closest points to the input coord
    three_closest = get_n_closest(coord, emb1, n=3)

    # Get the corresponding points from emb2
    corresponding_emb2 = emb2.iloc[three_closest.index]

    # Return the point from emb2 closest to average of those three corresponding points
    mean_corresponding_coord = (
        corresponding_emb2.component_1.mean(),
        corresponding_emb2.component_2.mean(),
    )
    closest_ix = (
        get_n_closest(mean_corresponding_coord, corresponding_emb2, n=1).iloc[0].name
    )

    return emb2.loc[closest_ix]


def get_midi_for_closest(coord, emb, parts=[]):
    """Find the point in emb closest to coord and loads the corresponding MIDI for the segment

    :parameters:
        coord : tuple
            The (x, y) coordinates of the point to find the closest to
        emb : pd.DataFrame
            The embedding of the data
        parts : list
            The parts to load from the MIDI file

    :returns:
        pmid_slice : pretty_midi.PrettyMIDI
    """
    # Find the closest point
    closest = emb.iloc[0]
    closest_dist = get_distance(coord, closest[["component_1", "component_2"]])
    for ix, row in emb.iterrows():
        dist = get_distance(coord, row[["component_1", "component_2"]])
        if dist < closest_dist:
            closest = row
            closest_dist = dist

    # Load the MIDI file
    pmid = load_midi_file(os.path.join(INPUT_DATA_DIR, closest["filename"] + ".mid"))
    pmid_slice = get_pmid_segment(
        pmid,
        segment_num=closest.name,
        resolution=4,
        seg_size=2,
        n_beat_bars=4,
        parts=parts,
    )
    print(f'Loaded MIDI for {closest["filename"]} segment {closest.name}')
    return pmid_slice


def plot_emb_interactive(emb):
    """Create an interactive plot of the embedding space"""

    # Enable interactive plotting
    plt.ion()

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.scatterplot(
        data=emb,
        x="component_1",
        y="component_2",
        hue="filename",
        legend=False,
        ax=ax,
    )

    tmp_filepath = "interactive-plotting-temp-01934893.mid"

    def onclick(event):
        if event.button != 1:
            return

        print(f"Clicked {event.xdata}, {event.ydata}")

        # Retrieve MIDI for the closest point to the input coord
        pmid = get_midi_for_closest(
            (event.xdata, event.ydata), emb, parts=["Drums", "Piano"]
        )

        pmid.write(tmp_filepath)
        play_midi_file(tmp_filepath)

    def onclose(event):
        print("Figure closed")
        os.remove(tmp_filepath)
        fig.canvas.mpl_disconnect(cid_click)
        fig.canvas.mpl_disconnect(cid_close)
        plt.ioff()

    # Connect to the event manager
    cid_click = fig.canvas.mpl_connect("button_press_event", onclick)
    cid_close = fig.canvas.mpl_connect("close_event", onclose)

    # Start plot
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        default="babyslakh_20_1bar_4res",
        help="Name of the dataset to make plots for.",
    )
    parser.add_argument(
        "--subset",
        type=int,
        default=25,
        help="Size of subset to use. If None, use all.",
    )
    parser.add_argument(
        "--descriptors_to_analyze",
        nargs="+",
        default=[
            "noi",
            "stepDensity",
            "polyDensity",
            "balance",
            "evenness",
            "sync",
            "syness",
        ],
        help="Names of the descriptors to analyze",
    )
    args = parser.parse_args()

    dataset_name = args.dataset
    descriptors_to_analyze = args.descriptors_to_analyze
    subset = args.subset

    # Create the output directory
    plots_dir = os.path.join(DATASETS_DIR, dataset_name, PLOTS_DIRNAME, "pairspaces")
    if not os.path.isdir(plots_dir):
        os.makedirs(plots_dir)

    # Load the data
    p1 = "Drums"
    p2 = "Piano"
    pair_df, _ = PartPairDataset(
        dataset_name, p1, p2, "descriptors", "descriptors"
    ).as_dfs(shuffle=False)

    # TODO: remove this once ensured no NA can be in dataset
    df = pair_df.dropna()

    if subset:
        df = df[:subset]
    n_pairs = len(df)

    filenames = df["filenamePiano"].values
    df = df[[c for c in df.columns if any([i in c for i in descriptors_to_analyze])]]

    # Pair space
    emb_tsne = get_embeddings(df, filenames, title="paired input")
    emb_mds = get_embeddings(df, filenames, title="paired input", method="MDS")

    # Make the Drums space
    dfd = df[[c for c in df.columns if "Drums" in c]]
    drums_emb_tsne = get_embeddings(dfd, filenames, title="Drums")
    drums_emb_mds = get_embeddings(dfd, filenames, title="Drums", method="MDS")

    # Make the Piano space
    dfp = df[[c for c in df.columns if "Piano" in c]]
    piano_emb_tsne = get_embeddings(dfp, filenames, title="Piano")
    piano_emb_mds = get_embeddings(dfp, filenames, title="Piano", method="MDS")

    # Embed the paired embedding spaces
    paired_emb_tsne = pd.concat(
        [
            drums_emb_tsne.drop("filename", axis=1),
            piano_emb_tsne.drop("filename", axis=1),
        ],
        axis=1,
    )
    paired_emb_mds = pd.concat(
        [
            drums_emb_mds.drop("filename", axis=1),
            piano_emb_mds.drop("filename", axis=1),
        ],
        axis=1,
    )

    emb_emb_tsne = get_embeddings(paired_emb_tsne, filenames, title="paired embedding")
    emb_emb_mds = get_embeddings(
        paired_emb_mds, filenames, title="paired embedding", method="MDS"
    )

    # Example: Get corresponding points from Drums <--> Piano
    find_corresponding_point((1, -1), piano_emb_mds, drums_emb_mds)

    # Example: Make an interactive plot of an embedding space
    plot_emb_interactive(drums_emb_mds)
