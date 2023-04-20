"""
The "transparent glue".

- Perform dimensionality reduction over PartPairDatasets.
- Use triangulation to find corresponding pairs of points in the resulting embedding spaces.
- Interpolate in the input domain to treat the embedding spaces as continuous.

Example usage:
    python scripts/pairspace.py --subset=2000
"""
import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from rhythmic_relationships import DATASETS_DIR, PLOTS_DIRNAME
from rhythmic_relationships.data import PartPairDataset
from rhythmic_relationships.io import (
    get_pmid_segment,
    load_midi_file,
    write_midi_from_hits,
    get_subdivisions,
)
from rhythmic_relationships.representations import get_representations
from scipy.spatial import Delaunay
from sklearn.manifold import MDS, TSNE
from utils import save_fig

sns.set_style("white")
sns.set_context("paper")


def get_distance(point1, point2):
    """Compute the Euclidean distance between two points"""
    return np.sqrt((point2[0] - point1[0]) ** 2 + (point2[1] - point1[1]) ** 2)


def get_embeddings(X, filenames, segment_ids, method="t-SNE", title="", outdir="."):
    """Create a 2D embedding space of the data, plot it, and save it to a csv"""
    reducer = TSNE(n_components=2, init="pca", learning_rate="auto", random_state=42)
    if method == "MDS":
        reducer = MDS(n_components=2, n_init=1, random_state=42)

    # Make the pair space using t-SNE
    X_transform = reducer.fit_transform(X)

    emb = pd.DataFrame(X_transform, columns=["x", "y"])
    emb["filename"] = filenames
    emb["segment_id"] = segment_ids
    outname = f"{method}_{title}"
    emb.to_csv(os.path.join(outdir, f"{outname}.csv"), index=False)
    sns.relplot(
        data=emb,
        x="x",
        y="y",
        hue="filename",
        height=8,
        aspect=1.25,
        legend=False,
    )
    if len(emb) < 100:
        for ix, row in emb.iterrows():
            plt.text(
                row["x"],
                row["y"] + 1,
                str(ix),
                ha="center",
                va="center",
                color="gray",
            )
    plt.grid("minor")

    save_fig(
        os.path.join(outdir, f"{outname}.png"),
        title=f"Paired {method} embeddings ({title})\n{dataset_name}\nColored by file",
    )

    return emb


def get_triangles(coords):
    """Compute the vertices of a triangle set that covers the coordinate space"""

    # Delaunay triangulations maximize the minimum of all the angles of the triangles
    tri = Delaunay(coords)

    return tri.simplices


def get_triangle_area(v1, v2, v3):
    """Return the area of a triangle given its vertices."""
    return 0.5 * abs(
        (v2[0] - v1[0]) * (v3[1] - v1[1]) - (v3[0] - v1[0]) * (v2[1] - v1[1])
    )


def is_coord_inside_triangle(coord, v1, v2, v3):
    """Return True if a point is inside a triangle.

    :parameter
        coord: tuple
            The coordinates of the point to test
        v1, v2, v3: tuple
            The coordinates of the vertices of the triangle

    :return
        True if the point is inside the triangle, False otherwise
    """
    a = get_triangle_area(v1, v2, v3)
    a1 = get_triangle_area(v1, v2, coord)
    a2 = get_triangle_area(v2, v3, coord)
    a3 = get_triangle_area(v1, v3, coord)

    if a < (a1 + a2 + a3):
        return False

    return True


def find_triangle(coord, triangles, coords):
    """Find the triangle to which a 2D coordinate belongs.

    :parameter
        coord: tuple
            The coordinates of the point to test
        triangles: list of tuples
            The triangles of the triangulation
        coords: list of tuples
            The coordinates of the vertices of the triangles

    :return
        The index of the triangle to which the coordinate belongs.
        If the coordinate is not in any triangle, return None.
    """
    for t in triangles:
        v1, v2, v3 = coords[t].tolist()
        if is_coord_inside_triangle(coord, v1, v2, v3):
            return t

    print("The coordinate is not in any triangle")
    return None


def get_vertex_weights(coord, vertex_coords):
    """Compute the weights of each vertex of a triangle on a point that falls inside the triangle.

    :param: coord, tuple
        A coordinate that falls inside the given triangle

    :param: vertex_coords, np.array
        Coordinates of the triangle vertices

    :return:
        The weight of each vertex on the coord
    """
    A, B, C = vertex_coords
    a = get_triangle_area(coord, B, C)
    b = get_triangle_area(coord, A, C)
    c = get_triangle_area(coord, A, B)

    # TODO: pull this assertion to a unit test
    assert np.allclose(np.array(coord), (a * A + b * B + c * C) / (a + b + c))

    return a, b, c


def interpolate_three_patterns(hits_list, weights):
    """Interpolate 3 hits vectors given a set of weights"""
    a, b, c = weights

    # Sum the weight vectors of the three patterns
    w0 = a * hits_list[0]
    w1 = b * hits_list[1]
    w2 = c * hits_list[2]
    w = w0 + w1 + w2

    # Keep the N onsets with the highest weights, where N is the avg density
    mean_density = np.mean([sum(h) for h in hits_list])
    onset_ixs = w.argsort()[-int(mean_density) :]
    interpolated = np.zeros(len(w), dtype=np.int8)
    interpolated[onset_ixs] = 1

    return interpolated


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        default="babyslakh_20_1bar_4res",
        help="Name of the dataset to make plots for.",
    )
    parser.add_argument(
        "--midi_dir",
        type=str,
        default="../input/babyslakh",
        help="Name of the directory from which to load MIDI data.",
    )
    parser.add_argument(
        "--subset",
        type=int,
        default=None,
        help="Number of segments to use. If None, use all.",
    )
    parser.add_argument(
        "--method",
        type=str,
        default="t-SNE",
        help="The dimensionality reduction technique to use. Either t-SNE or MDS.",
    )
    parser.add_argument(
        "--feature_names",
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
    feature_names = args.feature_names
    subset = args.subset
    midi_dir = args.midi_dir
    method = args.method

    # Load the data
    p1 = "Drums"
    p2 = "Piano"
    pair_df, _ = PartPairDataset(
        dataset_name, p1, p2, "descriptors", "descriptors"
    ).as_dfs(shuffle=False, subset=subset)

    # TODO: remove this once ensured no NA can be in dataset
    df = pair_df.dropna()

    if subset:
        df = df[:subset]
    n_pairs = len(df)

    filenames = df["filename"].values
    segment_ids = df["segment_id"].values
    fdf = df[[c for c in df.columns if any([i in c for i in feature_names])]]

    # Create the output directory
    output_dir = os.path.join(
        DATASETS_DIR, dataset_name, PLOTS_DIRNAME, "pairspaces", str(len(segment_ids))
    )
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    # Paired input space
    emb = get_embeddings(
        fdf,
        filenames,
        segment_ids,
        title="paired input",
        method=method,
        outdir=output_dir,
    )

    # Drums space
    drums_emb = get_embeddings(
        fdf[[c for c in fdf.columns if "Drums" in c]],
        filenames,
        segment_ids,
        title="Drums",
        method=method,
        outdir=output_dir,
    )
    # Piano space
    piano_emb = get_embeddings(
        fdf[[c for c in fdf.columns if "Piano" in c]],
        filenames,
        segment_ids,
        title="Piano",
        method=method,
        outdir=output_dir,
    )

    # Paired embedding space
    emb_emb = get_embeddings(
        pd.concat(
            [
                drums_emb.drop(["segment_id", "filename"], axis=1),
                piano_emb.drop(["segment_id", "filename"], axis=1),
            ],
            axis=1,
        ),
        filenames,
        segment_ids,
        title="paired embedding",
        method=method,
        outdir=output_dir,
    )

    # Coord should be a value between 0 and 1 because the coords have been normalized
    input_coord = (0.4, 0.4)

    selection_dir = os.path.join(output_dir, str(input_coord))
    if not os.path.isdir(selection_dir):
        os.makedirs(selection_dir)

    # Given a point in the Piano space, get a corresponding point from the Drums space using Delaunay triangulation
    p_emb = piano_emb.drop(["segment_id", "filename"], axis=1).values.astype(np.float64)
    d_emb = drums_emb.drop(["segment_id", "filename"], axis=1).values.astype(np.float64)

    # Min-max normalize the coordinates to be between 0 and 1
    piano_coords = (p_emb - p_emb.min()) / (p_emb.max() - p_emb.min())
    drums_coords = (d_emb - d_emb.min()) / (d_emb.max() - d_emb.min())

    piano_tris = get_triangles(piano_coords)
    drums_tris = get_triangles(drums_coords)

    triangle = find_triangle(input_coord, piano_tris, piano_coords)
    if triangle is None:
        raise ValueError("The coordinate is not in any triangle")

    piano_tri_coords = piano_coords[triangle]
    drums_tri_coords = drums_coords[triangle]

    # Compute a point in the second space using a weighted avg of inverse coord-to-vertex ratios in the first
    a, b, c = get_vertex_weights(input_coord, piano_tri_coords)
    A, B, C = drums_tri_coords
    drums_coord = (a * A + b * B + c * C) / (a + b + c)

    fig, ax = plt.subplots(1, 2, figsize=(20, 10), sharey=True, sharex=True)

    ax[0].triplot(
        piano_coords[:, 0], piano_coords[:, 1], piano_tris, color="gray", lw=0.5
    )
    ax[0].plot(piano_coords[:, 0], piano_coords[:, 1], "o", color="black", markersize=1)
    ax[0].plot(input_coord[0], input_coord[1], "o", color="red")
    if isinstance(triangle, np.ndarray):
        ax[0].plot(piano_tri_coords[:, 0], piano_tri_coords[:, 1], "o", color="blue")
    ax[0].set_title("Piano")

    ax[1].triplot(
        drums_coords[:, 0], drums_coords[:, 1], drums_tris, color="gray", lw=0.5
    )
    ax[1].plot(drums_coords[:, 0], drums_coords[:, 1], "o", color="black", markersize=1)
    ax[1].plot(drums_coord[0], drums_coord[1], "o", color="red")
    if isinstance(triangle, np.ndarray):
        ax[1].plot(drums_tri_coords[:, 0], drums_tri_coords[:, 1], "o", color="blue")
    ax[1].set_title("Drums")

    if len(segment_ids) < 100:
        for ix, coord in enumerate(piano_tri_coords):
            ax[0].text(
                coord[0] - 0.01,
                coord[1] + 0.05,
                str(ix),
                ha="center",
                va="center",
                color="blue",
            )

        for ix, coord in enumerate(drums_tri_coords):
            ax[1].text(
                coord[0] - 0.01,
                coord[1] + 0.05,
                str(ix),
                ha="center",
                va="center",
                color="blue",
            )

    plt.xlim(-0.1, 1.1)
    plt.ylim(-0.1, 1.1)
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()
    plt.savefig(os.path.join(selection_dir, "triangles.png"))

    # Load the MIDI slices & derive hits representations for each part
    piano_hits_list = []
    drums_hits_list = []
    for ix in triangle:
        piano_row = piano_emb.iloc[ix]
        filename = piano_row["filename"]
        segment_id = piano_row["segment_id"]
        print(filename, segment_id)

        pmid = load_midi_file(os.path.join(midi_dir, f"{filename}.mid"))

        # TODO: get_pmid_segment is broken. Use another slicing method until it is fixed
        # TODO: automatically determine seg_size from dataset_name, or take it as an arg
        piano_slice = get_pmid_segment(
            pmid,
            segment_num=segment_id,
            resolution=4,
            seg_size=1,
            n_beat_bars=4,
            parts=["Piano"],
        )
        drums_slice = get_pmid_segment(
            pmid,
            segment_num=segment_id,
            resolution=4,
            seg_size=1,
            n_beat_bars=4,
            parts=["Drums"],
        )

        # Derive hits representations from MIDI
        piano_subdivisions = get_subdivisions(piano_slice, resolution=4)
        piano_slice_representations = get_representations(
            piano_slice, piano_subdivisions
        )[0]
        piano_hits = (piano_slice_representations["hits"] > 0).astype(np.int8)
        piano_hits_list.append(piano_hits)

        drums_subdivisions = get_subdivisions(drums_slice, resolution=4)
        drums_slice_representations = get_representations(
            drums_slice, drums_subdivisions
        )[0]
        drums_hits = (drums_slice_representations["hits"] > 0).astype(np.int8)
        drums_hits_list.append(drums_hits)

        print(f"Loaded MIDI for {filename} segment {segment_id}")

        # Write MIDI
        output_filename = filename.replace("/", "_")
        piano_slice.write(
            os.path.join(selection_dir, f"{output_filename}_{segment_id}_piano.mid")
        )
        drums_slice.write(
            os.path.join(selection_dir, f"{output_filename}_{segment_id}_drums.mid")
        )

        # Write hits as MIDI
        write_midi_from_hits(
            piano_hits,
            os.path.join(
                selection_dir, f"{output_filename}_{segment_id}_piano_hits.mid"
            ),
            pitch=60,
        )
        write_midi_from_hits(
            drums_hits,
            os.path.join(
                selection_dir, f"{output_filename}_{segment_id}_drums_hits.mid"
            ),
            pitch=36,
        )

    interpolated_piano_hits = interpolate_three_patterns(piano_hits_list, (a, b, c))
    interpolated_drums_hits = interpolate_three_patterns(drums_hits_list, (a, b, c))

    write_midi_from_hits(
        interpolated_piano_hits,
        os.path.join(selection_dir, "interpolated_piano_hits.mid"),
        pitch=60,
    )
    write_midi_from_hits(
        interpolated_drums_hits,
        os.path.join(selection_dir, "interpolated_drums_hits.mid"),
        pitch=36,
    )
    print("Saved interpolated patterns")

    # TODO: interpolate in the MIDI domain
