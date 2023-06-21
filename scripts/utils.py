import os
import subprocess

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.manifold import MDS, TSNE


sns.set_style("white")
sns.set_context("paper")


def save_fig(filepath, title=None):
    """Save a figure to a file and close it"""
    if title:
        plt.title(title)
    plt.tight_layout()
    plt.savefig(filepath)
    print(f"Saved {filepath}")
    plt.close()


def play_midi_file(filepath):
    """Play a MIDI file using FluidSynth"""
    print("Playing MIDI...")
    subprocess.check_output(
        [
            "fluidsynth",
            "-i",  # deactivates the shell & causes FluidSynth to quit as soon as MIDI playback is completed
            "/usr/local/share/fluidsynth/generaluser.v.1.471.sf2",
            filepath,
        ]
    )


def get_embeddings(
    X,
    filenames,
    segment_ids,
    method="t-SNE",
    title="",
    outdir=".",
    dataset_name="",
    normalize=True,
):
    """Create a 2D embedding space of the data, plot it, and save it to a csv"""
    reducer = TSNE(n_components=2, init="pca", learning_rate="auto", random_state=42)
    if method.lower() == "mds":
        reducer = MDS(n_components=2, n_init=1, random_state=42)

    # Make the embedding space
    print(f"Performing {method}...")
    X_transform = reducer.fit_transform(X)

    emb = pd.DataFrame(X_transform, columns=["x", "y"])

    # Normalize embeddings to [0, 1] for ease of use
    if normalize:
        emb = (emb - emb.min()) / (emb.max() - emb.min())

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

    # Add text labels if the space is small enough
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
