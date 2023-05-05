import os
import subprocess

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.manifold import TSNE

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


def get_embeddings(X, title="", outdir="."):
    """Create a 2D embedding space of the data, plot it, and save it to a csv"""
    reducer = TSNE(n_components=2, init="pca", learning_rate="auto", random_state=42)

    # Make the pair space using t-SNE
    X_transform = reducer.fit_transform(X)

    emb = pd.DataFrame(X_transform, columns=["x", "y"])
    sns.relplot(
        data=emb,
        x="x",
        y="y",
        height=8,
        aspect=1.25,
        legend=False,
    )

    save_fig(os.path.join(outdir, "latent_samples.png"), title=title)
    return emb
