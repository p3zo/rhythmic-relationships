import argparse
import glob
import os
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pretty_midi
import pypianoroll
from rhythm_descriptors import pattlist2descriptors

INPUT_DIR = "input"
OUTPUT_DIR = "output"


def load_midi_file(filepath):
    # Warnings can be verbose when midi has no metadata e.g. tempo, key, time signature
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        try:
            midi = pretty_midi.PrettyMIDI(filepath)
        except Exception as e:
            print(f"Failed loading file {filepath}: {e}")
            return

    return midi


def get_bar_start_times(pmid, beat_division=4):
    """Adapted from https://github.com/ruiguo-bio/midi-miner/blob/master/tension_calculation.py#L687-L718"""

    beats = pmid.get_beats()
    beats = np.unique(beats, axis=0)

    subdivision_beats = []
    for i in range(len(beats) - 1):
        for j in range(beat_division):
            subdivision_beats.append(
                (beats[i + 1] - beats[i]) / beat_division * j + beats[i]
            )
    subdivision_beats.append(beats[-1])
    subdivision_beats = np.unique(subdivision_beats, axis=0)

    beat_ixs = []
    for beat in beats:
        beat_ixs.append(np.argwhere(subdivision_beats == beat)[0][0])

    # Bar starts
    down_beats = pmid.get_downbeats()
    if len(down_beats) > 1 and subdivision_beats[-1] > down_beats[-1]:
        down_beats = np.append(
            down_beats, down_beats[-1] - down_beats[-2] + down_beats[-1]
        )
    down_beats = np.unique(down_beats, axis=0)

    down_beat_ixs = []
    for down_beat in down_beats:
        down_beat_ixs.append(np.argmin(np.abs(down_beat - subdivision_beats)))

    return down_beats, down_beat_ixs


def plot_roll(roll, ax):
    pypianoroll.plot_pianoroll(
        ax=ax,
        pianoroll=roll,
        xticklabel=False,
        preset="frame",
        ytick="step",
        yticklabel="name",
        is_drum=True,
        grid_axis="off",
        cmap="Blues",
    )


def pianoroll_to_pattlist(roll):
    pattlist = []
    for t in roll:
        pattlist.append([ix for ix, i in enumerate(t) if i > 0])
    return pattlist


def analyze(filepath, seg_size=1, resolution=4, prefix="", plot_segments=True):
    """Slices a midi file and computes rhythmic descriptors for each segment.

    Parameters

    filepath: str
        Path to the input: either a MIDI file or a directory of MIDI files

    seg_size: int
        Number of bars per segment

    resolution: int
        Number of subdivisions per beat

    prefix: str
        An identifier for output filenames.

    plot_segments: bool
        Create piano roll plots for each segment.
    """

    mid_name = f"{prefix}_{os.path.splitext(os.path.basename(filepath))[0]}"
    print(f"Analyzing {mid_name}")

    pmid = load_midi_file(filepath)

    bar_start_times, bar_start_ixs = get_bar_start_times(pmid, resolution)

    # TODO: segments are empty when resolution=4, but not when resolution=24
    # but when resolution=24, notes are missing from the overall pianoroll
    multitrack = pypianoroll.read(filepath, resolution=resolution)

    roll = multitrack[0].pianoroll

    # Plot the piano roll of the entire track
    fig, ax = plt.subplots(figsize=(20, 8))
    plot_roll(roll, ax)

    # Add bar lines
    for t in bar_start_times:
        ax.axvline(x=t * multitrack.resolution, color="black", linewidth=0.1)

    plt.title(mid_name)

    plot_dir = os.path.join(OUTPUT_DIR, f"{mid_name}")
    if not os.path.isdir(plot_dir):
        os.makedirs(plot_dir)
    plot_path = os.path.join(plot_dir, "roll.png")
    plt.savefig(plot_path)
    plt.clf()
    print(f"  Saved {plot_path}")

    # Slice into n-bar segments
    segments = []
    segment_descriptors = []

    seg_iter = list(
        zip(bar_start_ixs[::seg_size], bar_start_ixs[seg_size:][::seg_size])
    )

    if len(seg_iter) == 0:
        segments = [roll]
        pattlist = pianoroll_to_pattlist(roll)
        segment_descriptors = [pattlist2descriptors(pattlist)]

    for i, (start, end) in enumerate(seg_iter):
        segment = roll[start:end]
        segments.append(segment)

        if plot_segments:
            plot_roll(segment, ax)
            plt.title(f"{mid_name}\nbar {i*seg_size}\n{start} - {end}")

            seg_plot_dir = os.path.join(plot_dir, f"segments-{seg_size}-bar")
            if not os.path.isdir(seg_plot_dir):
                os.makedirs(seg_plot_dir)
            seg_plot_path = os.path.join(seg_plot_dir, f"{start}-{end}.png")

            plt.savefig(seg_plot_path)
            plt.clf()
            print(f"  Saved {seg_plot_path}")

        # Compute rhythmic descriptors for each segment
        pattlist = pianoroll_to_pattlist(segment)
        segment_descriptors.append(pattlist2descriptors(pattlist))

    print(f"  {len(segments)} {seg_size}-bar segments analyzed")

    df = pd.DataFrame(segment_descriptors)
    df.index.name = "segment_id"
    df.reset_index(inplace=True)

    analysis_dir = os.path.join(OUTPUT_DIR, mid_name)
    if not os.path.isdir(analysis_dir):
        os.makedirs(analysis_dir)
    analysis_path = os.path.join(analysis_dir, f"descriptors-{seg_size}-bar.csv")
    df.to_csv(analysis_path, index=False)
    print(f"  Descriptors written to {analysis_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path",
        type=str,
        help="Path to the input: either a MIDI file or a directory of MIDI files.",
    )
    parser.add_argument(
        "--segment_length", type=int, default=1, help="Number of bars per segment."
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=4,
        help="Number of subdivisions per beat. Default = 4, i.e. 16th note.",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default="",
        help="An identifier for output filenames.",
    )
    parser.add_argument("--plot_segments", action="store_true", help="Create plots.")
    args = parser.parse_args()

    inpath = args.path
    segment_length = args.segment_length
    resolution = args.resolution
    prefix = args.prefix
    plot_segments = args.plot_segments

    if os.path.isdir(inpath):
        # Analyze all files in the directory
        for fp in glob.glob(os.path.join(inpath, "*.mid")):
            analyze(
                fp,
                seg_size=segment_length,
                resolution=resolution,
                prefix=prefix,
                plot_segments=plot_segments,
            )
    else:
        # Analyze a single file
        analyze(
            inpath,
            seg_size=segment_length,
            resolution=resolution,
            prefix=prefix,
            plot_segments=plot_segments,
        )
