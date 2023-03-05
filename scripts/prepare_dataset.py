import argparse
import glob
import itertools
import os
import sys
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rhythmic_complements import (
    ANNOTATIONS_FILENAME,
    DATASETS_DIR,
    PAIR_LOOKUPS_DIRNAME,
    REPRESENTATIONS_DIRNAME,
)
from rhythmic_complements.io import load_midi_file, write_image_from_roll
from rhythmic_complements.parts import PARTS, get_part_from_program, get_part_pairs
from rhythmic_complements.representations import parse_representations
from rhythmtoolbox import pianoroll2descriptors
from tqdm import tqdm

# Segments with little activity will be filtered out
# TODO: take these as args
MIN_SEG_PITCHES = 1
MIN_SEG_BEATS = 4


def roll_has_activity(roll, min_pitches=MIN_SEG_PITCHES, min_beats=MIN_SEG_BEATS):
    """Verify that a piano roll has at least some number of beats and pitches"""
    n_pitches = (roll.sum(axis=0) > 0).sum()
    n_beats = (roll.sum(axis=1) > 0).sum()
    return (n_pitches >= min_pitches) and (n_beats >= min_beats)


def resize_roll_to_n_beats(roll, n_beats, resolution):
    """Pad/truncate a piano roll to a beat length"""
    n_ticks = n_beats * resolution
    if len(roll) < n_ticks:
        pad_right = np.zeros((n_ticks - roll.shape[0], roll.shape[1]))
        roll = np.vstack((roll, pad_right)).astype(np.uint8)
    elif len(roll) > n_ticks:
        roll = roll[:n_ticks]
    return roll


def slice_midi_file(
    filepath,
    seg_size=1,
    resolution=4,
    n_beat_bars=4,
    binarize=False,
    write_images=False,
    im_size=None,
):
    """Slice a midi file and compute rhythmic descriptors for each segment.

    Parameters

        filepath: str
            Path to the input: either a MIDI file or a directory of MIDI files.

        seg_size: int
            Number of bars per segment.

        resolution: int
            Number of subdivisions per beat.

        n_beat_bars: int
            Process only segments with this number of beats per bar.

        prefix: str
            An identifier for output filenames.

        write_images, bool
            Write images of the piano rolls.

        im_size, tuple
            Specify target dimensions of the image. Roll will be padded with 0s on right and bottom.
    """

    pmid = load_midi_file(filepath)
    if not pmid:
        return None, None

    tracks, bar_start_ticks = parse_representations(pmid, resolution, binarize)

    seg_iter = list(zip(bar_start_ticks, bar_start_ticks[seg_size:]))
    if len(bar_start_ticks) <= seg_size:
        # There is only one segment in the track
        seg_iter = [(0, resolution * n_beat_bars * seg_size)]

    # Initialize output objects
    n_seg_ticks = resolution * n_beat_bars * seg_size
    part_seg_rolls = defaultdict(list)
    seg_list = []

    # Initialize output directories
    if write_images:
        im_dir = os.path.join(outdir, "seg_rolls")
        if im_size:
            im_dir = os.path.join(outdir, f"seg_rolls_{im_size[0]}x{im_size[1]}")
        if not os.path.isdir(im_dir):
            os.makedirs(im_dir)

    for track in tracks:
        roll = track["roll"]
        chroma = track["chroma"]
        hits = track["hits"]
        pattern = track["pattern"]

        part = get_part_from_program(track["program"])
        if track["is_drum"]:
            part = "Drums"

        # Slice the piano roll into segments of equal length
        for seg_ix, (start, end) in enumerate(seg_iter):
            seg_chroma = chroma[start:end]

            # Skip uninteresting segments
            if not roll_has_activity(seg_chroma):
                continue

            seg_roll = roll[start:end]
            seg_hits = hits[start:end]
            seg_pattern = pattern[start:end]

            # Skip segments that aren't the target number of beats
            if len(seg_roll) != n_seg_ticks:
                continue

            # Compute the `descriptors` representation of the roll
            seg_descriptors = np.array(
                list(pianoroll2descriptors(seg_roll.T, resolution=resolution).values()),
                dtype=np.float32,
            )

            # Save all representations together
            # IMPORTANT: these should be in the same order as `rhythmic_complements.representations.REPRESENTATIONS`
            part_seg_rolls[f"{seg_ix}_{part}"].append(
                np.array(
                    [seg_roll, seg_chroma, seg_pattern, seg_hits, seg_descriptors],
                    dtype="object",
                )
            )

            # Save part metadata for the segment
            seg_list.append([seg_ix, part])

            if write_images:
                seg_name = f"bar{seg_ix * seg_size}_{seg_ix * seg_size + seg_size}_subdivision{start}-{end}"
                img_outpath = os.path.join(im_dir, f"{seg_name}.png")
                write_image_from_roll(seg_roll, img_outpath, im_size=im_size)

    return part_seg_rolls, seg_list


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path",
        type=str,
        default="../input/lmd_clean",
        help="Path to the input: either a MIDI file or a directory of MIDI files.",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default="lmd_clean",
        help="An identifier for output filenames.",
    )
    parser.add_argument(
        "--seg_size",
        type=int,
        default=1,
        help="Number of bars per segment.",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=4,
        help="Number of subdivisions per beat.",
    )
    parser.add_argument(
        "--binarize",
        action="store_true",
        help="Replace [0,127] MIDI velocity values in piano rolls with binary values representing onsets.",
    )
    parser.add_argument(
        "--n_beat_bars",
        type=int,
        default=4,
        help="Process only segments with this number of beats per bar.",
    )
    parser.add_argument(
        "--subset",
        type=int,
        default=None,
        help="Number of MIDI files to process, for when you don't want to process everything in the directory.",
    )
    parser.add_argument(
        "--write_images",
        action="store_true",
        help="Write images of the piano rolls.",
    )
    parser.add_argument(
        "--im_size",
        type=str,
        default=None,
        help="A resolution to use for the piano roll images, e.g. 512x512.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print debug statements.",
    )
    args = parser.parse_args()

    path = args.path
    seg_size = args.seg_size
    resolution = args.resolution
    binarize = args.binarize
    n_beat_bars = args.n_beat_bars
    prefix = args.prefix if args.prefix else os.path.splitext(path)[0].replace("/", "_")
    write_images = args.write_images
    im_size = args.im_size
    if im_size:
        im_size = (int(x) for x in args.im_size.split("x"))
    subset = args.subset
    # TODO: if args.verbose, set log level to debug

    filepaths = [path]
    if os.path.isdir(path):
        filepaths = glob.glob(os.path.join(path, "**/*.mid"), recursive=True)
        # Adding a trailing slash helps with string splitting later
        path = path + "/" if not path.endswith("/") else path

    filepaths = filepaths[:subset]
    if len(filepaths) == 0:
        print(f"No MIDI files found in {path}")
        sys.exit(0)

    print(f"Processing {len(filepaths)} midi file(s)")

    dataset_name = f"{prefix}_{len(filepaths)}_{seg_size}bar_{resolution}res"
    output_dir = os.path.join(DATASETS_DIR, dataset_name)
    data_dir = os.path.join(output_dir, REPRESENTATIONS_DIRNAME)
    if not os.path.isdir(data_dir):
        os.makedirs(data_dir)

    failed_paths = []
    annotations = {p: [] for p in PARTS}
    annotations_list = []

    for file_ix, filepath in enumerate(tqdm(filepaths)):
        part_seg_rolls, seg_list = slice_midi_file(
            filepath=filepath,
            seg_size=seg_size,
            resolution=resolution,
            binarize=binarize,
            n_beat_bars=n_beat_bars,
            write_images=write_images,
            im_size=im_size,
        )

        if part_seg_rolls is None or not seg_list:
            failed_paths.append(filepath)
            continue

        # Create a unique ID for each file that isn't the input path
        file_id = os.path.splitext(filepath)[0].split(path)[1]

        # Update the top-level annotations
        annotations_list.append([file_id, seg_list])

        # Write the seg_rolls by part
        outpath = os.path.join(data_dir, f"{file_id}.npz")
        outdir = os.path.dirname(outpath)
        if not os.path.isdir(outdir):
            os.makedirs(outdir)
        np.savez_compressed(outpath, **part_seg_rolls)

    # Save the top-level segment map
    adf_files = []
    for i in annotations_list:
        adf_files.extend([i[0]] * len(i[1]))
    annotations_df = pd.DataFrame(
        np.concatenate([i[1] for i in annotations_list]),
        columns=["segment_id", "part_id"],
    )
    annotations_df["file_id"] = adf_files

    annotations_path = os.path.join(output_dir, ANNOTATIONS_FILENAME)
    annotations_df.to_csv(annotations_path, index=False)
    print(f"Saved {annotations_path}")

    # Save lookup tables for segment pairs
    print("Creating segment pair lookups...")
    pair_lookups = defaultdict(list)
    for group_ix, group in tqdm(annotations_df.groupby(["file_id", "segment_id"])):
        for p in get_part_pairs(group.part_id.unique()):
            p1 = group[group.part_id == p[0]].index.values
            p2 = group[group.part_id == p[1]].index.values
            product = list(itertools.product(p1, p2))
            pair_lookups[f"{p[0]}_{p[1]}"].extend(product)

    pair_lookups_dir = os.path.join(output_dir, PAIR_LOOKUPS_DIRNAME)
    if not os.path.isdir(pair_lookups_dir):
        os.makedirs(pair_lookups_dir)
    for pair_id in pair_lookups:
        pair_df = pd.DataFrame(pair_lookups[pair_id], columns=pair_id.split("_"))
        pair_df_path = os.path.join(pair_lookups_dir, f"{pair_id}.csv")
        pair_df.to_csv(pair_df_path, index=False)

    # Plot the percent of segments by part
    part_counts = annotations_df.part_id.value_counts()
    n_segments = part_counts.sum()
    part_pcts = part_counts / n_segments
    fig, ax = plt.subplots(figsize=(20, 8))
    part_pcts.sort_values(ascending=False).plot(kind="bar")
    plt.setp(ax.get_xticklabels(), ha="right", rotation=45)
    plt.title(f"Distribution of parts in {dataset_name}\n{n_segments} segments total")
    plt.ylabel("Fraction of segments")
    plt.tight_layout()
    dist_plot_path = os.path.join(output_dir, "segments_by_part.png")
    plt.savefig(dist_plot_path)
    print(f"Saved {dist_plot_path}")

    # Plot the percent of segments by part pair
    # TODO: reduce xtick font size a bit and make the figsize a bit wider
    pair_counts = pd.Series({k: len(v) for k, v in pair_lookups.items()})
    n_pairs = pair_counts.sum()
    pair_pcts = pair_counts / n_pairs
    fig, ax = plt.subplots(figsize=(20, 8))
    pair_pcts.sort_values(ascending=False).plot(kind="bar")
    plt.setp(ax.get_xticklabels(), ha="right", rotation=45)
    plt.title(f"Distribution of part pairs in {dataset_name}\n{n_pairs} pairs total")
    plt.ylabel("Fraction of segments")
    plt.tight_layout()
    pair_dist_plot_path = os.path.join(output_dir, "segments_by_part_pair.png")
    plt.savefig(pair_dist_plot_path)
    print(f"Saved {pair_dist_plot_path}")

    n_failed = len(failed_paths)
    if n_failed > 0:
        failed_paths_str = "\n".join(failed_paths)
        print(
            f"Successfully processed {len(filepaths) - n_failed} file(s); Failed to process {n_failed}"
        )
