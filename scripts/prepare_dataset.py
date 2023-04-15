"""
Create a new dataset

Example usage:
    python scripts/prepare_dataset.py --path input/babyslakh --prefix babyslakh --seg_size 1 --binarize
"""
import argparse
import glob
import itertools
import logging
import os
import sys
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rhythmic_relationships import (
    ANNOTATIONS_FILENAME,
    DATASETS_DIR,
    PAIR_LOOKUPS_DIRNAME,
    PLOTS_DIRNAME,
    REPRESENTATIONS_DIRNAME,
    logger,
)
from rhythmic_relationships.io import load_midi_file
from rhythmic_relationships.parts import PARTS, get_part_pairs
from rhythmic_relationships.representations import slice_midi
from tqdm import tqdm

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path",
        type=str,
        default="input/babyslakh",
        help="Path to the input: either a MIDI file or a directory of MIDI files.",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default="babyslakh",
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
        "--min_seg_pitches",
        type=int,
        default=1,
        help="Process only segments with at least this number of pitches.",
    )
    parser.add_argument(
        "--min_seg_beats",
        type=int,
        default=1,
        help="Process only segments with at least this number of beats.",
    )
    parser.add_argument(
        "--subset",
        type=int,
        default=None,
        help="Number of MIDI files to process, for when you don't want to process everything in the directory.",
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
    subset = args.subset
    min_seg_pitches = args.min_seg_pitches
    min_seg_beats = args.min_seg_beats

    if args.verbose:
        logger.setLevel(logging.DEBUG)

    if not os.path.exists(path):
        logger.error(f"Path does not exist: {path}")
        sys.exit(0)

    filepaths = [path]
    if os.path.isdir(path):
        filepaths = glob.glob(os.path.join(path, "**/*.mid"), recursive=True)
        # Adding a trailing slash helps with string splitting later
        path = path + "/" if not path.endswith("/") else path

    filepaths = filepaths[:subset]
    if len(filepaths) == 0:
        logger.error(f"No MIDI files found in {path}")
        sys.exit(0)

    dataset_name = f"{prefix}_{len(filepaths)}_{seg_size}bar_{resolution}res"
    output_dir = os.path.join(DATASETS_DIR, dataset_name)
    data_dir = os.path.join(output_dir, REPRESENTATIONS_DIRNAME)
    if not os.path.isdir(data_dir):
        os.makedirs(data_dir)

    failed_paths = []
    annotations = {p: [] for p in PARTS}
    annotations_list = []

    logger.info(f"Processing {len(filepaths)} midi file(s)")
    for file_ix, filepath in enumerate(tqdm(filepaths)):
        pmid = load_midi_file(filepath)
        if not pmid:
            failed_paths.append(filepath)
            continue

        seg_part_reprs = slice_midi(
            pmid=pmid,
            seg_size=seg_size,
            resolution=resolution,
            binarize=binarize,
            n_beat_bars=n_beat_bars,
            min_seg_pitches=min_seg_pitches,
            min_seg_beats=min_seg_beats,
        )

        seg_list = [i.split("_") for i in seg_part_reprs.keys()]

        if seg_part_reprs is None or not seg_list:
            failed_paths.append(filepath)
            continue

        # Save segment metadata
        file_id = os.path.splitext(filepath)[0].split(path)[1]
        annotations_list.append([file_id, seg_list])

        # Save segment-part representations
        outpath = os.path.join(data_dir, f"{file_id}.npz")
        outdir = os.path.dirname(outpath)
        if not os.path.isdir(outdir):
            os.makedirs(outdir)
        np.savez_compressed(outpath, **seg_part_reprs)

    # Save the segment map
    annotations_path = os.path.join(output_dir, ANNOTATIONS_FILENAME)
    annotations_df = pd.DataFrame(
        np.concatenate([i[1] for i in annotations_list]),
        columns=["segment_id", "part_id"],
    )
    annotations_df["segment_id"] = annotations_df["segment_id"].astype(int)
    adf_files = []
    for i in annotations_list:
        adf_files.extend([i[0]] * len(i[1]))
    annotations_df["file_id"] = adf_files
    annotations_df.to_csv(annotations_path, index=False)
    logger.info(f"Saved {annotations_path}")

    # Save lookup tables for segment pairs
    logger.info("Creating segment pair lookups...")
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

    # Initialize plots dir
    plots_dir = os.path.join(output_dir, PLOTS_DIRNAME)
    if not os.path.isdir(plots_dir):
        os.makedirs(plots_dir)

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
    dist_plot_path = os.path.join(plots_dir, "segments_by_part.png")
    plt.savefig(dist_plot_path)
    logger.info(f"Saved {dist_plot_path}")

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
    pair_dist_plot_path = os.path.join(plots_dir, "segments_by_part_pair.png")
    plt.savefig(pair_dist_plot_path)
    logger.info(f"Saved {pair_dist_plot_path}")

    n_failed = len(failed_paths)
    if n_failed > 0:
        failed_paths_str = "\n".join(failed_paths)
        logger.info(
            f"Successfully processed {len(filepaths) - n_failed} files; Failed to process {n_failed}"
        )
