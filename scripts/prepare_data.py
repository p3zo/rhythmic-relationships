import argparse
import glob
import itertools
import os
import sys
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pypianoroll
from rhythmtoolbox import pianoroll2descriptors, resample_pianoroll
from tqdm import tqdm

from rhythmic_complements import DATASETS_DIR
from rhythmic_complements.io import load_midi_file, write_image_from_roll
from rhythmic_complements.parts import PARTS, get_part_from_program, get_part_pairs


# Piano key numbers
MIDI_PITCH_RANGE = [21, 108]
N_MIDI_PITCHES = MIDI_PITCH_RANGE[1] - MIDI_PITCH_RANGE[0] + 1

# Segments with little activity will be filtered out
MIN_SEG_PITCHES = 1
MIN_SEG_BEATS = 4

global VERBOSE
VERBOSE = False


def get_9voice_drum_roll(roll):
    """Convert a piano roll to a 9-voice roll for drums"""

    # Follows paper mapping in https://magenta.tensorflow.org/datasets/groove
    drum_map_9_voice = {
        36: 36,
        38: 38,
        40: 38,
        37: 38,
        48: 50,
        50: 50,
        45: 47,
        47: 47,
        43: 43,
        58: 43,
        46: 46,
        26: 46,
        42: 42,
        22: 42,
        44: 42,
        49: 49,
        55: 49,
        57: 49,
        52: 49,
        51: 51,
        59: 51,
        53: 51,
    }

    voices = list(np.unique(list(drum_map_9_voice.values())))

    clip = np.zeros((len(roll), 9))
    for event_ix in range(len(clip)):
        for inst_ix, i in enumerate(roll[event_ix]):
            if i > 0 and inst_ix in drum_map_9_voice:
                clip_ix = voices.index(drum_map_9_voice.get(inst_ix))
                clip[event_ix][clip_ix] = i

    return clip


def roll_has_activity(roll, min_pitches=MIN_SEG_PITCHES, min_beats=MIN_SEG_BEATS):
    """Verify that a piano roll has at least some number of beats and pitches"""
    n_pitches = (roll.sum(axis=0) > 0).sum()
    n_beats = (roll.sum(axis=1) > 0).sum()
    return (n_pitches >= min_pitches) and (n_beats >= min_beats)


def get_bar_start_ixs(pmid, resolution):
    """Adapted from https://github.com/ruiguo-bio/midi-miner/blob/master/tension_calculation.py#L687-L718
    TODO: Replace this with pypianoroll downbeats
    """

    beats = pmid.get_beats()
    beats = np.unique(beats, axis=0)

    subdivision_beats = []
    for i in range(len(beats) - 1):
        for j in range(resolution):
            subdivision_beats.append(
                (beats[i + 1] - beats[i]) / resolution * j + beats[i]
            )
    subdivision_beats.append(beats[-1])
    subdivision_beats = np.unique(subdivision_beats, axis=0)

    # Bar start times
    down_beats = pmid.get_downbeats()
    if len(down_beats) > 1 and subdivision_beats[-1] > down_beats[-1]:
        down_beats = np.append(
            down_beats, down_beats[-1] - down_beats[-2] + down_beats[-1]
        )
    down_beats = np.unique(down_beats, axis=0)

    # Bar start ixs
    down_beat_ixs = []
    for down_beat in down_beats:
        down_beat_ixs.append(np.argmin(np.abs(down_beat - subdivision_beats)))

    return down_beat_ixs


def get_hits_from_roll(roll):
    """A binary vector of onsets. `0` is a silence and `1` is an onset."""
    # TODO: preserve only the onsets of held notes
    return (roll.sum(axis=1) > 0).astype(int)


def get_pattern_from_roll(roll, resolution, seg_size, binarized=False):
    """A `pattern` is a ternary vector of onsets and offsets. `0` is a silence, `1` is an onset, and `2` is a
    continuation of a previous onset.

    Adjacent nonzero values of the same pitch will be considered a single note with their mean as its velocity.
    TODO: parse patterns from MIDI to retain accurate onsets/offsets

    Adapted from https://salu133445.github.io/pypianoroll/_modules/pypianoroll/outputs.html#to_pretty_midi
    """

    if not binarized:
        roll = roll > 0

    padded = np.pad(roll, ((1, 1), (0, 0)), "constant")
    diff = np.diff(padded.astype(np.int8), axis=0)

    onsets = np.nonzero(diff > 0)[0]
    offsets = np.nonzero(diff < 0)[0]

    pattern = np.zeros(resolution * 4 * seg_size)
    for onset, offset in zip(onsets, offsets):
        pattern[onset] = 1
        pattern[onset + 1 : offset] = 2

    return pattern


def get_binary_chroma_from_roll(roll, resolution, seg_size, binarized=False):
    """A `chroma` is an N x 12 matrix of pitch class activations, where N is a number of time steps.
    See https://en.wikipedia.org/wiki/Chroma_feature for background on chroma features.
    A binary chroma does not retain velocities, so 0 is a silence and 1 is an onset.
    TODO: write a test for this
    """
    if not binarized:
        roll = roll > 0

    n = resolution * 4 * seg_size
    chroma = np.zeros((n, 12), np.int8)

    # Squeeze pitches into one octave
    # TODO: make this more efficient
    # TODO: need 2s for held notes, like pattern
    octaves = [i for i in range(0, roll.shape[1] + 12, 12)]
    for low, high in [o for o in zip(octaves, octaves[1:])]:
        oct = roll[:, low:high]
        if oct.shape[1] < 12:
            pad_top = np.zeros((n, 12 - oct.shape[1]), np.uint8)
            oct = np.hstack((pad_top, oct))
        chroma = np.add(chroma, oct)

    chroma = (chroma > 0).astype(np.uint8)

    return chroma


def get_segment_iterator(pmid, resolution, seg_size, track_len, overlapping=True):
    """Create an iterable with segment start/end indices.

    :param pmid, pretty_midi.PrettyMIDI
        Pretty MIDI object

    :param resolution, int
        Number of MIDI ticks per beat

    :param seg_size:
        Length of each segment in bars

    :param track_len:
        Length of the entire track, in time steps

    :param overlapping, bool
        Create the iterator using a sliding window with 1-bar hop-size
    """

    bar_ixs = get_bar_start_ixs(pmid, resolution)

    seg_iter = list(zip(bar_ixs, bar_ixs[seg_size:]))
    if not overlapping:
        seg_iter = list(zip(bar_ixs[::seg_size], bar_ixs[seg_size:][::seg_size]))

    if len(seg_iter) == 0:
        seg_iter = [(0, track_len)]

    return seg_iter


def remap_velocities(roll):
    # Remap velocities higher than the standard range
    # TODO: clip them instead?
    if roll.max() > 127:
        roll = np.array(
            list(
                map(
                    lambda x: np.interp(x, [0, roll.max()], [0, 127]),
                    roll,
                )
            ),
            dtype=np.uint8,
        )

    # Convert MIDI velocities to real numbers in [0, 1]
    return np.array(
        list(
            map(
                lambda x: np.interp(x, [0, 127], [0, 1]),
                roll,
            )
        ),
        dtype=np.uint8,
    )


def slice_midi_file(
    filepath,
    seg_size=1,
    resolution=24,
    binarize=False,
    resize_bars_to_4_beats=False,
    drum_roll=False,
    create_images=False,
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

        resize_bars_to_4_beats: bool
            Pad/truncate every segment to the same, 4/4 length

        prefix: str
            An identifier for output filenames.

        drum_roll, bool
            Converts the piano roll to a 9-voice roll for drums.

        create_images, bool
            Create images of the piano rolls.

        im_size, tuple
            Specify target dimensions of the image. Roll will be padded with 0s on right and bottom.
    """

    pmid = load_midi_file(filepath, resolution=resolution, verbose=VERBOSE)
    if not pmid:
        return None, None

    # There seems to be a bug in pypianoroll. Skip the files it can't parse.
    try:
        multitrack = pypianoroll.from_pretty_midi(pmid, resolution=resolution)
    except:
        return None, None

    # Create an iterable to segment each track's piano roll equally
    track_len = len(multitrack.tempo)
    seg_iter = get_segment_iterator(
        pmid, resolution, seg_size, track_len, overlapping=True
    )

    # Initialize output objects
    n_seg_ticks = resolution * 4 * seg_size
    part_segrolls = defaultdict(list)
    seg_list = []

    # Initialize output directories
    if create_images:
        im_dir = os.path.join(outdir, f"segrolls_{n_seg_ticks}x{N_MIDI_PITCHES}")
        if im_size:
            im_dir = os.path.join(outdir, f"segrolls_{im_size[0]}x{im_size[1]}")
        if not os.path.isdir(im_dir):
            os.makedirs(im_dir)

    non_empty_tracks = [t for t in multitrack.tracks if len(t) > 0]

    for track_ix, track in enumerate(non_empty_tracks):

        # Trim the top and bottom pitches of the roll
        roll = track.pianoroll[:, MIDI_PITCH_RANGE[0] : MIDI_PITCH_RANGE[1] + 1]

        part = get_part_from_program(track.program)
        if track.is_drum:
            part = "Drums"

            if drum_roll:
                roll = get_9voice_drum_roll(roll)

        # Slice the piano roll into segments of equal length
        for seg_ix, (start, end) in enumerate(seg_iter):
            seg_name = f"bar{seg_ix * seg_size}_{seg_ix * seg_size + seg_size}_subdivision{start}-{end}"

            segroll = roll[start:end]

            # Skip uninteresting segments
            if not roll_has_activity(segroll):
                continue

            if len(segroll) != n_seg_ticks:
                # Skip segments that aren't the same length
                if not resize_bars_to_4_beats:
                    continue

                # Pad/truncate every segment to the same, 4/4 length
                if len(segroll) < n_seg_ticks:
                    pad_right = np.zeros(
                        (n_seg_ticks - segroll.shape[0], N_MIDI_PITCHES)
                    )
                    segroll = np.vstack((segroll, pad_right)).astype(np.uint8)
                elif len(segroll) > n_seg_ticks:
                    segroll = segroll[:n_seg_ticks]

            if binarize:
                segroll = (segroll > 0).astype(np.uint8)

            else:
                segroll = remap_velocities(segroll)

            # Resample the roll to 4 ticks per beat to create the other representations
            resampled = resample_pianoroll(
                segroll, from_resolution=resolution, to_resolution=4
            )

            # Create the `hits` representation of the roll
            hits = get_hits_from_roll(resampled)

            # Create the `pattern` representation of the roll
            pattern = get_pattern_from_roll(resampled, 4, seg_size, binarized=binarize)

            # Create the `chroma` representation of the roll
            chroma = get_binary_chroma_from_roll(
                resampled, 4, seg_size, binarized=binarize
            )

            # Create the `descriptors` representation of the roll
            descriptors = np.array(
                list(pianoroll2descriptors(resampled, resolution=4).values())
            )

            # Save all the representations together
            part_segrolls[f"{seg_ix}_{part}"].append(
                np.array([segroll, hits, pattern, chroma, descriptors], dtype="object")
            )

            # Save part metadata for the segment
            seg_list.append([seg_ix, part])

            if create_images:
                img_outpath = os.path.join(im_dir, f"{seg_name}.png")
                write_image_from_roll(
                    segroll, img_outpath, im_size=im_size, verbose=VERBOSE
                )

    return part_segrolls, seg_list


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
        default=24,
        help="Number of subdivisions per beat.",
    )
    parser.add_argument(
        "--binarize",
        action="store_true",
        help="Replace [0,127] MIDI velocity values in piano rolls with binary values representing onsets.",
    )
    parser.add_argument(
        "--resize_bars_to_4_beats",
        action="store_true",
        help="Pad/truncate every segment to the same, 4/4 length",
    )
    parser.add_argument(
        "--subset",
        type=int,
        default=None,
        help="Number of MIDI files to process, for when you don't want to process everything in the directory.",
    )
    parser.add_argument(
        "--drum_roll",
        action="store_true",
        help="Use a 9-voice piano roll for drums only.",
    )
    parser.add_argument(
        "--create_images",
        action="store_true",
        help="Create images of the piano rolls.",
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
    resize_bars_to_4_beats = args.resize_bars_to_4_beats
    prefix = args.prefix if args.prefix else os.path.splitext(path)[0].replace("/", "_")
    create_images = args.create_images
    im_size = args.im_size
    if im_size:
        im_size = (int(x) for x in args.im_size.split("x"))
    drum_roll = args.drum_roll
    subset = args.subset
    VERBOSE = args.verbose

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
    data_dir = os.path.join(output_dir, "rolls")
    if not os.path.isdir(data_dir):
        os.makedirs(data_dir)

    failed_paths = []
    annotations = {p: [] for p in PARTS}
    annotations_list = []

    for file_ix, filepath in enumerate(tqdm(filepaths)):
        part_segrolls, seg_list = slice_midi_file(
            filepath=filepath,
            seg_size=seg_size,
            resolution=resolution,
            binarize=binarize,
            resize_bars_to_4_beats=resize_bars_to_4_beats,
            drum_roll=drum_roll,
            create_images=create_images,
            im_size=im_size,
        )

        if part_segrolls is None or not seg_list:
            failed_paths.append(filepath)
            continue

        # Create a unique ID for each file that isn't the input path
        file_id = os.path.splitext(filepath)[0].split(path)[1]

        # Update the top-level annotations
        annotations_list.append([file_id, seg_list])

        # Write the segrolls by part
        outpath = os.path.join(data_dir, f"{file_id}.npz")
        outdir = os.path.dirname(outpath)
        if not os.path.isdir(outdir):
            os.makedirs(outdir)
        np.savez_compressed(outpath, **part_segrolls)

    # Save the top-level segment map
    adf_files = []
    for i in annotations_list:
        adf_files.extend([i[0]] * len(i[1]))
    annotations_df = pd.DataFrame(
        np.concatenate([i[1] for i in annotations_list]),
        columns=["segment_id", "part_id"],
    )
    annotations_df["file_id"] = adf_files
    annotations_path = os.path.join(output_dir, "rolls.csv")
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

    pair_lookups_dir = os.path.join(output_dir, "pair_lookups")
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
