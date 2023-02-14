import argparse
import glob
import os
import pickle
import sys
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pypianoroll
from rhythmic_complements.io import load_midi_file, write_pil_image
from rhythmtoolbox import pianoroll2descriptors
from tqdm import tqdm

OUTPUT_DIR = "output"

# Piano key numbers
MIDI_PITCH_RANGE = [21, 108]
N_MIDI_PITCHES = MIDI_PITCH_RANGE[1] - MIDI_PITCH_RANGE[0]

# Segments with little activity will be filtered out
MIN_SEG_PITCHES = 1
MIN_SEG_BEATS = 4

# Program categories from the General MIDI Level 2 spec: https://en.wikipedia.org/wiki/General_MIDI_Level_2
# A category's key is the index of its first patch
PROGRAM_CATEGORIES = {
    1: "Piano",
    9: "Chromatic Percussion",
    17: "Organ",
    25: "Guitar",
    33: "Bass",
    41: "Orchestra Solo",
    49: "Orchestra Ensemble",
    57: "Brass",
    65: "Reed",
    73: "Wind",
    81: "Synth Lead",
    89: "Synth Pad",
    97: "Synth Sound FX",
    105: "Ethnic",
    113: "Percussive",
    121: "Sound Effect",
}

# NOTE: In the General MIDI spec, drums are on a separate MIDI channel
PARTS = ["Drums"] + list(PROGRAM_CATEGORIES.values())

global VERBOSE


def get_part_from_program(program):
    if program < 0 or program > 127:
        raise ValueError(
            f"Program number {program} is not in the valid range of [0, 127]"
        )
    return PROGRAM_CATEGORIES[[p for p in PROGRAM_CATEGORIES if p <= program + 1][-1]]


def get_bar_start_times(pmid, beat_division=24):
    """Adapted from https://github.com/ruiguo-bio/midi-miner/blob/master/tension_calculation.py#L687-L718
    TODO: Replace this with pypianoroll downbeats
    """

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
    """Plot a numpy array piano roll"""
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


def plot_multitrack(multitrack, output_dir, bar_start_times):
    """Plot the multitrack piano roll of a pretty_midi object"""
    fig, ax = plt.subplots(figsize=(20, 8))

    if len(multitrack.tracks) == 0:
        plot_roll(multitrack[0].pianoroll, ax)
    else:
        multitrack.plot(
            xticklabel=False,
            preset="frame",
            ytick="step",
            yticklabel="name",
            grid_axis="off",
        )

    # Add bar lines
    for t in bar_start_times:
        ax.axvline(x=t * multitrack.resolution, color="black", linewidth=0.1)

    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    plot_path = os.path.join(output_dir, "roll.png")
    plt.savefig(plot_path)
    plt.clf()
    if VERBOSE:
        print(f"  Saved {plot_path}")


def plot_segment(roll, seg_name, outdir):
    # Plot piano rolls as matplotlib plots
    seg_fig, seg_ax = plt.subplots(figsize=(20, 8))
    plot_roll(roll, seg_ax)

    seg_plot_dir = os.path.join(outdir, f"segplots_{seg_size}bar_{resolution}res")
    if not os.path.isdir(seg_plot_dir):
        os.makedirs(seg_plot_dir)
    seg_plot_path = os.path.join(
        seg_plot_dir,
        f"{seg_name}.png",
    )

    plt.savefig(seg_plot_path)
    plt.close()
    if VERBOSE:
        print(f"  Saved {seg_plot_path}")


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


def mk_mid_name(prefix, filepath):
    return f"{prefix}_{os.path.splitext(os.path.basename(filepath))[0]}"


def process(
    filepath,
    output_dir,
    seg_size=2,
    resolution=24,
    drum_roll=False,
    create_images=True,
    im_size=None,
    compute_descriptors=True,
    pypianoroll_plots=False,
):
    """Slice a midi file and compute rhythmic descriptors for each segment.

    Parameters

        filepath: str
            Path to the input: either a MIDI file or a directory of MIDI files.

        output_dir: str,
            Path to a directory in which to write output files.

        seg_size: int
            Number of bars per segment.

        resolution: int
            Number of subdivisions per beat.

        prefix: str
            An identifier for output filenames.

        drum_roll, bool
            Converts the piano roll to a 9-voice roll for drums.

        create_images, bool
            Create images of the piano rolls.

        im_size, tuple
            Specify target dimensions of the image. Roll will be padded with 0s on right and bottom.

        compute_descriptors, bool
            Use rhythmtoolbox to compute rhythmic descriptors for each segment.

        pypianoroll_plots: bool
            Create pypianoroll plots.
    """

    mid_name = mk_mid_name(prefix, filepath)
    mid_outdir = os.path.join(output_dir, mid_name)

    pmid = load_midi_file(filepath, resolution=resolution, verbose=VERBOSE)
    if not pmid:
        return None, None, None

    bar_times, bar_ixs = get_bar_start_times(pmid, resolution)

    # There seems to be an off-by-one bug in pypianoroll. Skip the files it can't parse.
    try:
        multitrack = pypianoroll.from_pretty_midi(pmid, resolution=resolution)
    except:
        return None, None, None

    # Plot the piano roll of the full multitrack midi input
    if pypianoroll_plots:
        plot_multitrack(multitrack, mid_outdir, bar_times)

    # Define an iterable to segment each track's piano roll equally
    # TODO: make overlapping segments (all possible 2-bar segments)
    seg_iter = list(zip(bar_ixs[::seg_size], bar_ixs[seg_size:][::seg_size]))
    # TODO: is this approach simpler?
    # bins = [i for i in range(0, len(bar_ixs)+1, seg_size)]
    # seg_iter = list(zip(bar_ixs], bar_ixs[1:]]))
    if len(seg_iter) == 0:
        # TODO: this assumes all tracks are the same length. Is that always true?
        seg_iter = [(0, len(multitrack[0]))]

    # Initialize output objects
    n_seg_ticks = resolution * 4 * seg_size
    part_segrolls = defaultdict(list)
    part_segdescs = defaultdict(list)
    segdescpairs = {}

    # Initialize output directories
    if create_images:
        im_dir = os.path.join(outdir, f"segrolls_{n_seg_ticks}x{N_MIDI_PITCHES}")
        if im_size:
            im_dir = os.path.join(outdir, f"segrolls_{im_size[0]}x{im_size[1]}")
        if not os.path.isdir(im_dir):
            os.makedirs(im_dir)

    non_empty_tracks = [t for t in multitrack.tracks if len(t) > 0]

    for track_ix, track in enumerate(non_empty_tracks):
        track_dir = os.path.join(mid_outdir, f"track{track_ix}")

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

            # Pad/truncate every 2-bar segment to the same length
            # TODO: this makes everything 4/4. Is that acceptable?
            if len(segroll) < n_seg_ticks:
                pad_right = np.zeros((n_seg_ticks - segroll.shape[0], N_MIDI_PITCHES))
                segroll = np.vstack((segroll, pad_right)).astype(np.uint8)
            elif len(segroll) > n_seg_ticks:
                segroll = segroll[:n_seg_ticks]

            # Ensure all MIDI velocities are in the valid range [0, 127]
            if segroll.max() > 127:
                segroll = np.array(
                    list(
                        map(
                            lambda x: np.interp(x, [0, segroll.max()], [0, 127]),
                            segroll,
                        )
                    ),
                    dtype=np.uint8,
                )

            part_segrolls[part].append(segroll)

            # Create piano roll images using PIL
            if create_images:
                img_outpath = os.path.join(im_dir, f"{seg_name}.png")
                write_pil_image(segroll, img_outpath, im_size=im_size, verbose=VERBOSE)

            if pypianoroll_plots:
                plot_segment(segroll, seg_name, track_dir)

            # Compute rhythmic descriptors for the segment
            if compute_descriptors:
                segdesc = pianoroll2descriptors(segroll, resolution=resolution)
                part_segdescs[part].append(segdesc)

                sdp_ix = f"{mid_name}_{seg_ix}"
                if sdp_ix not in segdescpairs:
                    segdescpairs[sdp_ix] = defaultdict(list)
                segdescpairs[sdp_ix][part].append(segdesc)

    return part_segrolls, part_segdescs, segdescpairs


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path",
        type=str,
        default="input/lmd_clean",
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
        default=2,
        help="Number of bars per segment.",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=24,
        help="Number of subdivisions per beat.",
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
        "--skip_npz",
        action="store_true",
        help="Don't save .npz files.",
    )
    parser.add_argument(
        "--compute_descriptors",
        action="store_true",
        help="Use rhythmtoolbox to compute rhythmic descriptors for each segment.",
    )
    parser.add_argument(
        "--pypianoroll_plots",
        action="store_true",
        help="Create a pypianoroll plot for each segment and another for the entire track.",
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
    prefix = args.prefix if args.prefix else os.path.splitext(path)[0].replace("/", "_")
    create_images = args.create_images
    im_size = args.im_size
    if im_size:
        im_size = (int(x) for x in args.im_size.split("x"))
    pypianoroll_plots = args.pypianoroll_plots
    drum_roll = args.drum_roll
    compute_descriptors = args.compute_descriptors
    skip_npz = args.skip_npz
    VERBOSE = args.verbose

    filepaths = [path]
    if os.path.isdir(path):
        filepaths = glob.glob(os.path.join(path, "**/*.mid"), recursive=True)
        # Adding a trailing slash helps with string splitting later
        path = path + "/" if not path.endswith("/") else path

    n_files = 10000
    filepaths = filepaths[:n_files]
    print(f"Processing {len(filepaths)} midi file(s)")

    dataset_name = f"{prefix}_{seg_size}bar_{resolution}res"
    output_dir = os.path.join(OUTPUT_DIR, dataset_name)
    dataset_dir = os.path.join(output_dir, path.split("/")[-2])
    if not os.path.isdir(dataset_dir):
        os.makedirs(dataset_dir)

    failed_paths = []
    annotations = {p: [] for p in PARTS}
    descriptors = {p: pd.DataFrame() for p in PARTS}
    descriptor_pairs = defaultdict(pd.DataFrame)
    part_counts = defaultdict(int)

    for file_ix, filepath in enumerate(tqdm(filepaths)):
        part_segrolls, part_segdescs, segdescpairs = process(
            filepath=filepath,
            output_dir=output_dir,
            seg_size=seg_size,
            resolution=resolution,
            drum_roll=drum_roll,
            create_images=create_images,
            im_size=im_size,
            compute_descriptors=compute_descriptors,
            pypianoroll_plots=pypianoroll_plots,
        )

        if part_segrolls is None:
            failed_paths.append(filepath)
            continue

        # Save the segments for each part
        outpath = os.path.join(
            dataset_dir, f"{os.path.splitext(filepath)[0].split(path)[1]}.npz"
        )
        if not skip_npz:
            outdir = os.path.dirname(outpath)
            if not os.path.isdir(outdir):
                os.makedirs(outdir)
            np.savez_compressed(outpath, **part_segrolls)

        for part in part_segrolls:
            annotations[part].append(outpath)
            part_counts[part] += 1

        if compute_descriptors:
            for part in part_segdescs:
                df = pd.DataFrame(part_segdescs[part])
                df["segment_id"] = range(len(df))
                df["filepath"] = filepath
                descriptors[part] = pd.concat([descriptors[part], df])

            for seg in segdescpairs:
                part_descs = segdescpairs[seg]
                if "Drums" in part_descs and "Bass" in part_descs:
                    dfd = pd.DataFrame(part_descs["Drums"], dtype=np.float32)
                    dfb = pd.DataFrame(part_descs["Bass"], dtype=np.float32)
                    merged = dfd.merge(
                        dfb,
                        how="cross",
                        suffixes=("_Drums", "_Bass"),
                    )
                    merged.index = [f"{filepath}_{seg}"] * len(merged)
                    descriptor_pairs["Drum_Bass"] = pd.concat(
                        [descriptor_pairs["Drum_Bass"], merged]
                    )

    # Save the dataset annotations
    annotations_path = os.path.join(output_dir, "annotations.pkl")
    with open(annotations_path, "wb") as f:
        pickle.dump(annotations, f)
    print(f"Saved {annotations_path}")

    # Plot the number of segments by part
    n_segments_total = sum(part_counts.values())
    part_pcts = {part: count / n_segments_total for part, count in part_counts.items()}
    fig, ax = plt.subplots(figsize=(20, 8))
    plt.bar(*zip(*part_pcts.items()))
    plt.title(
        f"Distribution of parts in {dataset_name}\n{n_segments_total} segments total"
    )
    plt.ylabel("Fraction of segments")
    plt.setp(ax.get_xticklabels(), ha="right", rotation=45)
    plt.tight_layout()
    dist_plot_path = os.path.join(output_dir, "parts_distribution.png")
    plt.savefig(dist_plot_path)
    print(f"Saved {dist_plot_path}")

    if compute_descriptors:
        # Save descriptors for individual parts
        desc_dir = os.path.join(output_dir, "descriptors")
        if not os.path.isdir(desc_dir):
            os.makedirs(desc_dir)
        for part in descriptors:
            if len(descriptors[part]) > 0:
                part_descs_path = os.path.join(desc_dir, f"{part}.csv")
                descriptors[part].to_csv(part_descs_path, index=False)
                print(f"Saved {part_descs_path}")

        # Save descriptors for pairs of parts
        exp = descriptor_pairs["Drum_Bass"]

        # Construct train/test splits keeping all segments from one file together
        from sklearn.model_selection import GroupShuffleSplit

        splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        split = splitter.split(exp, groups=exp.index)
        train_inds, test_inds = next(split)
        train = exp.iloc[train_inds]
        test = exp.iloc[test_inds]

        desc_pairs_dir = os.path.join(output_dir, "descriptors_pairs")
        if not os.path.isdir(desc_pairs_dir):
            os.makedirs(desc_pairs_dir)
        train_path = os.path.join(desc_pairs_dir, "Drums_Bass_train.csv")
        train.to_csv(train_path, index=False)
        print(f"Saved {train_path}")
        print(f"  Train size: {len(train)} segments from {train.index.nunique()} files")

        test_path = os.path.join(desc_pairs_dir, "Drums_Bass_test.csv")
        test.to_csv(test_path, index=False)
        print(f"Saved {test_path}")
        print(f"  Test size: {len(test)} segments from {test.index.nunique()} files")

    n_failed = len(failed_paths)
    if n_failed > 0:
        failed_paths_str = "\n".join(failed_paths)
        print(
            f"Successfully processed {len(filepaths) - n_failed} file(s); Failed to process {n_failed}"
        )

    if skip_npz:
        sys.exit(0)

    # Collect all segrolls for each part into individual files
    # TODO: this is memory-intensive. Read segrolls in batches and write them to many smaller files of equal size
    print("Aggregating parts")
    for part in PARTS:
        part_filepaths = annotations[part]
        if len(part_filepaths) == 0:
            continue
        print(part)

        part_segrolls = []
        for filepath in tqdm(part_filepaths):
            part_segrolls.extend(np.load(filepath)[part])

        part_dir = os.path.join(output_dir, "part_segrolls")
        if not os.path.isdir(part_dir):
            os.makedirs(part_dir)

        np.savez_compressed(
            os.path.join(part_dir, part), segrolls=np.array(part_segrolls)
        )
