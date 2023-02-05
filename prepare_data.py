import argparse
import glob
import os
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pretty_midi
import pypianoroll
from PIL import Image
from rhythmtoolbox import pianoroll2descriptors
from tqdm import tqdm

# TODO: fix catch_warnings block in load_midi_file and remove this
warnings.filterwarnings("ignore", category=RuntimeWarning)

INPUT_DIR = "input"
OUTPUT_DIR = "output"

N_MIDI_VOICES = 128

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

# In the General MIDI spec, drums are on a separate MIDI channel. We append it here for simplicity.
PARTS = list(PROGRAM_CATEGORIES.values())
PARTS.append("Drums")

global VERBOSE


def get_part_from_program(program):
    if program < 0 or program > 127:
        raise ValueError(
            f"Program number {program} is not in the valid range of [0, 127]"
        )
    return PROGRAM_CATEGORIES[[p for p in PROGRAM_CATEGORIES if p <= program + 1][-1]]


def load_midi_file(filepath, resolution=24):
    """Load a midi file as a pretty_midi object"""
    # Warnings can be verbose when midi has no metadata e.g. tempo, key, time signature
    with warnings.catch_warnings():
        # TODO: why doesn't filterwarnings work inside of catch_warnings?
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        try:
            midi = pretty_midi.PrettyMIDI(filepath, resolution=resolution)
        except Exception as e:
            if VERBOSE:
                print(f"Failed loading file {filepath}: {e}")
            return

    return midi


def get_bar_start_times(pmid, beat_division=24):
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


def create_pil_image(roll, outdir, seg_name, im_size=None):
    """Creates greyscale images of a piano roll suitable for model input.

    Parameters
        roll, np.array
            Piano roll array of size (v, t) where v is the number of voices and t is the number of time steps.

        outdir, str
            Path to directory in which to save the image

        seg_name, str
            Name of the segment

        im_size, tuple (optional)
            Specify target dimensions of the image. Roll will be padded with 0s on right and bottom.
    """

    # Map MIDI velocity to pixel brightness
    arr = np.array(list(map(lambda x: np.interp(x, [0, 127], [0, 255]), roll)))

    # Zero-pad below and to the right to get target resolution
    if im_size:
        im_size = (512, 512)
        pad_bot = np.zeros((arr.shape[0], im_size[1] - arr.shape[1]))
        pad_right = np.zeros((im_size[0] - arr.shape[0], im_size[1]))
        v_padded = np.hstack((arr, pad_bot))
        arr = np.vstack((v_padded, pad_right))

    # "L" mode is greyscale and requires an 8-bit pixel range of 0-255
    im = Image.fromarray(arr.T.astype(np.uint8), mode="L")

    # Save the image
    imdir = os.path.join(outdir, f"segrolls_{arr.shape[0]}x{arr.shape[1]}")
    if not os.path.isdir(imdir):
        os.makedirs(imdir)
    outpath = os.path.join(imdir, f"{seg_name}.png")
    im.save(outpath)
    if VERBOSE:
        print(f"  Saved {outpath}")


def mk_mid_name(prefix, filepath):
    return f"{prefix}_{os.path.splitext(os.path.basename(filepath))[0]}"


def process(
    filepath,
    output_dir,
    seg_size=2,
    resolution=4,
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

    pmid = load_midi_file(filepath, resolution=resolution)
    if not pmid:
        return None, None

    bar_times, bar_ixs = get_bar_start_times(pmid, resolution)

    # There seems to be an off-by-one bug in pypianoroll. Skip the files it can't parse.
    try:
        multitrack = pypianoroll.from_pretty_midi(pmid, resolution=resolution)
    except:
        return None, None

    # Plot the piano roll of the full multitrack midi input
    if pypianoroll_plots:
        plot_multitrack(multitrack, mid_outdir, bar_times)

    # Define an iterable to segment each track's piano roll equally
    seg_iter = list(zip(bar_ixs[::seg_size], bar_ixs[seg_size:][::seg_size]))
    if len(seg_iter) == 0:
        # TODO: this assumes all tracks are the same length. Is that always true?
        seg_iter = [(0, len(multitrack[0]))]

    # Initialize output objects
    n_seg_ticks = resolution * 4 * seg_size
    n_voices = 96
    n_voices_trim = (N_MIDI_VOICES - n_voices) // 2
    part_segrolls = {p: [] for p in PARTS}
    descriptor_dfs = []

    for track_ix, track in enumerate(multitrack.tracks):
        track_dir = os.path.join(mid_outdir, f"track{track_ix}")

        # Trim top & bottom of the roll to keep only to middle n_voices voices
        roll = track.pianoroll[:, n_voices_trim : N_MIDI_VOICES - n_voices_trim]

        part = get_part_from_program(track.program)
        if track.is_drum:
            part = "Drums"

        # Slice the piano roll into segments of equal length
        seg_descriptors = []

        for seg_ix, (start, end) in enumerate(seg_iter):
            seg_name = f"bar{seg_ix * seg_size}_{seg_ix * seg_size + seg_size}_subdivision{start}-{end}"

            segroll = roll[start:end]

            # Skip empty segments
            if not segroll.any():
                continue

            # Pad/truncate every 2-bar segment to the same length
            # TODO: this makes everything 4/4. Is that acceptable?
            if len(segroll) < n_seg_ticks:
                pad_right = np.zeros((n_seg_ticks - segroll.shape[0], segroll.shape[1]))
                segroll = np.vstack((segroll, pad_right)).astype(np.uint8)
            elif len(segroll) > n_seg_ticks:
                segroll = segroll[:n_seg_ticks]

            if drum_roll:
                segroll = get_9voice_drum_roll(segroll)

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
                create_pil_image(segroll, track_dir, seg_name, im_size=im_size)

            if pypianoroll_plots:
                plot_segment(segroll, seg_name, track_dir)

            # Compute rhythmic descriptors for the segment
            if compute_descriptors:
                seg_descriptors.append(pianoroll2descriptors(segroll))

        if compute_descriptors:
            df = pd.DataFrame(seg_descriptors)
            df.index.name = "segment_id"
            df.reset_index(inplace=True)
            df.insert(0, "track_ix", track_ix)
            descriptor_dfs.append(df)

    descriptors = pd.concat(descriptor_dfs) if compute_descriptors else None

    return part_segrolls, descriptors


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
    VERBOSE = args.verbose

    filepaths = [path]
    if os.path.isdir(path):
        filepaths = glob.glob(os.path.join(path, "**/*.mid"), recursive=True)
    print(f"Processing {len(filepaths)} midi file(s)")

    dataset = {p: [] for p in PARTS}
    descriptors = pd.DataFrame()

    output_dir = os.path.join(OUTPUT_DIR, f"{prefix}_{seg_size}bar_{resolution}res")
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    failed_paths = []
    for file_ix, filepath in enumerate(tqdm(filepaths)):
        part_segrolls, mid_df = process(
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

        for p in part_segrolls:
            dataset[p].extend(part_segrolls[p])

        if compute_descriptors:
            descriptors = pd.concat([descriptors, mid_df])

    # Convert the part segroll lists into arrays
    for p in dataset:
        dataset[p] = np.array(dataset[p])

    # Save the dataset
    dataset_path = os.path.join(output_dir, f"part_segrolls.npz")
    print("Compressing & saving dataset...")
    np.savez_compressed(dataset_path, **dataset)
    print(f"Saved ./{dataset_path}")

    # Plot the number of segments by part
    part_counts = {k: len(v) for k, v in dataset.items()}
    fig, ax = plt.subplots(figsize=(20, 8))
    plt.bar(*zip(*part_counts.items()))
    plt.setp(ax.get_xticklabels(), ha="right", rotation=45)
    plt.tight_layout()
    dist_plot_path = os.path.join(output_dir, "dataset_parts_distribution.png")
    plt.savefig(dist_plot_path)
    print(f"Saved {dist_plot_path}")

    if compute_descriptors:
        descriptors_path = os.path.join(output_dir, "descriptors.csv")
        descriptors.to_csv(descriptors_path, index=False)
        if VERBOSE:
            print(f"Descriptors written to {descriptors_path}")

    n_failed = len(failed_paths)
    if n_failed > 0:
        if VERBOSE:
            failed_paths_str = "\n".join(failed_paths)
            print(f"Failed {n_failed} file(s):\n{failed_paths_str}")
