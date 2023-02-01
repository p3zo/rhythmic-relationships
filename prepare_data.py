import argparse
import glob
import os
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pretty_midi
import pypianoroll
from tqdm import tqdm
from PIL import Image
from rhythmtoolbox import pianoroll2descriptors

INPUT_DIR = "input"
OUTPUT_DIR = "output"

global VERBOSE


def load_midi_file(filepath, resolution=24):
    """Load a midi file as a pretty_midi object"""
    # Warnings can be verbose when midi has no metadata e.g. tempo, key, time signature
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        try:
            midi = pretty_midi.PrettyMIDI(filepath, resolution=resolution)
        except Exception as e:
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


def plot_segment(roll, seg_name, track_dir):
    # Plot piano rolls as matplotlib plots
    seg_fig, seg_ax = plt.subplots(figsize=(20, 8))
    plot_roll(roll, seg_ax)

    seg_plot_dir = os.path.join(track_dir, f"segplots_{seg_size}bar_{resolution}res")
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


def create_pil_image(roll, track_dir, seg_name, target_size=None):
    """Creates greyscale images of a piano roll suitable for model input.

    Parameters
        roll, np.array
            Piano roll array of size (128, t) where t is the number of time steps..

        track_dir, str
            Path to directory in which to save the image

        seg_name, str
            Name of the segment

        target_size, tuple (optional)
            Specify target dimensions of the image. Roll will be padded with 0s on right and bottom.
    """

    # Map MIDI velocity to pixel brightness
    arr = np.array(list(map(lambda x: np.interp(x, [0, 127], [0, 255]), roll)))

    # Zero-pad below and to the right to get target resolution
    if target_size:
        target_size = (512, 512)
        pad_bot = np.zeros((arr.shape[0], target_size[1] - arr.shape[1]))
        pad_right = np.zeros((target_size[0] - arr.shape[0], target_size[1]))
        v_padded = np.hstack((arr, pad_bot))
        arr = np.vstack((v_padded, pad_right))

    # "L" mode is greyscale and requires an 8-bit pixel range of 0-255
    im = Image.fromarray(arr.T.astype(np.uint8), mode="L")

    # Save the image
    outdir = os.path.join(track_dir, f"segrolls_{arr.shape[0]}x{arr.shape[1]}")
    if not os.path.isdir(outdir):
        os.makedirs(outdir)
    outpath = os.path.join(outdir, f"{seg_name}.png")
    im.save(outpath)
    if VERBOSE:
        print(f"  Saved {outpath}")


def mk_output_dir(prefix, filepath, seg_size, resolution):
    mid_name = f"{prefix}_{os.path.splitext(os.path.basename(filepath))[0]}"
    return os.path.join(OUTPUT_DIR, mid_name, f"{seg_size}bar_{resolution}res")


def process(
    filepath,
    seg_size=2,
    resolution=4,
    prefix="",
    drum_roll=False,
    create_images=True,
    image_size=None,
    compute_descriptors=True,
    pypianoroll_plots=False,
):
    """Slice a midi file and compute rhythmic descriptors for each segment.

    Parameters

        filepath: str
            Path to the input: either a MIDI file or a directory of MIDI files.

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

        image_size, tuple
            Specify target dimensions of the image. Roll will be padded with 0s on right and bottom.

        compute_descriptors, bool
            Use rhythmtoolbox to compute rhythmic descriptors for each segment.

        pypianoroll_plots: bool
            Create pypianoroll plots.
    """

    output_dir = mk_output_dir(prefix, filepath, seg_size, resolution)

    pmid = load_midi_file(filepath, resolution=resolution)
    bar_times, bar_ixs = get_bar_start_times(pmid, resolution)

    multitrack = pypianoroll.read(filepath, resolution=resolution)

    # Plot the piano roll of the full multitrack midi input
    if pypianoroll_plots:
        plot_multitrack(multitrack, output_dir, bar_times)

    descriptor_dfs = []

    track_seg_rolls = {}
    for track_ix, track in tqdm(enumerate(multitrack.tracks)):
        track_seg_rolls[str(track_ix)] = []

        roll = track.pianoroll

        track_dir = os.path.join(output_dir, f"track{track_ix}")
        if not os.path.isdir(track_dir):
            os.makedirs(track_dir)

        seg_ticks = resolution * 4 * seg_size

        # Slice into segments of length seg_size
        seg_descriptors = []

        seg_iter = list(zip(bar_ixs[::seg_size], bar_ixs[seg_size:][::seg_size]))
        if len(seg_iter) == 0:
            seg_iter = [(0, len(roll))]

        for seg_ix, (start, end) in enumerate(seg_iter):
            seg_name = f"bar{seg_ix * seg_size}_{seg_ix * seg_size + seg_size}_subdivision{start}-{end}"

            segroll = roll[start:end]

            # Pad every 2-bar segment to the same length
            if len(segroll) < seg_ticks:
                pad_right = np.zeros((seg_ticks - segroll.shape[0], segroll.shape[1]))
                segroll = np.vstack((segroll, pad_right))

            track_seg_rolls[str(track_ix)].append(segroll)
            if drum_roll:
                roll = get_9voice_drum_roll(roll)

            # Create piano roll images using PIL
            if create_images:
                create_pil_image(segroll, track_dir, seg_name, target_size=image_size)

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

    arr_path = os.path.join(output_dir, "arr.npz")
    np.savez(arr_path, **track_seg_rolls)
    if VERBOSE:
        print(f"  Saved {arr_path}")

    if compute_descriptors:
        return pd.concat(descriptor_dfs)

    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path",
        type=str,
        required=True,
        help="Path to the input: either a MIDI file or a directory of MIDI files.",
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
        "--prefix",
        type=str,
        default="",
        help="An identifier for output filenames.",
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
        "--image_size",
        type=str,
        default="",
        help="A resolution to use for images, e.g. 512x512.",
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

    filepath = args.path
    seg_size = args.seg_size
    resolution = args.resolution
    prefix = args.prefix
    create_images = args.create_images
    image_size = args.image_size
    if image_size:
        image_size = (int(x) for x in args.image_size.split("x"))
    pypianoroll_plots = args.pypianoroll_plots
    drum_roll = args.drum_roll
    compute_descriptors = args.compute_descriptors
    VERBOSE = args.verbose

    df = pd.DataFrame()

    if os.path.isdir(filepath):
        # Process all files in the directory
        for filepath in glob.glob(os.path.join(filepath, "*.mid")):
            mid_df = process(
                filepath,
                seg_size=seg_size,
                resolution=resolution,
                prefix=prefix,
                drum_roll=drum_roll,
                create_images=create_images,
                image_size=image_size,
                compute_descriptors=compute_descriptors,
                pypianoroll_plots=pypianoroll_plots,
            )
            df = pd.concat([df, mid_df])
    else:
        # Process a single file
        df = process(
            filepath,
            seg_size=seg_size,
            resolution=resolution,
            prefix=prefix,
            drum_roll=drum_roll,
            create_images=create_images,
            image_size=image_size,
            compute_descriptors=compute_descriptors,
            pypianoroll_plots=pypianoroll_plots,
        )

    if compute_descriptors:
        output_dir = mk_output_dir(prefix, filepath, seg_size, resolution)
        descriptors_path = os.path.join(output_dir, f"descriptors.csv")
        df.to_csv(descriptors_path, index=False)
        if VERBOSE:
            print(f"  Descriptors written to {descriptors_path}")
