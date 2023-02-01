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

VERBOSE = False


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


def plot_multitrack(multitrack, output_dir, bar_start_times, mid_name):
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

    plt.title(mid_name)

    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    plot_path = os.path.join(output_dir, "roll.png")
    plt.savefig(plot_path)
    plt.clf()
    if VERBOSE:
        print(f"  Saved {plot_path}")


def plot_segment(roll, seg_name, track_ix, track_dir, mid_name):
    # Plot piano rolls as matplotlib plots
    seg_fig, seg_ax = plt.subplots(figsize=(20, 8))
    plot_roll(roll, seg_ax)
    plt.title(f"{mid_name}\ntrack {track_ix}\n{seg_name}")

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
    """Condense a 128-voice piano roll to a 9-voice roll for drums"""

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


def create_pil_images(roll, track_dir, seg_name, drum_roll=False):
    """Creates greyscale images of a piano roll suitable for model input.

    Parameters
        roll, np.array
            Piano roll array of size (128, t) where t is the number of time steps..

        track_dir, str
            Path to directory in which to save the image

        seg_name, str
            Name of the segment

        drum_roll, bool
            Condense a 128-voice piano roll to a 9-voice roll for drums
    """

    if drum_roll:
        roll = get_9voice_drum_roll(roll)

    # Map MIDI velocity to pixel brightness
    arr = np.array(list(map(lambda x: np.interp(x, [0, 127], [0, 255]), roll)))

    # PIL greyscale requires an 8-bit pixel range of 0-255
    arr = arr.astype(np.uint8)

    # Save the image at the original resolution
    pil_dir = os.path.join(track_dir, f"segrolls_{seg_size}bar_{resolution}res")
    if not os.path.isdir(pil_dir):
        os.makedirs(pil_dir)
    pil_path = os.path.join(
        pil_dir,
        f"{seg_name}.png",
    )

    # mode "L" is greyscale
    im = Image.fromarray(arr.T, mode="L")
    im.save(pil_path)
    if VERBOSE:
        print(f"  Saved {pil_path}")

    # Add padding below and to the right to get target resolution
    target_size = (512, 512)
    im_size = im.size
    pad_bot = np.zeros((im_size[0], target_size[1] - im_size[1]))
    pad_right = np.zeros((target_size[0] - im_size[0], target_size[1]))
    v_padded = np.hstack((arr, pad_bot))
    padded = np.vstack((v_padded, pad_right))

    padded_im = Image.fromarray(padded.T.astype(np.uint8), mode="L")

    # Save the padded image
    pil_upscaled_dir = os.path.join(
        track_dir,
        f"segrolls_{seg_size}bar_{resolution}res_{target_size[0]}x{target_size[1]}",
    )
    if not os.path.isdir(pil_upscaled_dir):
        os.makedirs(pil_upscaled_dir)
    pil_upscaled_path = os.path.join(
        pil_upscaled_dir,
        f"{seg_name}.png",
    )
    padded_im.save(pil_upscaled_path)
    if VERBOSE:
        print(f"  Saved {pil_upscaled_path}")


def analyze(filepath, seg_size=1, resolution=4, prefix="", pypianoroll_plots=True):
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

    pypianoroll_plots: bool
        Create pypianoroll plots.
    """

    mid_name = f"{prefix}_{os.path.splitext(os.path.basename(filepath))[0]}"
    output_dir = os.path.join(OUTPUT_DIR, f"{mid_name}")
    print(f"{mid_name}")

    pmid = load_midi_file(filepath, resolution=resolution)
    bar_start_times, bar_start_ixs = get_bar_start_times(pmid, resolution)

    multitrack = pypianoroll.read(filepath, resolution=resolution)

    # Plot the piano roll of the full multitrack midi input
    if pypianoroll_plots:
        plot_multitrack(multitrack, output_dir, mid_name, bar_start_times)

    descriptor_dfs = []

    for track_ix, track in tqdm(enumerate(multitrack.tracks)):
        if VERBOSE:
            print(f"  Track {track_ix}")

        roll = track.pianoroll

        track_dir = os.path.join(output_dir, f"track{track_ix}")
        if not os.path.isdir(track_dir):
            os.makedirs(track_dir)

        # Slice into segments of length seg_size
        segment_descriptors = []

        seg_subdivided_iter = list(
            zip(bar_start_ixs[::seg_size], bar_start_ixs[seg_size:][::seg_size])
        )

        if len(seg_subdivided_iter) == 0:
            seg_subdivided_iter = [(0, len(roll))]

        for seg_ix, (start, end) in enumerate(seg_subdivided_iter):
            if VERBOSE:
                print(f"    seg {seg_ix}")

            seg_name = f"bar{seg_ix * seg_size}_{seg_ix * seg_size + seg_size}_subdivision{start}-{end}"

            segroll = roll[start:end]

            if pypianoroll_plots:
                plot_segment(segroll, seg_name, track_ix, track_dir, mid_name)

            # Create piano roll images using PIL
            create_pil_images(segroll, track_dir, seg_name)

            # Compute rhythmic descriptors for the segment
            segment_descriptors.append(pianoroll2descriptors(segroll))

        if VERBOSE:
            print(f"  {len(segment_descriptors)} {seg_size}-bar segments analyzed")

        df = pd.DataFrame(segment_descriptors)
        df.index.name = "segment_id"
        df.reset_index(inplace=True)

        analysis_path = os.path.join(track_dir, f"descriptors-{seg_size}-bar.csv")
        df.to_csv(analysis_path, index=False)
        if VERBOSE:
            print(f"  Descriptors written to {analysis_path}")

        descriptor_dfs.append(df)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path",
        type=str,
        help="Path to the input: either a MIDI file or a directory of MIDI files.",
    )
    parser.add_argument(
        "--seg_size", type=int, default=1, help="Number of bars per segment."
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=24,
        help="Number of subdivisions per beat. Default = 4, i.e. 16th note.",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default="",
        help="An identifier for output filenames.",
    )
    parser.add_argument(
        "--pypianoroll_plots",
        action="store_true",
        help="Create a pypianoroll plot for each segment and another for the entire track.",
    )
    args = parser.parse_args()

    if not args.path:
        # args.path = os.path.join(INPUT_DIR, "slakh00006/MIDI/S00.mid")  # type 1
        args.path = os.path.join(INPUT_DIR, "slakh00006/all_src.mid")  # type 0
        args.prefix = "slakh00006"
        # args.path = os.path.join(INPUT_DIR, "rtb/boska")  # directory of type 0s
        # args.prefix = "boska"

    filepath = args.path
    seg_size = args.seg_size
    resolution = args.resolution
    prefix = args.prefix
    pypianoroll_plots = args.pypianoroll_plots

    if os.path.isdir(filepath):
        # Analyze all files in the directory
        for filepath in glob.glob(os.path.join(filepath, "*.mid")):
            analyze(
                filepath,
                seg_size=seg_size,
                resolution=resolution,
                prefix=prefix,
                pypianoroll_plots=pypianoroll_plots,
            )
    else:
        # Analyze a single file
        analyze(
            filepath,
            seg_size=seg_size,
            resolution=resolution,
            prefix=prefix,
            pypianoroll_plots=pypianoroll_plots,
        )
