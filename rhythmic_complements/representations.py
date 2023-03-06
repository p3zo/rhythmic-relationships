from collections import defaultdict

import numpy as np

from rhythmic_complements.parts import get_part_from_program
from rhythmtoolbox import pianoroll2descriptors

REPRESENTATIONS = ["roll", "chroma", "pattern", "hits", "descriptors"]

# Standard 88-key piano range
MIDI_PITCH_RANGE = [21, 108]


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


def get_multitrack_roll(tracks, drums=False):
    """Aggregate the piano rolls for all tracks into one.

    If `drums`, only include drum parts; else only include non-drum parts.

    Adapted from https://github.com/craffel/pretty-midi/blob/91f58bedb2cc4a5022f837809cf642afd6577e9b/pretty_midi/pretty_midi.py#LL785-L824C26
    """
    # If there are no instruments, return an empty array
    if len(tracks) == 0:
        return np.zeros((128, 0))

    # Get piano rolls for each instrument
    rolls = [i["roll"].T for i in tracks if not i["is_drum"]]
    if drums:
        rolls = [i["roll"].T for i in tracks if i["is_drum"]]

    # Number of columns is max number of columns in all piano rolls
    multitrack_roll = np.zeros((128, np.max([p.shape[1] for p in rolls])))

    # Aggregate the rolls into one, keeping the max velocity of duplicate notes
    for roll in rolls:
        multitrack_roll[:, : roll.shape[1]] = np.maximum(
            multitrack_roll[:, : roll.shape[1]], roll
        )

    return multitrack_roll.T


def parse_representations(pmid, resolution, binarize=True):
    """Parse all representations from a PrettyMIDI object.

    Adapted from https://github.com/salu133445/pypianoroll/blob/18a68d4a7e39673a739396d409a2ff99af06d643/pypianoroll/inputs.py#L103-L338
    and https://github.com/ruiguo-bio/midi-miner/blob/794dac3bdf95cc17ffb6b67ff254d9c56cd479f5/tension_calculation.py#L687-L718
    """

    beats = pmid.get_beats()
    one_more_beat = 2 * beats[-1] - beats[-2]
    beats_plus_one = np.append(beats, one_more_beat)

    # Upsample beat times to the input resolution using linear interpolation
    subdivisions = []
    for start, end in zip(beats_plus_one, beats_plus_one[1:]):
        for j in range(resolution):
            subdivisions.append((end - start) / resolution * j + start)
    subdivisions.append(beats_plus_one[-1])
    subdivisions = np.array(subdivisions)

    bar_start_ticks = []
    for bar_start in pmid.get_downbeats():
        bar_start_ticks.append(np.argmin(np.abs(bar_start - subdivisions)))

    n_ticks = len(subdivisions)

    tracks = []
    for instrument in pmid.instruments:
        if len(instrument.notes) == 0:
            continue

        roll = np.zeros((n_ticks, 128), np.uint8)
        chroma = np.zeros((n_ticks, 12), np.uint8)
        pattern = np.zeros(n_ticks, np.uint8)
        hits = np.zeros(n_ticks, np.uint8)

        onset_times = [note.start for note in instrument.notes]
        onsets = [np.argmin(np.abs(t - subdivisions)) for t in onset_times]
        velocities = [note.velocity for note in instrument.notes]
        pitches = [note.pitch for note in instrument.notes]

        hits[onsets] = velocities

        if instrument.is_drum:
            # Drum representations don't need to preserve offsets
            roll[onsets, pitches] = velocities
            chroma[onsets, [p % 12 for p in pitches]] = 1
            pattern[onsets] = 1
        else:
            offset_times = [note.end for note in instrument.notes]
            offsets = [np.argmin(np.abs(t - subdivisions)) for t in offset_times]

            for ix, (start, end) in enumerate(zip(onsets, offsets)):
                pitch = pitches[ix]
                velocity = velocities[ix]

                if velocity <= 0:
                    continue

                if velocity > 127:
                    velocity = 127

                # If start and end were quantized to the same tick, move end back by 1 tick
                if end - start <= 0:
                    end += 1

                if 0 < start < n_ticks:
                    if roll[start - 1, pitch]:
                        roll[start - 1, pitch] = 0

                if end < n_ticks - 1:
                    if roll[end, pitch]:
                        end -= 1

                # In the case of duplicate notes, preserve the one with the highest velocity
                roll[start:end, pitch] = np.maximum(roll[start:end, pitch], velocity)

                # Onsets take precedence over continuations
                continuation_ixs = np.where(pattern[start:end] == 0)[0] + start
                pattern[continuation_ixs] = 2
                pattern[start] = 1

                pc = pitch % 12
                chroma[start:end, pc] = 2
                chroma[start, pc] = 1

        # Trim the top and bottom pitches of the roll
        # TODO: uncomment when io code can handle non-128-voiced rolls without transposing
        # roll = roll[:, MIDI_PITCH_RANGE[0] : MIDI_PITCH_RANGE[1] + 1]

        if binarize:
            roll = (roll > 0).astype(np.uint8)

        # Convert MIDI velocities to real numbers in [0, 1]
        roll = roll / 127.0
        hits = hits / 127.0

        tracks.append(
            {
                "name": instrument.name,
                "program": instrument.program,
                "is_drum": instrument.is_drum,
                "roll": roll,
                "hits": hits,
                "pattern": pattern,
                "chroma": chroma,
                # "drum_roll": get_9voice_drum_roll(roll),
            }
        )

    return tracks, bar_start_ticks


def roll_has_activity(roll, min_pitches, min_beats):
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


def slice_midi(
    pmid,
    seg_size=1,
    resolution=4,
    n_beat_bars=4,
    binarize=False,
    min_seg_pitches=1,
    min_seg_beats=1,
):
    """Slice a midi file and compute several representations for each segment.

    Parameters

        pmid: pretty_midi.PrettyMIDI
            A PrettyMIDI object

        seg_size: int
            Number of bars per segment.

        resolution: int
            Number of subdivisions per beat of the output representations.

        n_beat_bars: int
            Process only segments with this number of beats per bar.

        min_seg_pitches: int
            Process only segments with at least this number of pitches.

        min_seg_beats: int
            Process only segments with at least this number of beats.

    Returns

        seg_part_reprs, defaultdict(list)
            A dictionary with all representations for each segment-part pair.
    """

    tracks, bar_start_ticks = parse_representations(pmid, resolution, binarize)

    seg_iter = list(zip(bar_start_ticks, bar_start_ticks[seg_size:]))
    if len(bar_start_ticks) <= seg_size:
        # There is only one segment in the track
        seg_iter = [(0, resolution * n_beat_bars * seg_size)]

    # Initialize output objects
    n_seg_ticks = resolution * n_beat_bars * seg_size
    seg_part_reprs = defaultdict(list)

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

            # Skip segments with little activity
            if not roll_has_activity(seg_chroma, min_seg_pitches, min_seg_beats):
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

            # Join all representations in a single object array
            # IMPORTANT: these should be in the same order as `rhythmic_complements.representations.REPRESENTATIONS`
            seg_part_reprs[f"{seg_ix}_{part}"].append(
                np.array(
                    [seg_roll, seg_chroma, seg_pattern, seg_hits, seg_descriptors],
                    dtype="object",
                )
            )

    return seg_part_reprs
