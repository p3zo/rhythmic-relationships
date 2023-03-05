import numpy as np

REPRESENTATIONS = ["roll", "chroma", "pattern", "hits", "descriptors"]

# Standard 88-key piano range
MIDI_PITCH_RANGE = [21, 108]


def get_pattern_from_roll(roll, resolution, seg_size, binarized=False):
    """__DEPRECATED__ A `pattern` is a ternary vector of onsets and offsets. `0` is a silence, `1` is an onset, and `2`
    is a continuation of a previous onset.

    IMPORTANT: in this implementation, adjacent nonzero values of the same pitch will be considered a single note with
    their mean as its velocity. Use the patterns from `parse_representations` instead, as they preserve offsets
    properly.

    Adapted from https://salu133445.github.io/pypianoroll/_modules/pypianoroll/outputs.html#to_pretty_midi
    """

    if not binarized:
        roll = roll > 0

    padded = np.pad(roll, ((1, 1), (0, 0)), "constant")
    diff = np.diff(padded.astype(np.int8), axis=0)

    onsets = np.nonzero(diff > 0)[0]
    offsets = np.nonzero(diff < 0)[0]

    pattern = np.zeros(resolution * 4 * seg_size, dtype=int)
    for onset, offset in zip(onsets, offsets):
        pattern[onset] = 1
        pattern[onset + 1 : offset] = 2

    return pattern


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


def get_ticks_from_time(times, beats, subdivisions, resolution):
    # TODO: are these really not quantized? Understand what these ratios do
    beat_ixs = np.searchsorted(beats, times) - 1
    remained = times - beats[beat_ixs]
    ratios = remained / (beats[beat_ixs + 1] - beats[beat_ixs])
    ticks = np.round((beat_ixs + ratios) * resolution).astype(int)

    quantized = [np.argmin(np.abs(t - subdivisions)) for t in times]

    return ticks, quantized


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


def remap_velocities(arr):
    """Convert MIDI velocities to real numbers in [0, 1]"""
    return np.array(
        list(
            map(
                lambda x: np.interp(x, [0, 127], [0, 1]),
                arr,
            )
        ),
        dtype=np.float32,
    )


def parse_representations(pmid, resolution, quantize=True, binarize=True):
    """Parse all representations from a PrettyMIDI object.

    See the Dataset section of this repo's README for a list of representations.

    Adapted from https://github.com/salu133445/pypianoroll/blob/18a68d4a7e39673a739396d409a2ff99af06d643/pypianoroll/inputs.py#L103-L338
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

    n_time_steps = resolution * len(beats)

    tracks = []
    for instrument in pmid.instruments:
        if len(instrument.notes) == 0:
            continue

        roll = np.zeros((n_time_steps, 128), np.uint8)
        chroma = np.zeros((n_time_steps, 12), np.uint8)
        pattern = np.zeros(n_time_steps, np.uint8)
        hits = np.zeros(n_time_steps, np.uint8)

        onset_times = np.array([note.start for note in instrument.notes])

        onsets, onsets_quantized = get_ticks_from_time(
            onset_times, beats_plus_one, subdivisions, resolution
        )

        velocities = [note.velocity for note in instrument.notes]
        pitches = [note.pitch for note in instrument.notes]

        hits[onsets] = velocities

        if instrument.is_drum:
            roll[onsets, pitches] = velocities
            chroma[onsets, [p % 12 for p in pitches]] = 1
            pattern[onsets] = 1
        else:
            offset_times = np.array([note.end for note in instrument.notes])
            offsets, offsets_quantized = get_ticks_from_time(
                offset_times, beats_plus_one, subdivisions, resolution
            )

            note_iter = (
                zip(onsets_quantized, offsets_quantized)
                if quantize
                else zip(onsets, offsets)
            )

            for ix, (start, end) in enumerate(note_iter):
                # If start and end were quantized to the same tick, move end back by 1 tick
                if quantize and end - start <= 0:
                    end += 1

                pitch = pitches[ix]
                velocity = velocities[ix]

                if velocity <= 0:
                    continue
                if velocity > 127:
                    velocity = 127

                if 0 < start < n_time_steps:
                    if roll[start - 1, pitch]:
                        roll[start - 1, pitch] = 0

                if end < n_time_steps - 1:
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

        tracks.append(
            {
                "name": instrument.name,
                "program": instrument.program,
                "is_drum": instrument.is_drum,
                "roll": remap_velocities(roll),
                "hits": remap_velocities(hits),
                "pattern": pattern,
                "chroma": chroma,
                # "drum_roll": get_9voice_drum_roll(roll),
            }
        )

    return tracks, bar_start_ticks
