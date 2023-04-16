import numpy as np

from rhythmtoolbox import pianoroll2descriptors

REPRESENTATIONS = [
    "roll",
    "onset_roll",
    "onset_roll_3_octave",
    "binary_onset_roll",
    "drum_roll",
    "chroma",
    "pattern",
    "hits",
    "descriptors",
]


# Standard 88-key piano range
MIDI_PITCH_RANGE = [21, 108]

# Follows paper mapping in https://magenta.tensorflow.org/datasets/groove
DRUM_MAP_9_VOICE = {
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


def get_9voice_drum_roll_pitches(pitches):
    """Map a list of pitches to the pitches of a 9-voice drum roll

    :param pitches: list of MIDI pitches
    :return: list of drum roll pitches
    """
    drum_roll_voices = sorted(list(np.unique(list(DRUM_MAP_9_VOICE.values()))))

    drum_roll_pitches = []
    for p in pitches:
        drum_roll_pitch = 0
        if p in DRUM_MAP_9_VOICE:
            drum_roll_pitch = drum_roll_voices.index(DRUM_MAP_9_VOICE[p])
        drum_roll_pitches.append(drum_roll_pitch)

    return drum_roll_pitches


def get_three_octave_pitches(pitches):
    """Map a list of pitches three octaves centered around C4 with range [48, 84].

    Pitches above 84 are mapped to the top octave (MIDI range [72, 84]).
    Pitches below 48 are mapped to the bottom octave with (MIDI range [48, 60]).
    Pitches between 48 and 84 are mapped to the center octave with (MIDI range [60, 72]).
    """
    three_octave_pitches = []
    for p in pitches:
        pc = p % 12
        if p < 48:
            three_octave_pitches.append(pc)
        elif p > 84:
            three_octave_pitches.append(pc + 24)
        else:
            three_octave_pitches.append(pc + 12)

    return three_octave_pitches


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


def get_representations(pmid, subdivisions):
    """Compute all representations from a PrettyMIDI object.

    :param pmid: PrettyMIDI object
    :param subdivisions: list of tick times
    :return: dict of representations
    """

    # TODO: take a parameter to allow for only a subset of representations to be computed
    # LEFT OFF: handle the case when len(subdivisions) == 1

    n_ticks = len(subdivisions) - 1

    tracks = []
    for instrument in pmid.instruments:
        if len(instrument.notes) == 0:
            continue

        roll = np.zeros((n_ticks, 128), np.uint8)
        onset_roll = np.zeros((n_ticks, 128), np.uint8)
        onset_roll_3_octave = np.zeros((n_ticks, 36), np.uint8)
        drum_roll = np.zeros((n_ticks, 9))
        chroma = np.zeros((n_ticks, 12), np.uint8)
        pattern = np.zeros(n_ticks, np.uint8)
        hits = np.zeros(n_ticks, np.uint8)

        onsets_unquantized = [note.start for note in instrument.notes]
        onsets = [np.argmin(np.abs(t - subdivisions)) for t in onsets_unquantized]
        pitches = [note.pitch for note in instrument.notes]
        velocities = [note.velocity for note in instrument.notes]
        for ix, v in enumerate(velocities):
            if v < 0:
                velocities[ix] = 0
            if v > 127:
                velocities[ix] = 127

        # Construct representations don't need to preserve offsets
        hits[onsets] = velocities
        onset_roll[onsets, pitches] = velocities

        three_octave_pitches = get_three_octave_pitches(pitches)
        onset_roll_3_octave[onsets, three_octave_pitches] = velocities

        if instrument.is_drum:
            roll[onsets, pitches] = velocities
            chroma[onsets, [p % 12 for p in pitches]] = 1
            pattern[onsets] = 1

            drum_roll_pitches = get_9voice_drum_roll_pitches(pitches)
            drum_roll[onsets, drum_roll_pitches] = velocities

        else:
            # Construct representations that need to preserve offsets
            offsets_unquantized = [note.end for note in instrument.notes]
            offsets = [np.argmin(np.abs(t - subdivisions)) for t in offsets_unquantized]

            for note_ix, (on_tick, off_tick) in enumerate(zip(onsets, offsets)):
                pitch = pitches[note_ix]
                velocity = velocities[note_ix]

                if velocity == 0:
                    continue

                # If on_tick and off_tick were quantized to the same tick, move off_tick back by 1 tick
                if off_tick - on_tick <= 0:
                    off_tick = on_tick + 1

                # If the note ends after the last tick, move it back to the last tick
                if off_tick > n_ticks:
                    off_tick = n_ticks

                # In the case of duplicate notes, preserve the one with the highest velocity
                roll[on_tick:off_tick, pitch] = np.maximum(
                    roll[on_tick:off_tick, pitch], velocity
                )

                # Onsets take precedence over continuations
                continuation_ixs = np.where(pattern[on_tick:off_tick] == 0)[0] + on_tick
                pattern[continuation_ixs] = 2
                pattern[on_tick] = 1

                pc = pitch % 12
                chroma[on_tick:off_tick, pc] = 2
                chroma[on_tick, pc] = 1

        # Trim the top and bottom pitches of the roll
        # TODO: uncomment when io code handles non-128-voiced rolls without transposing
        # roll = roll[:, MIDI_PITCH_RANGE[0] : MIDI_PITCH_RANGE[1] + 1]

        # Convert MIDI velocities to real numbers in [0, 1]
        roll = roll / 127.0
        onset_roll = onset_roll / 127.0
        onset_roll_3_octave = onset_roll_3_octave / 127.0
        hits = hits / 127.0

        binary_onset_roll = (onset_roll > 0).astype(np.uint8)

        tracks.append(
            {
                "name": instrument.name,
                "program": instrument.program,
                "is_drum": instrument.is_drum,
                "roll": roll,
                "onset_roll": onset_roll,
                "binary_onset_roll": binary_onset_roll,
                "onset_roll_3_octave": onset_roll_3_octave,
                "drum_roll": drum_roll,
                "chroma": chroma,
                "pattern": pattern,
                "hits": hits,
            }
        )

    return tracks


def resize_roll_to_n_beats(roll, n_beats, resolution):
    """Pad/truncate a piano roll to a beat length"""
    n_ticks = n_beats * resolution
    if len(roll) < n_ticks:
        pad_right = np.zeros((n_ticks - roll.shape[0], roll.shape[1]))
        roll = np.vstack((roll, pad_right)).astype(np.uint8)
    elif len(roll) > n_ticks:
        roll = roll[:n_ticks]
    return roll


def get_descriptors_from_roll(roll, resolution):
    return np.array(
        list(pianoroll2descriptors(roll, resolution=resolution).values()),
        dtype=np.float32,
    )
