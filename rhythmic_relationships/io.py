import warnings
from collections import defaultdict

import numpy as np
import pretty_midi as pm
from PIL import Image
from rhythmic_relationships import logger
from rhythmic_relationships.parts import (
    get_part_from_program,
    get_program_from_part,
    PARTS,
)
from rhythmic_relationships.representations import (
    get_representations,
    get_descriptors_from_roll,
    REPRESENTATIONS,
)

# TODO: fix catch_warnings block in load_midi_file and remove this
warnings.filterwarnings("ignore", category=RuntimeWarning)


def roll_has_activity(roll, min_pitches, min_beats):
    """Verify that a piano roll has at least some number of beats and pitches"""
    n_pitches = (roll.sum(axis=0) > 0).sum()
    n_beats = (roll.sum(axis=1) > 0).sum()
    return (n_pitches >= min_pitches) and (n_beats >= min_beats)


def load_midi_file(filepath, resolution=24):
    """Load a midi file as a PrettyMidi object"""
    # Warnings occur when midi has no metadata e.g. tempo, key, time signature
    with warnings.catch_warnings():
        # TODO: why doesn't filterwarnings work inside of catch_warnings?
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        try:
            midi = pm.PrettyMIDI(filepath, resolution=resolution)
        except Exception as e:
            logger.debug(f"Failed loading file {filepath}: {e}")
            return
    return midi


def get_subdivisions(pmid, resolution=4, n_beat_bars=4):
    """Parse beats from a PrettyMIDI object and create an array of subdivisions at a given resolution.

    :param pmid: PrettyMIDI object
    :param resolution: Resolution of the output array
    :param n_beat_bars: Number of beats per bar.
    :return: Array of subdivisions
    """
    beats = pmid.get_beats()

    # Assume a single bar
    if len(beats) <= 1:
        beats = np.arange(0, n_beat_bars + 1)
    else:
        additional_beat = 2 * beats[-1] - beats[-2]
        beats = np.append(beats, additional_beat)

    # Upsample beat times to the input resolution using linear interpolation
    subdivisions = []
    for start, end in zip(beats, beats[1:]):
        for j in range(resolution):
            subdivisions.append((end - start) / resolution * j + start)
    subdivisions.append(beats[-1])

    return np.array(subdivisions)


def get_bar_start_ticks(pmid, subdivisions):
    """Quantize the bar start times from PrettyMIDI object to a subdivision array."""

    bar_start_ticks = []
    for bar_start in pmid.get_downbeats():
        bar_start_ticks.append(np.argmin(np.abs(bar_start - subdivisions)))

    return np.array(bar_start_ticks)


def get_seg_iter(bar_start_ticks, seg_size, resolution, n_beat_bars):
    """Get an iterator over the start and end ticks of each segment.

    Currently, only overlapping segments are supported.
    # TODO: allow for non-overlapping segments

    :parameters:
        bar_start_ticks: np.array
            The start ticks of each bar in the midi file.
        seg_size: int
            The number of bars per segment.
        resolution: int
            The resolution of the midi file.
        n_beat_bars: int
            Process only segments with this number of beats per bar.

    :returns:
        seg_iter: list
            A list of tuples of the start and end ticks of each segment.
    """
    # Handle case when there is only one segment in the track
    if len(bar_start_ticks) <= seg_size:
        return [(0, resolution * n_beat_bars * seg_size)]

    return list(zip(bar_start_ticks, bar_start_ticks[seg_size:]))


def roll_contains_mono_melody(roll, min_n_pitches, max_n_rests):
    """Check if a piano roll contains a melody.

    A melody is defined as a monophonic sequence with at least `n_pitches`
    unique pitches and at most `max_n_rests` consecutive ticks of rests.

    :param roll: Piano roll
    :param min_n_pitches: Minimum number of unique pitches
    :param max_n_rests: Maximum number of consecutive ticks of rests.
        Note that this depends on the resolution of the roll.
    :return: Boolean
    """
    # Check that the roll is monophonic
    if not np.all(roll.sum(axis=1) <= 1):
        return False

    # Check that the roll has at least `min_n_pitches` unique pitches
    if (roll.sum(axis=0) > 0).sum() < min_n_pitches:
        return False

    # Check that the roll has at most `max_n_rests` consecutive ticks of rests
    n_rests = 0
    for i in range(roll.shape[0]):
        if roll[i].sum() == 0:
            n_rests += 1
            if n_rests > max_n_rests:
                return False
        else:
            n_rests = 0

    return True


def slice_midi(
    pmid,
    seg_size=2,
    resolution=4,
    n_beat_bars=4,
    min_seg_pitches=1,
    min_seg_beats=1,
    min_melody_pitches=2,
    max_melody_rests=4,
    representations=REPRESENTATIONS,
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

        min_melody_pitches: int
            For melodic instruments, process only segments with at least this number of pitches.

        max_melody_rests: int
            For melodic instruments, process only segments with at most this number of consecutive beats of rest.

        representations: list
            A list of representations to compute for each segment.

    Returns

        seg_part_reprs, defaultdict(list)
            A dictionary with all representations for each segment-part pair.
    """

    subdivisions = get_subdivisions(pmid, resolution, n_beat_bars)
    tracks = get_representations(pmid, subdivisions)

    bar_start_ticks = get_bar_start_ticks(pmid, subdivisions)
    seg_iter = get_seg_iter(
        bar_start_ticks,
        seg_size=seg_size,
        resolution=resolution,
        n_beat_bars=n_beat_bars,
    )

    max_melody_rest_ticks = max_melody_rests * resolution

    # Initialize output objects
    n_seg_ticks = resolution * n_beat_bars * seg_size
    seg_part_reprs = defaultdict(list)

    for track in tracks:
        part = get_part_from_program(track["program"])
        if track["is_drum"]:
            part = "Drums"
        if not part:
            continue

        # Slice the piano roll into segments of equal length
        for seg_ix, (start, end) in enumerate(seg_iter):
            seg_chroma = track["chroma"][start:end]

            # Skip segments with little activity
            if not roll_has_activity(seg_chroma, min_seg_pitches, min_seg_beats):
                continue

            seg_onset_roll = track["onset_roll"][start:end]

            # Skip segments that don't have the target number of beats
            if len(seg_onset_roll) != n_seg_ticks:
                continue

            if part == "Melody" and not roll_contains_mono_melody(
                seg_onset_roll, min_melody_pitches, max_melody_rest_ticks
            ):
                continue

            # Join the representations into a single object array
            seg_reprs = []
            for representation in representations:
                if representation == "descriptors":
                    seg_reprs.append(
                        get_descriptors_from_roll(seg_onset_roll, resolution)
                    )
                    continue
                if representation not in track:
                    raise ValueError(
                        f"Invalid representation `{representation}`. Must be one of {REPRESENTATIONS}"
                    )

                seg_reprs.append(track[representation][start:end])

            seg_part_reprs[f"{seg_ix}_{part}"].append(np.array(seg_reprs, dtype=object))

    return seg_part_reprs


def get_pmid_segment_reprs(pmid, segment_id, parts):
    """Get the representations for a segment of a MIDI file.

    :param pmid: The PrettyMIDI object to load the segment from.
    :param segment_id: The ID of the segment to load.
    :param parts: The parts to load.
    :return: A tuple of (roll, onset_roll, onset_roll_3_octave, hits) for each part.
    """
    seg_part_reprs = slice_midi(pmid)

    roll_list = []
    or_list = []
    or3_list = []
    hits_list = []
    for part in parts:
        reprs = seg_part_reprs[f"{segment_id}_{part}"][0]
        # TODO: convert all reprs to pmid
        roll_list.append(reprs[REPRESENTATIONS.index("roll")])
        or_list.append(reprs[REPRESENTATIONS.index("onset roll")])
        or3_list.append(reprs[REPRESENTATIONS.index("3-octave onset roll")])
        hits_list.append(reprs[REPRESENTATIONS.index("hits")] * 127)

    pmid_roll = get_pretty_midi_from_roll_list(roll_list, parts=parts)
    pmid_or = get_pretty_midi_from_roll_list(or_list, parts=parts, onset_roll=True)
    pmid_or3 = get_pretty_midi_from_roll_list(
        or3_list, parts=parts, n_octaves=3, onset_roll=True
    )
    return (
        pmid_roll,
        pmid_or,
        pmid_or3,
        hits_list,
    )


def get_pmid_segment(
    pmid, segment_num, seg_size=1, resolution=4, n_beat_bars=4, parts=None
):
    """Get a segment of a midi file as a PrettyMIDI object.

    This is different from get_pmid_segment_reprs in that it reads and slices the midi directly instead of reading a
    representation of MIDI and then writing to MIDI, which is lossy in ways depending on the representation.

    # TODO: this is currently broken
    # TODO: write a test for this

    :parameters:
        pmid : pretty_midi.PrettyMIDI
            The PrettyMIDI object to slice
        segment_num : int
            The segment number to slice
        seg_size : int
            The number of bars per segment
        resolution : int
            The resolution of the midi file
        n_beat_bars : int
            Process only segments with this number of beats per bar.
        parts: list
            A list of parts to include in the output midi file. If empty, include all parts.

    :returns:
        pmid_slice : pretty_midi.PrettyMIDI
    """
    subdivisions = get_subdivisions(pmid, resolution, n_beat_bars)
    bar_start_ticks = get_bar_start_ticks(pmid, subdivisions)
    seg_iter = get_seg_iter(bar_start_ticks, seg_size, resolution, n_beat_bars)

    seg_start, seg_end = seg_iter[segment_num]

    # Sort instruments by part index for convenience
    pmid.instruments.sort(key=lambda x: PARTS.index(get_part_from_program(x.program)))

    pmid_segment = pm.PrettyMIDI()

    # Create a new PrettyMIDI object with only the notes in the segment
    for instrument in pmid.instruments:
        if len(instrument.notes) == 0:
            continue

        # Note: this swap may look redundant, but it keeps the program numbers consistent per part
        part = get_part_from_program(instrument.program)
        if instrument.is_drum:
            part = "Drums"
        program = get_program_from_part(part)

        if parts and part not in parts:
            continue

        new_instrument = pm.Instrument(
            program=program, is_drum=instrument.is_drum, name=instrument.name
        )

        for note in instrument.notes:
            if (
                note.start >= subdivisions[seg_start]
                and note.end <= subdivisions[seg_end]
            ):
                note.start = note.start - subdivisions[seg_start]
                note.end = note.end - subdivisions[seg_start]
                new_instrument.notes.append(note)

        if len(new_instrument.notes) > 0:
            pmid_segment.instruments.append(new_instrument)

    if len(pmid_segment.instruments) == 0:
        logger.warning("No instruments found in segment.")

    return pmid_segment


def get_pretty_midi_from_roll(
    roll,
    resolution,
    binary=False,
    program=0,
    is_drum=False,
    part="",
    n_octaves=None,
    onset_roll=False,
):
    """Convert a piano roll to a PrettyMidi object with a single instrument.

    Adapted from https://github.com/craffel/pretty-midi/blob/main/examples/reverse_pianoroll.py

    Parameters
        roll : np.ndarray, shape=(voices, frames), dtype=int
            Piano roll of one instrument
        resolution : int
            The ticks per beat resolution of the piano roll
        program : int
            The program number of the instrument.
        is_drum : bool
            Indicates if the instrument is a drum or not
        binary : bool
            Indicates if the roll is binary or not
        part : str
            The name of the part
        n_octaves : int
            The number of octaves in each roll. Centers the roll in the middle of the midi range at C4 (60).
            If none, use the entire midi range.
        onset_roll: bool
            Indicates if the piano roll is an onset roll or not

    Returns
        pmid : pretty_midi.PrettyMIDI
    """
    if binary:
        roll[roll.nonzero()] = 100
    else:
        roll = (roll * 127).astype(np.uint8)

    if n_octaves:
        # Trim the top and bottom pitches of the roll
        pad_size_bot = 60 - (n_octaves - 1) * 12 // 2
        pad_size_top = 128 - (72 + (n_octaves - 1) * 12 // 2)

        pad_bot = np.zeros((roll.shape[0], pad_size_bot), dtype=roll.dtype)
        pad_top = np.zeros((roll.shape[0], pad_size_top), dtype=roll.dtype)

        roll = np.hstack((pad_bot, roll, pad_top))

    pmid = pm.PrettyMIDI()
    instrument = pm.Instrument(program=program, is_drum=is_drum, name=part)

    fs = resolution * 2

    if onset_roll:
        for tick, pitch in zip(*roll.nonzero()):
            instrument.notes.append(
                pm.Note(
                    velocity=roll[tick][pitch],
                    pitch=pitch,
                    start=tick / fs,
                    end=(tick + 1) / fs,
                )
            )
        pmid.instruments.append(instrument)

        return pmid

    # TODO: remove this transpose once the lib consistently uses rolls of shape (voices, frames)
    roll = roll.T

    # Pad 1 column of zeros to accommodate the starting and ending events
    roll = np.pad(roll, [(0, 0), (1, 1)], "constant")

    # Use changes in velocities to infer onsets and offsets
    velocity_changes = np.nonzero(np.diff(roll).T)

    n_pitches = roll.shape[0]
    prev_velocities = np.zeros(n_pitches, dtype=int)
    onset_times = np.zeros(n_pitches)

    for tick, pitch in zip(*velocity_changes):
        # Use `time + 1` because of padding above
        velocity = roll[pitch, tick + 1]

        time = tick / fs
        if velocity > 0:
            if prev_velocities[pitch] == 0:
                onset_times[pitch] = time
                prev_velocities[pitch] = velocity
        else:
            pm_note = pm.Note(
                velocity=prev_velocities[pitch],
                pitch=pitch,
                start=onset_times[pitch],
                end=time,
            )
            instrument.notes.append(pm_note)
            prev_velocities[pitch] = 0
    pmid.instruments.append(instrument)

    return pmid


def get_pretty_midi_from_roll_list(
    roll_list, resolution=4, binary=False, parts=None, n_octaves=None, onset_roll=False
):
    """Combines a list of piano rolls into a single PrettyMidi object.

    Parameters
    ----------
    roll_list : list[np.ndarray]
        A list of piano rolls, one for each instrument.

    resolution : int
        The ticks per beat resolution of the piano roll

    binary : bool
        Indicates if the roll is binary or not

    parts : list[str]
        A list of parts corresponding to the rolls, in the same order.

    n_octaves : int
        The number of octaves in each roll. Centers the roll in the middle of the midi range at C4 (60).
        If none, use the entire midi range.

    onset_roll: bool
        Indicates if the piano roll is an onset roll or not

    Returns
    -------
    pmid : pretty_midi.PrettyMIDI
        A PrettyMidi object with one instrument per piano roll.
    """
    pmid = pm.PrettyMIDI()
    for ix, roll in enumerate(roll_list):
        program = 0
        part = ""
        is_drum = False

        if parts:
            program = get_program_from_part(parts[ix])
            part = parts[ix]
            if parts[ix] == "Drums":
                is_drum = True

        roll_pm = get_pretty_midi_from_roll(
            roll,
            resolution=resolution,
            binary=binary,
            program=program,
            is_drum=is_drum,
            part=part,
            n_octaves=n_octaves,
            onset_roll=onset_roll,
        )

        pmid.instruments.append(roll_pm.instruments[0])

    return pmid


def write_midi_from_roll(
    roll,
    outpath,
    resolution=4,
    binary=False,
    part=None,
    n_octaves=None,
    onset_roll=False,
):
    """Writes a single piano roll to a MIDI file"""
    program = 0
    is_drum = False
    if part:
        program = get_program_from_part(part)
        if part == "Drums":
            is_drum = True

    pmid = get_pretty_midi_from_roll(
        roll,
        resolution=resolution,
        binary=binary,
        program=program,
        is_drum=is_drum,
        part=part,
        n_octaves=n_octaves,
        onset_roll=onset_roll,
    )

    pmid.write(outpath)
    logger.debug(f"Saved {outpath}")


def write_midi_from_roll_list(
    roll_list, outpath, resolution=4, binary=False, parts=None
):
    """Combines a list of piano rolls into a single multi-track MIDI file"""
    pmid = get_pretty_midi_from_roll_list(
        roll_list, resolution=resolution, binary=binary, parts=parts
    )
    pmid.write(outpath)
    logger.info(f"Saved {outpath}")


def write_image_from_roll(roll, outpath, im_size=None, binary=False):
    """Create a greyscale image of a piano roll.

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
    from_range = [0, 127]
    if binary:
        from_range = [0, 1]

    arr = np.array(list(map(lambda x: np.interp(x, from_range, [0, 255]), roll)))

    # Zero-pad below and to the right to get target resolution
    if im_size:
        im_size = (512, 512)
        pad_bot = np.zeros((arr.shape[0], im_size[1] - arr.shape[1]))
        pad_right = np.zeros((im_size[0] - arr.shape[0], im_size[1]))
        v_padded = np.hstack((arr, pad_bot))
        arr = np.vstack((v_padded, pad_right))

    # "L" mode is greyscale and requires an 8-bit pixel range of 0-255
    im = Image.fromarray(arr.T.astype(np.uint8), mode="L")

    im.save(outpath)
    logger.debug(f"  Saved {outpath}")


def write_midi_from_hits(hits, outpath, pitch=36, part=""):
    # TODO: explain choice of note_duration
    note_duration = 0.25

    instrument = pm.Instrument(program=0, is_drum=False, name=part)

    for event_ix, vel in enumerate(hits):
        if vel:
            start = event_ix * note_duration
            note = pm.Note(
                velocity=127, pitch=pitch, start=start, end=start + note_duration
            )
            instrument.notes.append(note)

    track = pm.PrettyMIDI()
    track.instruments.append(instrument)
    track.write(outpath)
    logger.debug(f"Saved {outpath}")


def get_image_from_vector(hits):
    # Map MIDI velocity to pixel brightness
    arr = np.array(list(map(lambda x: 255 if x else x, hits)))
    arr = arr.reshape((1, arr.shape[0]))

    # "L" mode is greyscale and requires an 8-bit pixel range of 0-255
    return Image.fromarray(arr.astype(np.uint8), mode="L")


def write_image_from_hits(hits, outpath):
    im = get_image_from_vector(hits)
    im.save(outpath)
    logger.debug(f"Saved {outpath}")


def write_image_from_pattern(pattern, outpath):
    binary_pattern = (pattern > 0).astype(int)
    im = get_image_from_vector(binary_pattern)
    im.save(outpath)
    logger.debug(f"Saved {outpath}")


def tick_to_time(tick, resolution, tempo=100):
    """Convert absolute time in ticks to seconds.
    Adapted from https://github.com/mido/mido/blob/main/mido/midifiles/units.py
    """
    scale = tempo * 0.005 / resolution
    return tick * scale


def write_midi_from_pattern(pattern, outpath, resolution=24, pitch=36, part=""):
    instrument = pm.Instrument(program=0, is_drum=False, name=part)

    binary_pattern = pattern > 0
    padded = np.pad(binary_pattern, (1, 0), "constant")
    diff = np.diff(padded.astype(np.int8), axis=0)

    onsets = np.nonzero(diff > 0)[0]
    offsets = np.nonzero(diff < 0)[0]

    for onset, offset in zip(onsets, offsets):
        start = tick_to_time(onset, resolution=resolution)
        end = tick_to_time(offset, resolution=resolution)
        note = pm.Note(velocity=127, pitch=pitch, start=start, end=end)
        instrument.notes.append(note)

    track = pm.PrettyMIDI()
    track.instruments.append(instrument)
    track.write(outpath)
    logger.debug(f"Saved {outpath}")
