import warnings

import numpy as np
import pretty_midi as pm
from PIL import Image
from rhythmic_relationships import logger
from rhythmic_relationships.parts import (
    get_part_from_program,
    get_program_from_part,
    PARTS,
)

# TODO: fix catch_warnings block in load_midi_file and remove this
warnings.filterwarnings("ignore", category=RuntimeWarning)


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


def parse_bar_start_ticks(pmid, resolution):
    """Parse the bar start times from a PrettyMIDI object.

    Adapted from https://github.com/ruiguo-bio/midi-miner/blob/794dac3bdf95cc17ffb6b67ff254d9c56cd479f5/tension_calculation.py#L687-L718
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

    return bar_start_ticks, subdivisions


def get_pmid_segment(
    pmid, segment_num, seg_size=1, resolution=4, n_beat_bars=4, parts=[]
):
    """Get a segment of a midi file as a PrettyMIDI object.

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
    bar_start_ticks, subdivisions = parse_bar_start_ticks(pmid, resolution)

    seg_iter = list(zip(bar_start_ticks, bar_start_ticks[seg_size:]))
    if len(bar_start_ticks) <= seg_size:
        # There is only one segment in the track
        seg_iter = [(0, resolution * n_beat_bars * seg_size)]

    slice_start, slice_end = seg_iter[segment_num]

    # Sort instruments by part index for convenience
    pmid.instruments.sort(key=lambda x: PARTS.index(get_part_from_program(x.program)))

    pmid_slice = pm.PrettyMIDI()

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

        instrument_slice = pm.Instrument(
            program=program, is_drum=instrument.is_drum, name=instrument.name
        )

        for note in instrument.notes:
            if (
                note.start >= subdivisions[slice_start]
                and note.end <= subdivisions[slice_end]
            ):
                note.start = note.start - subdivisions[slice_start]
                note.end = note.end - subdivisions[slice_start]
                instrument_slice.notes.append(note)

        pmid_slice.instruments.append(instrument_slice)

    if len(pmid_slice.instruments) == 0:
        logger.warning("No instruments found in segment.")

    return pmid_slice


def get_pretty_midi_from_roll(roll, resolution, binary=False, program=0, is_drum=False):
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
            Indicates if the piano roll has been binarized or not
    """
    # TODO: remove this transpose once the rest of the lib is refactored to use (voices, frames) rolls
    roll = roll.T

    fs = resolution * 2

    if binary:
        roll[roll.nonzero()] = 100

    notes, frames = roll.shape
    pmid = pm.PrettyMIDI()
    instrument = pm.Instrument(program=program, is_drum=is_drum)

    # Pad 1 column of zeros to accommodate the starting and ending events
    roll = np.pad(roll, [(0, 0), (1, 1)], "constant")

    # use changes in velocities to find note on / note off events
    velocity_changes = np.nonzero(np.diff(roll).T)

    # keep track on velocities and note on times
    prev_velocities = np.zeros(notes, dtype=int)
    note_on_time = np.zeros(notes)

    for time, note in zip(*velocity_changes):
        # use time + 1 because of padding above
        velocity = roll[note, time + 1]
        time = time / fs
        if velocity > 0:
            if prev_velocities[note] == 0:
                note_on_time[note] = time
                prev_velocities[note] = velocity
        else:
            pm_note = pm.Note(
                velocity=prev_velocities[note],
                pitch=note,
                start=note_on_time[note],
                end=time,
            )
            instrument.notes.append(pm_note)
            prev_velocities[note] = 0
    pmid.instruments.append(instrument)

    return pmid


def get_pretty_midi_from_roll_list(roll_list, resolution=4, binary=False, parts=[]):
    """Combines a list of piano rolls into a single PrettyMidi object.

    Param: parts, list[str]
        A list of parts corresponding to the rolls, in the same order.
    """
    pmid = pm.PrettyMIDI()
    for ix, roll in enumerate(roll_list):
        program = 0
        is_drum = False

        if parts:
            program = get_program_from_part(parts[ix])
            if parts[ix] == "Drums":
                is_drum = True

        roll_pm = get_pretty_midi_from_roll(
            roll,
            resolution=resolution,
            binary=binary,
            program=program,
            is_drum=is_drum,
        )

        pmid.instruments.append(roll_pm.instruments[0])

    return pmid


def write_midi_from_roll(roll, outpath, resolution=4, binary=False, part=None):
    """Writes a single piano roll to a MIDI file"""
    program = 0
    is_drum = False
    if part:
        program = get_program_from_part(part)
        if part == "Drums":
            is_drum = True

    pmid = get_pretty_midi_from_roll(
        roll, resolution=resolution, binary=binary, program=program, is_drum=is_drum
    )
    pmid.write(outpath)
    logger.debug(f"Saved {outpath}")


def write_midi_from_roll_list(roll_list, outpath, resolution=4, binary=False, parts=[]):
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


def write_midi_from_hits(hits, outpath, pitch=36):
    # TODO: explain choice of note_duration
    note_duration = 0.25

    instrument = pm.Instrument(program=0, is_drum=False)

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


def write_image_from_hits(hits, outpath):
    # Map MIDI velocity to pixel brightness
    arr = np.array(list(map(lambda x: 255 if x else x, hits)))
    arr = arr.reshape((1, arr.shape[0]))

    # "L" mode is greyscale and requires an 8-bit pixel range of 0-255
    im = Image.fromarray(arr.astype(np.uint8), mode="L")

    im.save(outpath)
    logger.debug(f"Saved {outpath}")


def write_image_from_pattern(pattern, outpath):
    binary_pattern = (pattern > 0).astype(int)

    # Map MIDI velocity to pixel brightness
    arr = np.array(list(map(lambda x: 255 if x else x, binary_pattern)))
    arr = arr.reshape((1, arr.shape[0]))

    # "L" mode is greyscale and requires an 8-bit pixel range of 0-255
    im = Image.fromarray(arr.astype(np.uint8), mode="L")

    im.save(outpath)
    logger.debug(f"Saved {outpath}")


def tick_to_time(tick, resolution, tempo=100):
    """Convert absolute time in ticks to seconds.
    Adapted from https://github.com/mido/mido/blob/main/mido/midifiles/units.py
    """
    scale = tempo * 0.005 / resolution
    return tick * scale


def write_midi_from_pattern(pattern, outpath, resolution=24, pitch=36):
    instrument = pm.Instrument(program=0, is_drum=False)

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
