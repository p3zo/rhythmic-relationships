import warnings

import numpy as np
import pretty_midi as pm
from PIL import Image
from rhythmic_complements import logger

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


def piano_roll_to_pretty_midi(roll, fs=8, program=0):
    """Convert a piano roll to a PrettyMidi object with a single instrument.

    From https://github.com/craffel/pretty-midi/blob/main/examples/reverse_pianoroll.py

    Parameters
        roll : np.ndarray, shape=(voices ,frames), dtype=int
            Piano roll of one instrument
        fs : int
            Sampling frequency of the columns, i.e. each column is spaced apart by 1./fs seconds.
        program : int
            The program number of the instrument.
    """
    notes, frames = roll.shape
    pmid = pm.PrettyMIDI()
    instrument = pm.Instrument(program=program)

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


def write_midi_from_roll(roll, outpath, resolution=4):
    pmid = piano_roll_to_pretty_midi(roll.T, fs=resolution * 2)
    pmid.write(outpath)
    logger.debug(f"Saved {outpath}")


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
