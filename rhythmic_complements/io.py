import warnings

import numpy as np
import pretty_midi as pm
from PIL import Image

# TODO: fix catch_warnings block in load_midi_file and remove this
warnings.filterwarnings("ignore", category=RuntimeWarning)


def load_midi_file(filepath, resolution=24, verbose=True):
    """Load a midi file as a pretty_midi object"""
    # Warnings can be verbose when midi has no metadata e.g. tempo, key, time signature
    with warnings.catch_warnings():
        # TODO: why doesn't filterwarnings work inside of catch_warnings?
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        try:
            midi = pm.PrettyMIDI(filepath, resolution=resolution)
        except Exception as e:
            if verbose:
                print(f"Failed loading file {filepath}: {e}")
            return

    return midi


def write_midi_from_roll(roll, outpath, resolution=24, binary=False):
    note_duration = 0.02  # a reasonable bpm close to 120 (?)

    if binary:
        # Assign the onsets a reasonable MIDI velocity
        roll[roll.nonzero()] = 80

    instrument = pm.Instrument(program=0, is_drum=False)

    for voice in range(len(roll)):
        events = roll[voice]
        for event_ix, vel in enumerate(events):
            start = event_ix * note_duration
            note = pm.Note(
                velocity=vel, pitch=voice, start=start, end=start + note_duration
            )
            instrument.notes.append(note)

    track = pm.PrettyMIDI(resolution=resolution)
    track.instruments.append(instrument)
    track.write(outpath)


def write_image_from_roll(roll, outpath, im_size=None, binary=False, verbose=True):
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
    if verbose:
        print(f"  Saved {outpath}")


def write_midi_from_pattern(pattern, outpath):
    note_duration = 0.02  # a reasonable bpm close to 120 (?)

    instrument = pm.Instrument(program=0, is_drum=False)

    for event_ix, vel in enumerate(pattern):
        start = event_ix * note_duration
        note = pm.Note(velocity=127, pitch=36, start=start, end=start + note_duration)
        instrument.notes.append(note)

    track = pm.PrettyMIDI(resolution=4)
    track.instruments.append(instrument)
    track.write(outpath)
    print(f"Saved {outpath}")
