import wave
from base64 import b64encode
from io import BytesIO

import numpy as np
from chord_progressions.chord import get_note_from_midi_num
from chord_progressions.io.audio import mk_chord_buffer, mk_sin
from chord_progressions.solver import select_voicing
from chord_progressions.type_templates import TYPE_TEMPLATES

SAMPLE_RATE = 44100


def mk_wav(arr):
    """Transform a numpy array to a PCM bytestring
    Adapted from https://github.com/ipython/ipython/blob/main/IPython/lib/display.py#L146
    """
    scaled = arr * 32767
    scaled = scaled.astype("<h").tobytes()

    fp = BytesIO()
    waveobj = wave.open(fp, mode="wb")
    waveobj.setnchannels(1)
    waveobj.setframerate(SAMPLE_RATE)
    waveobj.setsampwidth(2)
    waveobj.setcomptype("NONE", "NONE")
    waveobj.writeframes(scaled)
    val = fp.getvalue()
    waveobj.close()

    return val


def get_audio_el(audio):
    wav = mk_wav(audio)
    b64 = b64encode(wav).decode("ascii")
    return f'<audio controls="controls"><source src="data:audio/wav;base64,{b64}" type="audio/wav"/></audio>'


def mk_voiced_chroma_buffer(voiced_hits, duration, n_overtones):
    bufs = []

    for hit in voiced_hits:
        pos_dur = duration / len(voiced_hits)

        buf = mk_sin(0, pos_dur, 0).squeeze()
        if hit:
            chord = [get_note_from_midi_num(m) for m in hit]
            buf = mk_chord_buffer(chord, pos_dur, n_overtones)
        bufs.append(buf)

    return np.concatenate(bufs, axis=0).reshape(-1, 1)


def get_voiced_hits_from_chroma(chroma):
    voiced = []
    for c in chroma:
        voiced.append(select_voicing(c.tolist(), note_range_low=60, note_range_high=72))
    return voiced


def get_chroma_vocab():
    vocab_types = [
        "minor-seventh chord",
        "major-seventh chord",
        "perfect-fourth major tetrachord",
        "quartal tetramirror",
        "incomplete dominant-seventh chord 2",
        "half-diminished seventh chord",
        "whole-tone",
        "incomplete minor-seventh chord",
        "major-second major tetrachord",
        "dominant-seventh / german-sixth chord",
        "quartal trichord",
        "minor third",
        "major third",
        "perfect fourth",
        "minor chord",
        "major chord",
        "unison",
        "silence",
    ]

    # Add silence
    silence_template = "000000000000"
    TYPE_TEMPLATES.update({"silence": silence_template})

    # Filter type templates based on vocab_types
    vocab = {}
    for k, v in TYPE_TEMPLATES.items():
        if k in vocab_types:
            vocab[k] = v

    # Add an out-of-vocab token and map it to silence
    vocab["oov"] = silence_template

    return vocab
