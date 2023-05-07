import numpy as np

from rhythmtoolbox import DESCRIPTOR_NAMES
from rhythmic_relationships.io import (
    get_seg_iter,
    slice_midi,
    load_midi_file,
    chroma_contains_mono_melody,
)
from rhythmic_relationships.representations import REPRESENTATIONS

N_DESCRIPTORS = len(DESCRIPTOR_NAMES)

BABYSLAKH_18_FILEPATH = "tests/midi/babyslakh/Track00018.mid"
BABYSLAKH_18_VOCALS_8_FILEPATH = "tests/midi/bslakh_t18_vocals_8.mid"

BOSKA_3_PIANO_FILEPATH = "tests/midi/boska/3_piano.mid"
BOSKA_3_DRUMS_FILEPATH = "tests/midi/boska/3_drums.mid"


def test_roll_contains_mono_melody():
    pmid = load_midi_file(BABYSLAKH_18_VOCALS_8_FILEPATH)

    assert chroma_contains_mono_melody(
        pmid.get_piano_roll().T / 127, min_n_pitches=2, max_n_rests=4
    )


def test_get_seg_iter():
    bar_start_ticks = [0, 4, 8, 12]
    seg_size = 1
    resolution = 4
    n_beat_bars = 4
    seg_iter = get_seg_iter(bar_start_ticks, seg_size, resolution, n_beat_bars)
    assert seg_iter == [(0, 4), (4, 8), (8, 12)]


def test_slice_midi():
    pmid = load_midi_file(BABYSLAKH_18_VOCALS_8_FILEPATH)
    sp_reprs = slice_midi(
        pmid,
        seg_size=1,
        resolution=4,
        n_beat_bars=4,
        min_seg_pitches=1,
        min_seg_beats=1,
    )
    assert np.array_equal(list(sp_reprs), ["0_Melody"])

    reprs = sp_reprs["0_Melody"][0]
    assert reprs.shape[0] == len(REPRESENTATIONS)
    assert reprs[REPRESENTATIONS.index("roll")].shape == (16, 128)
    assert reprs[REPRESENTATIONS.index("chroma")].shape == (16, 12)
    assert reprs[REPRESENTATIONS.index("pattern")].shape == (16,)
    assert reprs[REPRESENTATIONS.index("hits")].shape == (16,)
    assert reprs[REPRESENTATIONS.index("descriptors")].shape == (N_DESCRIPTORS,)

    pmid = load_midi_file(BOSKA_3_DRUMS_FILEPATH)
    sp_reprs = slice_midi(
        pmid,
        seg_size=1,
        resolution=4,
        n_beat_bars=4,
        min_seg_pitches=1,
        min_seg_beats=1,
    )
    assert np.array_equal(list(sp_reprs), ["0_Drums"])

    reprs = sp_reprs["0_Drums"][0]
    assert reprs.shape[0] == len(REPRESENTATIONS)
    assert reprs[REPRESENTATIONS.index("roll")].shape == (16, 128)
    assert reprs[REPRESENTATIONS.index("chroma")].shape == (16, 12)
    assert reprs[REPRESENTATIONS.index("pattern")].shape == (16,)
    assert reprs[REPRESENTATIONS.index("hits")].shape == (16,)
    assert reprs[REPRESENTATIONS.index("descriptors")].shape == (N_DESCRIPTORS,)
