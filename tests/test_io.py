import numpy as np

from rhythmtoolbox import DESCRIPTOR_NAMES
from rhythmic_relationships.io import get_seg_iter, slice_midi, load_midi_file

N_DESCRIPTORS = len(DESCRIPTOR_NAMES)

BOSKA_3_PIANO_FILEPATH = "tests/midi/boska/3_piano.mid"
BOSKA_3_DRUMS_FILEPATH = "tests/midi/boska/3_drums.mid"


def test_get_seg_iter():
    bar_start_ticks = [0, 4, 8, 12]
    seg_size = 1
    resolution = 4
    n_beat_bars = 4
    seg_iter = get_seg_iter(bar_start_ticks, seg_size, resolution, n_beat_bars)
    assert seg_iter == [(0, 4), (4, 8), (8, 12)]


def test_slice_midi():
    pmid = load_midi_file(BOSKA_3_PIANO_FILEPATH)
    sp_reprs = slice_midi(
        pmid,
        seg_size=1,
        resolution=4,
        n_beat_bars=4,
        binarize=True,
        min_seg_pitches=1,
        min_seg_beats=1,
    )
    assert np.array_equal(list(sp_reprs), ["0_Piano"])

    reprs = sp_reprs["0_Piano"][0]
    assert reprs.shape == (5,)
    assert reprs[0].shape == (16, 128)
    assert reprs[1].shape == (16, 12)
    assert reprs[2].shape == (16,)
    assert reprs[3].shape == (16,)
    assert reprs[4].shape == (N_DESCRIPTORS,)

    pmid = load_midi_file(BOSKA_3_DRUMS_FILEPATH)
    sp_reprs = slice_midi(
        pmid,
        seg_size=1,
        resolution=4,
        n_beat_bars=4,
        binarize=True,
        min_seg_pitches=1,
        min_seg_beats=1,
    )
    assert np.array_equal(list(sp_reprs), ["0_Drums"])

    reprs = sp_reprs["0_Drums"][0]
    assert reprs.shape == (5,)
    assert reprs[0].shape == (16, 128)
    assert reprs[1].shape == (16, 12)
    assert reprs[2].shape == (16,)
    assert reprs[3].shape == (16,)
    assert reprs[4].shape == (N_DESCRIPTORS,)
