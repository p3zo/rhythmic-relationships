import numpy as np
from rhythmtoolbox import DESCRIPTOR_NAMES
from rhythmic_relationships.io import load_midi_file, parse_bar_start_ticks
from rhythmic_relationships.representations import (
    get_representations,
    slice_midi,
)

N_DESCRIPTORS = len(DESCRIPTOR_NAMES)

BOSKA_3_PIANO_FILEPATH = "tests/midi/boska/3_piano.mid"
BOSKA_8_PIANO_FILEPATH = "tests/midi/boska/8_piano.mid"
BOSKA_9_PIANO_FILEPATH = "tests/midi/boska/9_piano.mid"
BOSKA_3_DRUMS_FILEPATH = "tests/midi/boska/3_drums.mid"
BOSKA_8_DRUMS_FILEPATH = "tests/midi/boska/8_drums.mid"
BOSKA_9_DRUMS_FILEPATH = "tests/midi/boska/9_drums.mid"
MUTED_GTR_FILEPATH = "tests/midi/muted_gtr.mid"


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


def test_get_respresentations():
    resolution = 4

    pmid = load_midi_file(BOSKA_3_PIANO_FILEPATH)
    bar_start_ticks, subdivisions = parse_bar_start_ticks(pmid, resolution)
    tracks = get_representations(pmid, subdivisions)
    track = tracks[0]
    boska_3_onsets = [1, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 1, 1]
    assert np.array_equal(
        (track["hits"] > 0).astype(int),
        boska_3_onsets,
    )
    assert np.array_equal(
        track["pattern"], [1, 0, 1, 1, 1, 2, 1, 0, 1, 1, 1, 0, 1, 2, 1, 1]
    )
    assert np.array_equal(
        track["chroma"],
        np.array(
            [
                [1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0],
                [1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0],
                [0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            ],
            dtype=np.uint8,
        ),
    )

    pmid = load_midi_file(BOSKA_3_DRUMS_FILEPATH)
    bar_start_ticks, subdivisions = parse_bar_start_ticks(pmid, resolution)
    tracks = get_representations(pmid, subdivisions)
    track = tracks[0]
    binary_hits = (track["hits"] > 0).astype(int)
    assert np.array_equal(binary_hits, boska_3_onsets)
    assert np.array_equal(track["pattern"], binary_hits)
    assert np.array_equal(
        track["chroma"],
        np.array(
            [
                [1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            ],
            dtype=np.uint8,
        ),
    )

    pmid = load_midi_file(BOSKA_8_PIANO_FILEPATH)
    bar_start_ticks, subdivisions = parse_bar_start_ticks(pmid, resolution)
    tracks = get_representations(pmid, subdivisions)
    track = tracks[0]
    boska_8_onsets = [1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 1]
    assert np.array_equal(
        (track["hits"] > 0).astype(int),
        boska_8_onsets,
    )
    assert np.array_equal(
        track["pattern"], [1, 2, 1, 1, 2, 2, 1, 2, 1, 1, 2, 2, 1, 2, 1, 1]
    )
    assert np.array_equal(
        track["chroma"],
        np.array(
            [
                [0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0],
                [0, 0, 2, 2, 0, 0, 0, 0, 0, 0, 2, 0],
                [0, 1, 2, 2, 0, 0, 0, 0, 0, 0, 2, 0],
                [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0],
                [0, 0, 2, 2, 0, 0, 0, 0, 0, 0, 2, 0],
                [0, 1, 2, 2, 0, 0, 0, 0, 0, 0, 2, 0],
                [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0],
                [0, 0, 2, 2, 0, 0, 0, 0, 0, 0, 2, 0],
                [0, 1, 0, 2, 0, 0, 0, 0, 0, 0, 2, 0],
                [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            ],
            dtype=np.uint8,
        ),
    )

    pmid = load_midi_file(BOSKA_8_DRUMS_FILEPATH)
    bar_start_ticks, subdivisions = parse_bar_start_ticks(pmid, resolution)
    tracks = get_representations(pmid, subdivisions)
    track = tracks[0]
    binary_hits = (track["hits"] > 0).astype(int)
    assert np.array_equal(binary_hits, boska_8_onsets)
    assert np.array_equal(track["pattern"], binary_hits)
    assert np.array_equal(
        track["chroma"],
        np.array(
            [
                [0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            ],
            dtype=np.uint8,
        ),
    )

    pmid = load_midi_file(BOSKA_9_PIANO_FILEPATH)
    bar_start_ticks, subdivisions = parse_bar_start_ticks(pmid, resolution)
    tracks = get_representations(pmid, subdivisions)
    track = tracks[0]
    boska_9_onsets = [1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 1, 0, 1, 0]
    assert np.array_equal(
        (track["hits"] > 0).astype(int),
        boska_9_onsets,
    )
    assert np.array_equal(
        track["pattern"], [1, 2, 1, 0, 1, 0, 1, 2, 1, 1, 2, 1, 1, 2, 1, 2]
    )
    assert np.array_equal(
        track["chroma"],
        np.array(
            [
                [1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                [2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                [2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0],
                [0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                [1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                [2, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 1, 1, 1, 0, 0, 1, 0, 0, 0, 1, 0],
                [0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 2, 0],
            ],
            dtype=np.uint8,
        ),
    )

    pmid = load_midi_file(BOSKA_9_DRUMS_FILEPATH)
    bar_start_ticks, subdivisions = parse_bar_start_ticks(pmid, resolution)
    tracks = get_representations(pmid, subdivisions)
    track = tracks[0]
    binary_hits = (track["hits"] > 0).astype(int)
    assert np.array_equal(binary_hits, boska_9_onsets)
    assert np.array_equal(track["pattern"], binary_hits)
    assert np.array_equal(
        track["chroma"],
        np.array(
            [
                [1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                [1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 1, 1, 1, 0, 0, 1, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            ],
            dtype=np.uint8,
        ),
    )
