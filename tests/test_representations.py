import numpy as np
from rhythmtoolbox import DESCRIPTOR_NAMES
from rhythmic_relationships.io import load_midi_file, get_subdivisions
from rhythmic_relationships.representations import get_representations

N_DESCRIPTORS = len(DESCRIPTOR_NAMES)

BOSKA_3_PIANO_FILEPATH = "tests/midi/boska/3_piano.mid"
BOSKA_8_PIANO_FILEPATH = "tests/midi/boska/8_piano.mid"
BOSKA_9_PIANO_FILEPATH = "tests/midi/boska/9_piano.mid"
BOSKA_3_DRUMS_FILEPATH = "tests/midi/boska/3_drums.mid"
BOSKA_8_DRUMS_FILEPATH = "tests/midi/boska/8_drums.mid"
BOSKA_9_DRUMS_FILEPATH = "tests/midi/boska/9_drums.mid"
MUTED_GTR_FILEPATH = "tests/midi/muted_gtr.mid"


def test_get_respresentations():
    resolution = 4

    pmid = load_midi_file(BOSKA_3_PIANO_FILEPATH)
    subdivisions = get_subdivisions(pmid, resolution)
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
    subdivisions = get_subdivisions(pmid, resolution)
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
    subdivisions = get_subdivisions(pmid, resolution)
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
    subdivisions = get_subdivisions(pmid, resolution)
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
    subdivisions = get_subdivisions(pmid, resolution)
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
    subdivisions = get_subdivisions(pmid, resolution)
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
