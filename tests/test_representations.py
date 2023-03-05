import numpy as np
from rhythmic_complements.io import load_midi_file
from rhythmic_complements.representations import parse_representations, remap_velocities


def test_remap_velocities():
    assert np.allclose(remap_velocities(range(0, 128)), np.linspace(0, 1, 128))


def test_parse_respresentations():
    resolution = 4

    pmid = load_midi_file("input/boska/3.mid")
    tracks, _ = parse_representations(pmid, resolution=resolution, quantize=True)
    track = tracks[0]
    assert np.array_equal(
        (track["hits"] > 0).astype(int),
        [1, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 1, 1],
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

    pmid = load_midi_file("input/boska/8.mid")
    tracks, _ = parse_representations(pmid, resolution=resolution, quantize=True)
    track = tracks[0]
    assert np.array_equal(
        (track["hits"] > 0).astype(int),
        [1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 1],
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

    pmid = load_midi_file("input/boska/9.mid")
    tracks, _ = parse_representations(pmid, resolution=resolution, quantize=True)
    track = tracks[0]
    assert np.array_equal(
        (track["hits"] > 0).astype(int),
        [1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 1, 0, 1, 0],
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
