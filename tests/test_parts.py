import pytest
from rhythmic_relationships.parts import get_instrument_from_program, get_part_pairs


def test_get_instrument_from_program():
    assert get_instrument_from_program(0) == "Piano"
    assert get_instrument_from_program(6) == "Piano"
    assert get_instrument_from_program(8) == "Chromatic Percussion"
    assert get_instrument_from_program(10) == "Chromatic Percussion"
    assert get_instrument_from_program(127) == "Sound Effect"
    with pytest.raises(Exception):
        get_instrument_from_program(-1)


def test_get_part_pairs():
    parts = ["Melody", "Drums", "Bass"]
    assert get_part_pairs(parts) == [
        ("Drums", "Melody"),
        ("Drums", "Bass"),
        ("Bass", "Melody"),
    ]
