from rhythmic_complements.parts import get_part_from_program, get_part_pairs


def test_get_part_from_program():
    assert get_part_from_program(0) == "Piano"
    assert get_part_from_program(6) == "Piano"
    assert get_part_from_program(8) == "Chromatic Percussion"
    assert get_part_from_program(10) == "Chromatic Percussion"
    assert get_part_from_program(127) == "Sound Effect"


def test_get_part_pairs():
    parts = ["Brass", "Piano", "Organ"]
    assert get_part_pairs(parts) == [
        ("Piano", "Brass"),
        ("Piano", "Organ"),
        ("Organ", "Brass"),
    ]
