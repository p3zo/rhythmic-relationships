from scripts.prepare_data import get_part_from_program


def test_get_part_from_program():
    assert get_part_from_program(0) == "Piano"
    assert get_part_from_program(6) == "Piano"
    assert get_part_from_program(8) == "Chromatic Percussion"
    assert get_part_from_program(10) == "Chromatic Percussion"
    assert get_part_from_program(127) == "Sound Effect"
