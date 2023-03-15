import itertools

# Program categories from the General MIDI Level 2 spec: https://en.wikipedia.org/wiki/General_MIDI_Level_2
# A category's key is the index of its first patch
PROGRAM_CATEGORIES = {
    1: "Piano",
    9: "Chromatic Percussion",
    17: "Organ",
    25: "Guitar",
    33: "Bass",
    41: "Orchestra Solo",
    49: "Orchestra Ensemble",
    57: "Brass",
    65: "Reed",
    73: "Wind",
    81: "Synth Lead",
    89: "Synth Pad",
    97: "Synth Sound FX",
    105: "Ethnic",
    113: "Percussive",
    121: "Sound Effect",
}

# NOTE: In the General MIDI spec, drums are on a separate MIDI channel
PARTS = ["Drums"] + list(PROGRAM_CATEGORIES.values())


def get_part_from_program(program):
    """Returns the part name associated with a given program number"""
    if program < 0 or program > 127:
        raise ValueError(
            f"Program number {program} is not in the valid range of [0, 127]"
        )
    return PROGRAM_CATEGORIES[[p for p in PROGRAM_CATEGORIES if p <= program + 1][-1]]


def get_program_from_part(part):
    """Returns the program number associated with a given part name"""
    if part == "Drums":
        return 0
    return list(PROGRAM_CATEGORIES)[list(PROGRAM_CATEGORIES.values()).index(part)]


def get_part_pairs(parts):
    return [
        i
        for i in itertools.permutations(parts, 2)
        if PARTS.index(i[0]) < PARTS.index(i[1])
    ]
