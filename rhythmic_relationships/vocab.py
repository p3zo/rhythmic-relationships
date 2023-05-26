import itertools
from rhythmic_relationships.parts import PARTS

# Do not change
START_TOKEN = 1


def get_vocab(part):
    if part not in PARTS:
        raise ValueError(f"part must be one of {PARTS}")

    # Create a mapping from token to integer, including first any special tokens
    itot = {1: "start"}

    if part == "Drums":
        patterns = [
            "".join([str(j) for j in i])
            for i in list(itertools.product([0, 1], repeat=9))
        ]
        # NOTE: add another 1 because 0 is reserved for pad ix
        itot.update({ix + len(itot) + 1: p for ix, p in enumerate(patterns)})

        return itot

    # Standard 88-key piano range
    pitch_min = 21
    pitch_max = 108
    n_velocity_bins = 4

    pitches = list(range(pitch_min, pitch_max + 1))
    velocity_bins = list(range(n_velocity_bins))

    itot.update({2: "rest"})
    # NOTE: add another 1 because 0 is reserved for pad ix
    itot.update(
        {
            ix + len(itot) + 1: i
            for ix, i in enumerate(list(itertools.product(pitches, velocity_bins)))
        }
    )

    return itot


def get_vocab_sizes():
    return {part: len(get_vocab(part)) for part in PARTS}


def get_vocab_encoder_decoder(part):
    if part not in PARTS:
        raise ValueError(f"part must be one of {PARTS}")

    itot = get_vocab(part)
    ttoi = {v: k for k, v in itot.items()}

    # encoder: takes a list of tokens, output a list of integers
    encode = lambda s: [ttoi[c] for c in s]

    # decoder: takes a list of integers, output a list of tokens
    decode = lambda l: [itot[i] for i in l]

    return encode, decode
