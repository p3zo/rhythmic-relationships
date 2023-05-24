import itertools
import numpy as np
from rhythmic_relationships.parts import PARTS

# TODO: this should be derived from a vocabulary
PAD_TOKEN = 1


def get_vocab(part):
    if part == "Drums":
        patterns = [
            "".join([str(j) for j in i])
            for i in list(itertools.product([0, 1], repeat=9))
        ]
        itot = {ix: p for ix, p in enumerate(patterns)}
        ttoi = {v: k for k, v in itot.items()}
        return itot, ttoi

    # Standard 88-key piano range
    pitch_min = 21
    pitch_max = 108
    n_velocity_bins = 4

    pitches = list(range(pitch_min, pitch_max + 1))
    velocity_bins = list(range(n_velocity_bins))

    # Create a mapping from token to integer, including first the rest and pad tokens as 0 and 1
    itot = {0: "rest", 1: "pad"}
    itot.update(
        {
            ix + 2: i
            for ix, i in enumerate(list(itertools.product(pitches, velocity_bins)))
        }
    )
    ttoi = {v: k for k, v in itot.items()}

    return itot, ttoi


def get_vocab_sizes():
    return {part: len(get_vocab(part)[0]) for part in PARTS}


def get_vocab_encoder_decoder(part):
    itot, ttoi = get_vocab(part)

    # encoder: takes a list of tokens, output a list of integers
    encode = lambda s: [ttoi[c] for c in s]

    # decoder: takes a list of integers, output a list of tokens
    decode = lambda l: [itot[i] for i in l]

    return encode, decode


def tokenize_roll(roll, part):
    encode, _ = get_vocab_encoder_decoder(part)

    if part == "Drums":
        if len(roll[0]) != 9:
            raise Exception("Representation must be drum roll for Drums part")
        binarized = (roll > 0).astype(int)
        patterns = ["".join(i.astype(str)) for i in binarized]
        return encode(patterns)

    # Will select the higher note in the case of polyphony
    pitches = roll.argmax(axis=1)

    # Truncate to the standard 88-key piano midi range
    for ix, p in enumerate(pitches):
        if p < 21:
            pitches[ix] = 0
        elif p > 108:
            pitches[ix] = 0

    velocities = roll.max(axis=1)
    velocity_bins = np.array([0.25, 0.5, 0.75, 1])
    binned_velocities = np.digitize(velocities, velocity_bins, right=True)

    pitch_velocities = list(zip(pitches, binned_velocities))

    tokens = []
    for p, v in pitch_velocities:
        if p == 0:
            tokens.append("rest")
            continue
        tokens.append((p, v))

    return encode(tokens)


def get_roll_from_sequence(seq, part):
    """Convert a monophonic sequence of pitches to a piano roll."""
    _, decode = get_vocab_encoder_decoder(part)

    roll = np.zeros((len(seq), 128), np.uint8)

    if part == "Drums":
        raise ValueError("Not yet implemented")

    decoded = decode(seq)

    velocity_bins = np.array([0.25, 0.5, 0.75, 1])
    velocity_bin_vals = (velocity_bins * 127).astype(int)

    for tick, token in enumerate(decoded):
        if isinstance(token, str):
            continue

        elif isinstance(token, tuple):
            pitch, velocity_bin = token
            roll[tick, pitch] = velocity_bin_vals[velocity_bin]

        else:
            raise ValueError("Invalid token type")

    return roll
