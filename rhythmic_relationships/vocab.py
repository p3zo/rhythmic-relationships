import itertools
import numpy as np

from rhythmic_relationships.parts import PARTS


def get_vocab(part):
    if part not in PARTS:
        raise ValueError(f"part must be one of {PARTS}")

    # Create a mapping from token to integer, including first any special tokens
    itot = {0: "pad", 1: "start"}

    if part == "Drums":
        patterns = [
            "".join([str(j) for j in i])
            for i in list(itertools.product([0, 1], repeat=9))
        ]

        itot.update({ix + len(itot): p for ix, p in enumerate(patterns)})

        return itot

    # Standard 88-key piano range
    pitch_min = 21
    pitch_max = 108
    n_velocity_bins = 4

    pitches = list(range(pitch_min, pitch_max + 1))
    velocity_bins = list(range(n_velocity_bins))

    itot.update({2: "rest"})
    itot.update(
        {
            ix + len(itot): i
            for ix, i in enumerate(list(itertools.product(pitches, velocity_bins)))
        }
    )

    return itot


def get_vocab_sizes():
    return {part: len(get_vocab(part)) for part in PARTS}


def get_hits_vocab():
    return {0: "pad", 1: 0, 2: 0.25, 3: 0.5, 4: 0.75, 5: 1.0}


def get_hits_vocab_size(block_size):
    # TODO: make programatically
    sizes = {1: 6, 2: 31, 3: 1081, 4: 1399681}
    return sizes[block_size]


def encode_hits(hits, n_bins):
    vel_bins = np.linspace(0, 1, n_bins + 1)
    tokenized = np.digitize(hits, vel_bins, right=True).tolist()
    # Add 1 to account for padding token at ix 0
    return [i + 1 for i in tokenized]


def decode_hits(tokenized_hits, block_size=1):
    hits_vocab = get_hits_vocab()
    if block_size == 1:
        return [hits_vocab[i] for i in tokenized_hits]

    tokens = list(itertools.product(hits_vocab.keys(), repeat=block_size))
    tokens = ["".join([str(j) for j in i]) for i in tokens if i[0] != 0]
    tokens.insert(0, "0" * block_size)
    decoded = [tokens[i] for i in tokenized_hits]
    decoded_flat = list(itertools.chain(*[[int(i) for i in list(j)] for j in decoded]))
    return [hits_vocab[i] for i in decoded_flat]


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
