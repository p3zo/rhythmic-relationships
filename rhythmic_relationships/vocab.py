import itertools
import numpy as np

from rhythmic_relationships.parts import PARTS

# Both vocabularies reserve the same two ids so that a pad or a start token means the same thing
# whichever representation a model is trained on
PAD_IX = 0
START_IX = 1


def get_vocab(part):
    """Build the vocabulary used for onset rolls of a given part"""
    if part not in PARTS:
        raise ValueError(f"part must be one of {PARTS}")

    # Create a mapping from token to integer, including first any special tokens
    itot = {PAD_IX: "pad", START_IX: "start"}

    if part == "Drums":
        patterns = [
            "".join([str(j) for j in i])
            for i in list(itertools.product([0, 1], repeat=9))
        ]

        itot.update({ix + len(itot): p for ix, p in enumerate(patterns)})

        return itot

    if part == "Harmony":
        patterns = [
            "".join([str(j) for j in i])
            for i in list(itertools.product([0, 1], repeat=12))
        ]

        itot.update({ix + len(itot): p for ix, p in enumerate(patterns)})

        return itot

    # Standard 88-key piano range
    pitch_min = 21
    pitch_max = 108
    n_velocity_bins = 4

    pitches = list(range(pitch_min, pitch_max + 1))
    velocity_bins = list(range(n_velocity_bins))

    itot.update({len(itot): "rest"})
    itot.update(
        {
            ix + len(itot): i
            for ix, i in enumerate(list(itertools.product(pitches, velocity_bins)))
        }
    )

    return itot


def get_vocab_encoder_decoder(part):
    """Get an encoder/decoder for the vocabulary used for onset rolls of a given part"""
    if part not in PARTS:
        raise ValueError(f"part must be one of {PARTS}")

    itot = get_vocab(part)
    ttoi = {v: k for k, v in itot.items()}

    # encoder: takes a list of tokens, output a list of integers
    encode = lambda s: [ttoi[c] for c in s]

    # decoder: takes a list of integers, output a list of tokens
    decode = lambda l: [itot[i] for i in l]

    return encode, decode


def get_vocab_sizes():
    """Get the sizes of the vocabularies used for onset rolls"""
    return {part: len(get_vocab(part)) for part in PARTS}


def get_hits_vocab():
    return {PAD_IX: "pad", START_IX: "start", 2: 0, 3: 0.25, 4: 0.5, 5: 0.75, 6: 1.0}


def get_hits_block_tokens(block_size):
    """Build the vocabulary of `block_size`-step hit blocks.

    Each block is a string of single-step hits token ids. The first two entries are the all-`pad`
    and all-`start` blocks, so that they keep the ids they have as single-step tokens. The rest
    are every combination of single-step tokens that begins with a hit rather than a special.
    """
    blocks = [
        "".join([str(j) for j in i])
        for i in itertools.product(get_hits_vocab().keys(), repeat=block_size)
        if i[0] not in (PAD_IX, START_IX)
    ]
    return [str(PAD_IX) * block_size, str(START_IX) * block_size] + blocks


def get_hits_vocab_size(block_size):
    return len(get_hits_block_tokens(block_size))


def encode_hits(hits, n_bins):
    vel_bins = np.linspace(0, 1, n_bins + 1)
    tokenized = np.digitize(hits, vel_bins, right=True).tolist()
    # Offset past the pad and start tokens, which occupy the first ids
    return [i + START_IX + 1 for i in tokenized]


def decode_hits(tokenized_hits, block_size=1):
    hits_vocab = get_hits_vocab()
    if block_size == 1:
        return [hits_vocab[i] for i in tokenized_hits]

    tokens = get_hits_block_tokens(block_size)
    decoded = [tokens[i] for i in tokenized_hits]
    decoded_flat = list(itertools.chain(*[[int(i) for i in list(j)] for j in decoded]))
    return [hits_vocab[i] for i in decoded_flat]
