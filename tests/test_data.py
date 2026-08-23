import numpy as np
import pytest
import torch
from rhythmic_relationships.data import (
    PartDataset,
    PartPairDataset,
    get_roll_from_sequence,
    tokenize_hits,
    tokenize_roll,
    get_sequences,
    get_pair_sequences,
)
from rhythmic_relationships.vocab import (
    decode_hits,
    get_hits_vocab_size,
    get_vocab_encoder_decoder,
)
from rhythmtoolbox import DESCRIPTOR_NAMES
from torch.utils.data import DataLoader

DATASET_NAME = "babyslakh_20_1bar_4res"
batch_size = 1
PART_1 = "Drums"
PART_2 = "Bass"

N_DESCRIPTORS = len(DESCRIPTOR_NAMES)


def test_get_pair_sequences():
    p1 = [1, 3, 4]
    p2 = [1, 2, 2]
    context_len = 3

    # The encoder is conditioned on all of part 1, so X is the full source at every position of
    # the target, and Y is the growing target prefix
    X, Y = get_pair_sequences(p1, p2, context_len)
    assert X == [[1, 3, 4], [1, 3, 4], [1, 3, 4]]
    assert Y == [[1], [1, 2], [1, 2, 2]]


def test_get_pair_sequences_pads_the_target_to_the_context_length():
    pad_ix = 0
    X, Y = get_pair_sequences([1, 3, 4], [1, 2, 2], context_len=3, pad_ix=pad_ix)
    assert X == [[1, 3, 4], [1, 3, 4], [1, 3, 4]]
    assert Y == [[1, pad_ix, pad_ix], [1, 2, pad_ix], [1, 2, 2]]


def test_get_sequences():
    tokenized = [1, 2, 3, 4]
    ctx, tgt = get_sequences(tokenized, context_len=len(tokenized), pad_ix=0)

    assert ctx == [[0, 0, 0, 0], [1, 0, 0, 0], [1, 2, 0, 0], [1, 2, 3, 0]]
    assert tgt == [[1, 0, 0, 0], [1, 2, 0, 0], [1, 2, 3, 0], [1, 2, 3, 4]]


def get_expected_velocity_from_bin(vel_bin, n_bins):
    return int((vel_bin + 1) * (127 / n_bins))


def test_tokenize_hits():
    # Ids start at 2 because `pad` and `start` occupy 0 and 1
    hits = [1, 0, 0.25, 0, 0.5, 0, 0.5, 0]
    assert tokenize_hits(hits, block_size=1) == [6, 2, 3, 2, 4, 2, 4, 2]
    assert tokenize_hits(hits, block_size=2) == [32, 11, 18, 18]
    assert tokenize_hits(hits, block_size=4) == [1495, 816]


def test_tokenize_hits_ids_fit_the_vocab_size():
    """Every id `tokenize_hits` can emit must be addressable by a model of `get_hits_vocab_size`"""
    hits = [1, 0, 0.25, 0, 0.5, 0, 0.5, 0]
    for block_size in [1, 2, 4]:
        tokenized = tokenize_hits(hits, block_size=block_size)
        assert max(tokenized) < get_hits_vocab_size(block_size)

    # The largest id for a block size is the all-loudest block, which is the last block token
    for block_size in [1, 2, 4, 8]:
        loudest = [1.0] * block_size
        assert tokenize_hits(loudest, block_size=block_size) == [
            get_hits_vocab_size(block_size) - 1
        ]


def test_tokenize_hits_round_trips_through_decode_hits():
    hits = [1, 0, 0.25, 0, 0.5, 0, 0.5, 0]
    for block_size in [1, 2, 4, 8]:
        tokenized = tokenize_hits(hits, block_size=block_size)
        assert decode_hits(tokenized, block_size=block_size) == hits


def test_tokenize_roll_validates_shape_per_part():
    drum_roll = np.zeros((32, 9))
    chroma = np.zeros((32, 12))

    # The correct representation for each part is accepted
    assert len(tokenize_roll(drum_roll, part="Drums")) == 32
    assert len(tokenize_roll(chroma, part="Harmony")) == 32

    # The wrong one is rejected, and the message names the part that was asked for
    with pytest.raises(Exception, match="drum roll"):
        tokenize_roll(chroma, part="Drums")
    with pytest.raises(Exception, match="chroma"):
        tokenize_roll(drum_roll, part="Harmony")


def test_get_roll_from_sequence():
    tokens = ["start", "rest", (60, 3), (72, 1), "rest"]

    encode, _ = get_vocab_encoder_decoder("Melody")
    sequence = np.array(encode(tokens))
    roll = get_roll_from_sequence(sequence, part="Melody")
    assert roll.shape == (4, 128)
    assert roll[0].sum() == 0
    assert roll[3].sum() == 0

    nz10 = roll[1].nonzero()[0]
    assert len(nz10) == 1
    assert nz10[0] == 60
    assert roll[1][nz10[0]] == get_expected_velocity_from_bin(tokens[2][1], n_bins=4)

    nz20 = roll[2].nonzero()[0]
    assert len(nz20) == 1
    assert nz20[0] == 72
    assert roll[2][nz20[0]] == get_expected_velocity_from_bin(tokens[3][1], n_bins=4)

    assert roll[3].sum() == 0

    drum_seq = np.array([1, 2, 11, 101])
    roll = get_roll_from_sequence(drum_seq, part="Drums")
    assert roll.shape == (3, 128)
    assert roll[0].sum() == 0
    assert roll[1].nonzero()[0].tolist() == [47, 51]
    assert roll[2].nonzero()[0].tolist() == [42, 43, 50, 51]


def test_PartDataset(prepared_dataset):
    datasets_dir, DATASET_NAME = prepared_dataset

    with pytest.raises(Exception):
        PartDataset(DATASET_NAME, PART_1, "incorrect-descriptor-name")

    data = PartDataset(DATASET_NAME, PART_1, "descriptors", datasets_dir=datasets_dir, tokenize_rolls=False)
    loader = DataLoader(data, batch_size=batch_size, shuffle=True)
    x = next(iter(loader))
    assert x.size() == torch.Size([batch_size, N_DESCRIPTORS])

    data = PartDataset(DATASET_NAME, PART_1, "roll", datasets_dir=datasets_dir, tokenize_rolls=False)
    loader = DataLoader(data, batch_size=batch_size, shuffle=True)
    x = next(iter(loader))
    assert x.size() == torch.Size([batch_size, 16, 128])


def test_PartDataset_tokenizes_only_roll_representations(prepared_dataset):
    datasets_dir, DATASET_NAME = prepared_dataset

    # A roll of the right shape for the part is tokenized to one id per tick
    data = PartDataset(DATASET_NAME, "Drums", "drum_roll", datasets_dir=datasets_dir)
    x = next(iter(DataLoader(data, batch_size=batch_size)))
    assert x.size() == torch.Size([batch_size, 16])
    assert x.dtype == torch.int64

    # Descriptors are a vector, so they come back untouched even with tokenizing on
    data = PartDataset(DATASET_NAME, "Drums", "descriptors", datasets_dir=datasets_dir)
    x = next(iter(DataLoader(data, batch_size=batch_size)))
    assert x.size() == torch.Size([batch_size, N_DESCRIPTORS])
    assert x.dtype == torch.float32

    # So is a pattern
    data = PartDataset(DATASET_NAME, "Drums", "pattern", datasets_dir=datasets_dir)
    x = next(iter(DataLoader(data, batch_size=batch_size)))
    assert x.size() == torch.Size([batch_size, 16])

    # A roll of the wrong shape for the part is still rejected
    data = PartDataset(DATASET_NAME, "Drums", "roll", datasets_dir=datasets_dir)
    with pytest.raises(Exception, match="drum roll"):
        next(iter(DataLoader(data, batch_size=batch_size)))


def test_PartPairDatasets(prepared_dataset):
    datasets_dir, DATASET_NAME = prepared_dataset

    with pytest.raises(Exception):
        PartPairDataset(
            DATASET_NAME, PART_1, PART_2, "desc", "incorrect-descriptor-name"
        )

    seq_len = 16

    data = PartPairDataset(
        DATASET_NAME,
        PART_1,
        PART_2,
        "roll",
        "roll",
        datasets_dir=datasets_dir,
        tokenize_rolls=False,
    )
    loader = DataLoader(data, batch_size=batch_size, shuffle=True)
    x, y = next(iter(loader))
    assert x.size() == torch.Size([batch_size, seq_len, 128])
    assert y.size() == torch.Size([batch_size, seq_len, 128])

    data = PartPairDataset(
        DATASET_NAME,
        PART_1,
        PART_2,
        "descriptors",
        "descriptors",
        datasets_dir=datasets_dir,
        tokenize_rolls=False,
    )
    loader = DataLoader(data, batch_size=batch_size, shuffle=True)
    x, y = next(iter(loader))
    assert x.size() == torch.Size([batch_size, N_DESCRIPTORS])
    assert y.size() == torch.Size([batch_size, N_DESCRIPTORS])

    data = PartPairDataset(
        DATASET_NAME,
        PART_1,
        PART_2,
        "roll",
        "descriptors",
        datasets_dir=datasets_dir,
        tokenize_rolls=False,
    )
    loader = DataLoader(data, batch_size=batch_size, shuffle=True)
    x, y = next(iter(loader))
    assert x.size() == torch.Size([batch_size, seq_len, 128])
    assert y.size() == torch.Size([batch_size, N_DESCRIPTORS])

    data = PartPairDataset(
        DATASET_NAME,
        PART_1,
        PART_2,
        "descriptors",
        "roll",
        datasets_dir=datasets_dir,
        tokenize_rolls=False,
    )
    loader = DataLoader(data, batch_size=batch_size, shuffle=True)
    x, y = next(iter(loader))
    assert x.size() == torch.Size([batch_size, N_DESCRIPTORS])
    assert y.size() == torch.Size([batch_size, seq_len, 128])

    data = PartPairDataset(
        DATASET_NAME,
        PART_1,
        PART_2,
        "hits",
        "hits",
        datasets_dir=datasets_dir,
        tokenize_rolls=False,
    )
    loader = DataLoader(data, batch_size=batch_size, shuffle=True)
    x, y = next(iter(loader))
    assert x.size() == torch.Size([batch_size, seq_len])
    assert y.size() == torch.Size([batch_size, seq_len])

    data = PartPairDataset(
        DATASET_NAME,
        PART_1,
        PART_2,
        "hits",
        "roll",
        datasets_dir=datasets_dir,
        tokenize_rolls=False,
    )
    loader = DataLoader(data, batch_size=batch_size, shuffle=True)
    x, y = next(iter(loader))
    assert x.size() == torch.Size([batch_size, seq_len])
    assert y.size() == torch.Size([batch_size, seq_len, 128])

    data = PartPairDataset(
        DATASET_NAME,
        PART_1,
        PART_2,
        "descriptors",
        "hits",
        datasets_dir=datasets_dir,
        tokenize_rolls=False,
    )
    loader = DataLoader(data, batch_size=batch_size, shuffle=True)
    x, y = next(iter(loader))
    assert x.size() == torch.Size([batch_size, N_DESCRIPTORS])
    assert y.size() == torch.Size([batch_size, seq_len])

    data = PartPairDataset(
        DATASET_NAME,
        PART_1,
        PART_2,
        "pattern",
        "pattern",
        datasets_dir=datasets_dir,
        tokenize_rolls=False,
    )
    loader = DataLoader(data, batch_size=batch_size, shuffle=True)
    x, y = next(iter(loader))
    assert x.size() == torch.Size([batch_size, seq_len])
    assert y.size() == torch.Size([batch_size, seq_len])

    data = PartPairDataset(
        DATASET_NAME,
        PART_1,
        PART_2,
        "roll",
        "pattern",
        datasets_dir=datasets_dir,
        tokenize_rolls=False,
    )
    loader = DataLoader(data, batch_size=batch_size, shuffle=True)
    x, y = next(iter(loader))
    assert x.size() == torch.Size([batch_size, seq_len, 128])
    assert y.size() == torch.Size([batch_size, seq_len])

    data = PartPairDataset(
        DATASET_NAME,
        PART_1,
        PART_2,
        "pattern",
        "chroma",
        datasets_dir=datasets_dir,
        tokenize_rolls=False,
    )
    loader = DataLoader(data, batch_size=batch_size, shuffle=True)
    x, y = next(iter(loader))
    assert x.size() == torch.Size([batch_size, seq_len])
    assert y.size() == torch.Size([batch_size, seq_len, 12])

    data = PartPairDataset(
        DATASET_NAME,
        PART_1,
        PART_2,
        "chroma",
        "roll",
        datasets_dir=datasets_dir,
        tokenize_rolls=False,
    )
    loader = DataLoader(data, batch_size=batch_size, shuffle=True)
    x, y = next(iter(loader))
    assert x.size() == torch.Size([batch_size, seq_len, 12])
    assert y.size() == torch.Size([batch_size, seq_len, 128])
